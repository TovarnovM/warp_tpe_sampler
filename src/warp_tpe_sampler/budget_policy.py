from __future__ import annotations

"""Budget policy for CachedTPESampler (single acceleration knob: FREEZE).

This module is intentionally *independent* from Optuna/Ray. It only decides:
  - whether to REFRESH (fetch + build) or FREEZE (reuse snapshot) or RANDOM
  - if REFRESH, what reduction size N to use (N >= n_min)

Objective
---------
Maximize quality subject to an average overhead budget:

    alpha = overhead / (bb_eff + overhead)

Where:
  - overhead = t_fetch + t_sampler
  - bb_eff = max(t_blackbox, t_min_sec)

Equivalently: overhead <= beta * bb_eff, where beta = alpha/(1-alpha).
We enforce this over time using a token bank (seconds):

    bank += beta*bb_eff - overhead

Bank is clamped to [-max_bank_s, +max_bank_s] to limit burstiness.

Decision priority (maximize quality)
-----------------------------------
1) epsilon-greedy triggers -> RANDOM
2) else if REFRESH(full) fits -> REFRESH(full)
3) else if REFRESH(reduced) fits -> REFRESH(reduced) with largest feasible N (>= n_min)
4) else if snapshot exists, FREEZE fits, and freeze_streak < K -> FREEZE
5) else -> RANDOM (budget fallback)

Reduction sizing
----------------
We treat sampler refresh cost as linear in N_used:

    t_sampler_refresh ~= refresh_per_trial * N_used

Reduction itself is assumed cheap; only N_used matters.

Freeze streak
-------------
At most max_freeze_streak consecutive FREEZE decisions are allowed.
Any non-FREEZE decision resets the streak.

Simulation
----------
Run:
  python -m pbt_funcs.budget_policy --steps 50000 --parallel-lam 50

"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import math
import random
import time


class Action(str, Enum):
    RANDOM = "RANDOM"
    REFRESH = "REFRESH"
    FREEZE = "FREEZE"


@dataclass(frozen=True)
class BudgetPolicyConfig:
    epsilon: float = 0.05
    alpha: float = 0.15
    t_min_sec: float = 0.5

    n_min: int = 64
    n_max: int = 10_000_000

    warmup_steps: int = 3
    max_freeze_streak: int = 5

    ema_halflife: float = 30.0
    max_bank_s: float = 30.0

    # <= 1.0 makes the policy more conservative under burstiness
    safety: float = 1.0

    seed: Optional[int] = None

    @property
    def beta(self) -> float:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        return self.alpha / (1.0 - self.alpha)

    def validate(self) -> None:
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0,1]")
        if self.t_min_sec <= 0:
            raise ValueError("t_min_sec must be > 0")
        if self.n_min <= 0:
            raise ValueError("n_min must be > 0")
        if self.n_max < self.n_min:
            raise ValueError("n_max must be >= n_min")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.max_freeze_streak < 0:
            raise ValueError("max_freeze_streak must be >= 0")
        if self.ema_halflife <= 0:
            raise ValueError("ema_halflife must be > 0")
        if self.max_bank_s <= 0:
            raise ValueError("max_bank_s must be > 0")
        if self.safety <= 0:
            raise ValueError("safety must be > 0")


@dataclass(frozen=True)
class Decision:
    action: Action
    # REFRESH only: keep at most reduce_n trials after reduction (None => use all)
    reduce_n: Optional[int]
    freeze_next: bool
    reason: str
    alerts: Tuple[str, ...]

    # diagnostics
    available_budget_s: float
    predicted_bb_eff_s: float
    predicted_overhead_s: float
    predicted_fetch_s: float
    predicted_sampler_s: float


@dataclass(frozen=True)
class Observation:
    action: Action
    t_blackbox_s: float
    t_fetch_s: float
    t_sampler_s: float
    n_trials_total: int
    n_trials_used: int
    ts: float = 0.0


@dataclass
class _State:
    step: int = 0
    bank_s: float = 0.0
    freeze_streak: int = 0

    bb_ema_s: Optional[float] = None
    fetch_per_trial_ema_s: Optional[float] = None
    refresh_per_trial_ema_s: Optional[float] = None
    freeze_cost_ema_s: Optional[float] = None


class BudgetedReductionPolicy:
    def __init__(self, cfg: BudgetPolicyConfig):
        cfg.validate()
        self.cfg = cfg
        self.state = _State()
        self._rng = random.Random(cfg.seed)

    # ----------------
    # public API
    # ----------------

    def decide(self, *, n_trials_total: int, has_snapshot: bool) -> Decision:
        st = self.state
        cfg = self.cfg
        st.step += 1

        available = self._available_budget_s()
        bb_pred = self._pred_bb_eff_s()

        # 1) epsilon
        if self._rng.random() < cfg.epsilon:
            return self._mk(Action.RANDOM, None, "epsilon", (), available, bb_pred, 0.0, 0.0, 0.0)

        n_total = int(max(0, n_trials_total))
        if n_total <= 0:
            return self._mk(Action.RANDOM, None, "no_trials", ("NO_TRIALS",), available, bb_pred, 0.0, 0.0, 0.0)

        # warmup: force REFRESH to learn slopes
        if st.step <= cfg.warmup_steps:
            pf = self._pred_fetch_s(n_total)
            ps = self._pred_refresh_s(n_total)
            return self._mk(Action.REFRESH, None, "warmup", (), available, bb_pred, pf + ps, pf, ps)

        safety = cfg.safety
        alerts: list[str] = []

        pf = self._pred_fetch_s(n_total)
        if pf > available:
            alerts.append("FETCH_TOO_EXPENSIVE_REDUCTION_WONT_HELP")

        # 2) REFRESH full
        ps_full = self._pred_refresh_s(n_total)
        p_full = pf + ps_full
        if p_full <= safety * available:
            return self._mk(Action.REFRESH, None, "refresh_full", tuple(alerts), available, bb_pred, p_full, pf, ps_full)

        # 3) REFRESH reduced (largest feasible N)
        n_used = self._max_n_used(n_total=n_total, available_s=available, pred_fetch_s=pf, safety=safety)
        if n_used is not None:
            ps = self._pred_refresh_s(n_used)
            p = pf + ps
            return self._mk(Action.REFRESH, n_used, "refresh_reduced", tuple(alerts), available, bb_pred, p, pf, ps)

        # 4) FREEZE
        if has_snapshot and st.freeze_streak < cfg.max_freeze_streak:
            ps_fz = self._pred_freeze_s()
            if ps_fz <= safety * available:
                return self._mk(Action.FREEZE, None, "freeze", tuple(alerts), available, bb_pred, ps_fz, 0.0, ps_fz)

        # 5) RANDOM fallback
        extra = ("FREEZE_STREAK_LIMIT",) if (has_snapshot and st.freeze_streak >= cfg.max_freeze_streak) else ()
        return self._mk(Action.RANDOM, None, "budget_fallback", tuple(alerts) + extra, available, bb_pred, 0.0, 0.0, 0.0)

    def observe(self, obs: Observation) -> None:
        cfg = self.cfg
        st = self.state

        bb_eff = max(max(0.0, float(obs.t_blackbox_s)), cfg.t_min_sec)
        t_fetch = max(0.0, float(obs.t_fetch_s))
        t_sampler = max(0.0, float(obs.t_sampler_s))
        overhead = t_fetch + t_sampler

        # bank
        st.bank_s += cfg.beta * bb_eff - overhead
        st.bank_s = max(-cfg.max_bank_s, min(cfg.max_bank_s, st.bank_s))

        # freeze streak
        st.freeze_streak = st.freeze_streak + 1 if obs.action == Action.FREEZE else 0

        # EWMAs
        self._ema("bb_ema_s", bb_eff)

        n_total = max(1, int(obs.n_trials_total))
        if t_fetch > 0:
            self._ema("fetch_per_trial_ema_s", t_fetch / float(n_total))

        if obs.action == Action.REFRESH:
            n_used = max(1, int(obs.n_trials_used))
            if t_sampler > 0:
                self._ema("refresh_per_trial_ema_s", t_sampler / float(n_used))

        if obs.action == Action.FREEZE and t_sampler > 0:
            self._ema("freeze_cost_ema_s", t_sampler)

    # ----------------
    # internals
    # ----------------

    def _pred_bb_eff_s(self) -> float:
        v = self.state.bb_ema_s
        return max(self.cfg.t_min_sec, float(v) if (v is not None and math.isfinite(v)) else self.cfg.t_min_sec)

    def _available_budget_s(self) -> float:
        # current spendable overhead for the next step
        return max(0.0, float(self.state.bank_s) + self.cfg.beta * self._pred_bb_eff_s())

    def _pred_fetch_s(self, n_total: int) -> float:
        per = self.state.fetch_per_trial_ema_s
        if per is None or not math.isfinite(per) or per <= 0:
            per = 1e-6
        return float(per) * float(max(0, n_total))

    def _pred_refresh_s(self, n_used: int) -> float:
        per = self.state.refresh_per_trial_ema_s
        if per is None or not math.isfinite(per) or per <= 0:
            per = 5e-6
        return float(per) * float(max(0, n_used))

    def _pred_freeze_s(self) -> float:
        v = self.state.freeze_cost_ema_s
        if v is None or not math.isfinite(v) or v <= 0:
            return 1e-4
        return float(v)

    def _max_n_used(self, *, n_total: int, available_s: float, pred_fetch_s: float, safety: float) -> Optional[int]:
        cfg = self.cfg
        per = self.state.refresh_per_trial_ema_s
        if per is None or not math.isfinite(per) or per <= 0:
            return None

        remaining = safety * float(available_s) - float(pred_fetch_s)
        if remaining <= 0:
            return None

        n_max_by_budget = int(remaining / float(per))
        n_used = min(n_total, n_max_by_budget, cfg.n_max)
        if n_used < cfg.n_min:
            return None
        return n_used

    def _ema(self, field: str, value: float) -> None:
        st = self.state
        old = getattr(st, field)
        hl = float(self.cfg.ema_halflife)
        w = 1.0 - math.exp(-math.log(2.0) / hl)
        if old is None:
            setattr(st, field, float(value))
        else:
            setattr(st, field, float(old) * (1.0 - w) + float(value) * w)

    @staticmethod
    def _mk(action: Action, reduce_n: Optional[int], reason: str, alerts: Tuple[str, ...],
            available: float, bb: float, po: float, pf: float, ps: float) -> Decision:
        return Decision(
            action=action,
            reduce_n=reduce_n,
            freeze_next=(action == Action.FREEZE),
            reason=reason,
            alerts=alerts,
            available_budget_s=float(available),
            predicted_bb_eff_s=float(bb),
            predicted_overhead_s=float(po),
            predicted_fetch_s=float(pf),
            predicted_sampler_s=float(ps),
        )


# ------------------------------
# Simulation / benchmark
# ------------------------------

def _poisson(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def _bench() -> None:
    import argparse
    from dataclasses import dataclass
    from collections import Counter

    @dataclass(frozen=True)
    class Scenario:
        name: str
        fetch_a: float
        fetch_b: float
        fetch_p: float
        refresh_c: float
        freeze_cost: float
        bb_base: float
        bb_jitter: float
        repeat_prob: float
        repeat_mult: float
        burst_prob: float = 0.0
        burst_mult: float = 1.0

    def make_scenarios() -> list[Scenario]:
        return [
            Scenario(
                name="fast_fetch",
                fetch_a=0.002, fetch_b=2e-7, fetch_p=1.0,
                refresh_c=8e-7,
                freeze_cost=2e-4,
                bb_base=0.30, bb_jitter=0.05,
                repeat_prob=0.05, repeat_mult=2.5,
            ),
            Scenario(
                name="slow_fetch",
                fetch_a=0.005, fetch_b=8e-7, fetch_p=1.0,
                refresh_c=8e-7,
                freeze_cost=2e-4,
                bb_base=0.30, bb_jitter=0.05,
                repeat_prob=0.05, repeat_mult=2.5,
            ),
            Scenario(
                name="bursty_fetch",
                fetch_a=0.002, fetch_b=2e-7, fetch_p=1.0,
                refresh_c=8e-7,
                freeze_cost=2e-4,
                bb_base=0.30, bb_jitter=0.05,
                repeat_prob=0.05, repeat_mult=2.5,
                burst_prob=0.02, burst_mult=25.0,
            ),
            Scenario(
                name="very_fast_blackbox",
                fetch_a=0.002, fetch_b=2e-7, fetch_p=1.0,
                refresh_c=8e-7,
                freeze_cost=2e-4,
                bb_base=0.03, bb_jitter=0.01,
                repeat_prob=0.02, repeat_mult=2.0,
            ),
        ]

    def fetch_time(s: Scenario, n_total: int, rng: random.Random) -> float:
        t = s.fetch_a + s.fetch_b * (float(n_total) ** s.fetch_p)
        if s.burst_prob > 0 and rng.random() < s.burst_prob:
            t *= s.burst_mult
        t *= 1.0 + 0.05 * (rng.random() - 0.5)
        return max(0.0, t)

    def bb_time(s: Scenario, step: int, rng: random.Random) -> float:
        t = s.bb_base + s.bb_jitter * math.sin(step / 17.0) + 0.01 * (rng.random() - 0.5)
        t = max(0.0, t)
        if rng.random() < s.repeat_prob:
            t *= s.repeat_mult
        return t

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--parallel-lam", type=float, default=50.0)
    parser.add_argument("--scenario", type=str, default="all")

    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--t-min", type=float, default=0.5)
    parser.add_argument("--n-min", type=int, default=64)
    parser.add_argument("--max-freeze", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--bank", type=float, default=30.0)
    parser.add_argument("--safety", type=float, default=1.0)
    parser.add_argument("--ema-halflife", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=200)
    args = parser.parse_args()

    cfg = BudgetPolicyConfig(
        epsilon=args.epsilon,
        alpha=args.alpha,
        t_min_sec=args.t_min,
        n_min=args.n_min,
        warmup_steps=args.warmup,
        max_freeze_streak=args.max_freeze,
        max_bank_s=args.bank,
        safety=args.safety,
        ema_halflife=args.ema_halflife,
        seed=args.seed,
    )

    scenarios = make_scenarios()
    if args.scenario != "all":
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            print("Unknown scenario. Available:")
            for s in make_scenarios():
                print("  -", s.name)
            raise SystemExit(2)

    for scn in scenarios:
        rng = random.Random(args.seed + 1337)
        pol = BudgetedReductionPolicy(cfg)

        n_total = 0
        has_snapshot = False

        total_over = 0.0
        total_bb = 0.0

        modes = Counter()
        alerts = Counter()
        reduce_sum = 0
        reduce_cnt = 0

        # windowed alpha
        w_over = 0.0
        w_tot = 0.0
        w_alphas: list[float] = []

        for step in range(args.steps):
            n_total += max(1, _poisson(args.parallel_lam, rng))

            dec = pol.decide(n_trials_total=n_total, has_snapshot=has_snapshot)
            modes[dec.action.value] += 1
            for a in dec.alerts:
                alerts[a] += 1

            t_bb = bb_time(scn, step, rng)
            bb_eff = max(t_bb, cfg.t_min_sec)

            if dec.action == Action.RANDOM:
                t_fetch = 0.0
                t_sampler = 0.0
                n_used = 0
            elif dec.action == Action.FREEZE:
                t_fetch = 0.0
                t_sampler = scn.freeze_cost
                n_used = 0
            else:
                t_fetch = fetch_time(scn, n_total, rng)
                n_used = dec.reduce_n if dec.reduce_n is not None else n_total
                n_used = max(cfg.n_min, min(n_used, n_total))
                t_sampler = scn.refresh_c * float(n_used)
                has_snapshot = True
                reduce_sum += n_used
                reduce_cnt += 1

            pol.observe(
                Observation(
                    action=dec.action,
                    t_blackbox_s=t_bb,
                    t_fetch_s=t_fetch,
                    t_sampler_s=t_sampler,
                    n_trials_total=n_total,
                    n_trials_used=n_used,
                    ts=time.time(),
                )
            )

            overhead = t_fetch + t_sampler
            total_over += overhead
            total_bb += bb_eff

            w_over += overhead
            w_tot += (bb_eff + overhead)
            if (step + 1) % args.window == 0:
                w_alphas.append((w_over / w_tot) if w_tot > 0 else 0.0)
                w_over = 0.0
                w_tot = 0.0

        achieved = total_over / (total_bb + total_over) if (total_bb + total_over) > 0 else 0.0
        avg_reduce = (reduce_sum / reduce_cnt) if reduce_cnt else 0.0

        p95 = 0.0
        if w_alphas:
            xs = sorted(w_alphas)
            idx = int(0.95 * (len(xs) - 1))
            p95 = xs[max(0, min(idx, len(xs) - 1))]

        print(f"\nScenario: {scn.name}")
        print(f"  alpha_target        : {cfg.alpha:.3f}")
        print(f"  alpha_achieved      : {achieved:.3f}")
        print(f"  alpha_p95_window    : {p95:.3f} (window={args.window})")
        print(f"  final_n_total       : {n_total}")
        print(f"  final_bank_s        : {pol.state.bank_s:.3f}")
        print(f"  freeze_streak_final : {pol.state.freeze_streak}")
        print(f"  avg_reduce_n        : {avg_reduce:.1f}")
        if reduce_cnt:
            print(f"  avg_reduce_ratio    : {avg_reduce / n_total:.6f}")
        print(f"  modes               : {dict(modes)}")
        if alerts:
            print(f"  alerts              : {dict(alerts)}")


if __name__ == "__main__":
    _bench()
