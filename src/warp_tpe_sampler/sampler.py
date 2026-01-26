from __future__ import annotations

"""WarpTpeSampler.

This sampler unifies CachedTPESampler with a per-worker budget policy.

Design constraints
------------------
* Intended for n_jobs == 1 (single-thread / single-process) usage.
* The only source of RANDOM trials should be:
    - budget policy (Action.RANDOM)
    - Optuna startup behavior (n_startup_trials)
  Internal CachedTPESampler epsilon is forcibly disabled to avoid ambiguity.
* Reduction size N can be driven by the budget policy via the `n_keep` argument
  forwarded into reduce_trials().
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Literal, Optional

try:
    import optuna
    from optuna.trial import FrozenTrial, TrialState
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "WarpTpeSampler requires optuna to be installed and importable."
    ) from e

from .cached_tpe_sampler import CachedTPESampler
from .budget_policy import (
    Action,
    BudgetPolicyConfig,
    BudgetedReductionPolicy,
    Decision,
    Observation,
)


TrialAttrsMode = Literal["none", "basic", "full"]
ReduceKind = Literal["last_n", "tail_plus_random"]


def _gamma_from_cfg(cfg: "WarpTpeConfig"):
    """Return Optuna-compatible gamma(n) callable based on cfg.gamma_mode."""

    def _default_gamma(n: int) -> int:
        return min(int(0.1 * n) + 1, 25)

    def _hyperopt_gamma(n: int) -> int:
        return min(int(0.25 * math.sqrt(max(n, 1))) + 1, 25)

    if cfg.gamma_mode == "default":
        return _default_gamma
    if cfg.gamma_mode == "hyperopt":
        return _hyperopt_gamma

    frac = float(cfg.gamma_frac)
    cap = int(cfg.gamma_cap)

    def _gamma(n: int) -> int:
        return max(1, min(cap, int(math.ceil(frac * max(n, 1)))))

    return _gamma


def _weights_from_cfg(cfg: "WarpTpeConfig"):
    """Return Optuna-compatible weights(n) callable based on cfg.weights_mode."""

    mode = cfg.weights_mode
    p = float(cfg.weights_power)

    def _flat(n: int) -> list[float]:
        return [1.0] * int(max(0, n))

    def _power(n: int) -> list[float]:
        n = int(max(0, n))
        if n <= 0:
            return []
        return [((i + 1) / n) ** p for i in range(n)]

    if mode == "flat":
        return _flat
    # "default" and "hyperopt" are approximated with a recency-weighted power law.
    return _power


@dataclass(frozen=True)
class WarpTpeConfig:
    # --- TPE params (mirrors start_worker.TPEConfig/CachedTPEConfig) ---
    seed: Optional[int] = None
    n_startup_trials: int = 20
    n_ei_candidates: int = 24

    gamma_mode: Literal["default", "hyperopt", "fraction"] = "default"
    gamma_frac: float = 0.20
    gamma_cap: int = 25

    weights_mode: Literal["default", "hyperopt", "flat", "power"] = "default"
    weights_power: float = 1.0

    consider_endpoints: bool = True
    consider_magic_clip: bool = True
    prior_weight: float = 1.0
    multivariate: bool = True
    group: bool = True
    warn_independent_sampling: bool = False
    constant_liar: bool = False

    # --- exploration knobs ---
    # epsilon-greedy probability to force RANDOM via the embedded budget policy.
    # (Promoted from BudgetPolicyConfig to avoid nested config churn.)
    epsilon: float = 0.05

    # Optional override for BudgetPolicyConfig.alpha.
    # If set, it is applied when constructing the embedded BudgetedReductionPolicy.
    alpha: Optional[float] = None

    epsilon2: float = 0.0

    # --- reduction (used on refresh) ---
    reduce_kind: ReduceKind = "last_n"
    reduce_tail_frac: float = 0.50
    reduce_seed: Optional[int] = None

    # --- embedded budget policy ---
    budget_policy: Optional[BudgetPolicyConfig] = None
    budget_policy_enabled: bool = True

    # --- telemetry persistence to trial.user_attrs ---
    trial_attrs: TrialAttrsMode = "basic"


def _mk_decision(action: Action, reduce_n: Optional[int], reason: str) -> Decision:
    """Conservative fallback Decision object for error paths."""
    return Decision(
        action=action,
        reduce_n=reduce_n,
        freeze_next=(action == Action.FREEZE),
        reason=str(reason),
        alerts=(),
        available_budget_s=0.0,
        predicted_bb_eff_s=0.0,
        predicted_overhead_s=0.0,
        predicted_fetch_s=0.0,
        predicted_sampler_s=0.0,
    )


def _safe_set_trial_user_attr(
    study: "optuna.study.Study", trial: FrozenTrial, key: str, value: Any
) -> None:
    """Set trial user attr from sampler hooks (FrozenTrial is immutable)."""
    try:
        storage = getattr(study, "_storage", None)
        trial_id = getattr(trial, "_trial_id", None)
        if storage is None or trial_id is None:
            return
        storage.set_trial_user_attr(int(trial_id), str(key), value)
    except Exception:
        return


class WarpTpeSampler(CachedTPESampler):
    """CachedTPESampler + embedded BudgetedReductionPolicy (n_jobs=1 intended)."""

    def __init__(
        self,
        cfg: WarpTpeConfig,
        *,
        policy: Optional[Any] = None,
        alpha: Optional[float] = None,
    ) -> None:
        self.cfg = cfg

        # Build reduction function.
        reduce_trials = self._make_reduce_trials(cfg)

        # IMPORTANT: disable internal epsilon to avoid ambiguity.
        super().__init__(
            n_startup_trials=int(cfg.n_startup_trials),
            n_ei_candidates=int(cfg.n_ei_candidates),
            seed=cfg.seed,
            gamma=_gamma_from_cfg(cfg),
            weights=_weights_from_cfg(cfg),
            multivariate=bool(cfg.multivariate),
            group=bool(cfg.group),
            constant_liar=bool(cfg.constant_liar),
            consider_endpoints=bool(cfg.consider_endpoints),
            consider_magic_clip=bool(cfg.consider_magic_clip),
            prior_weight=float(cfg.prior_weight),
            warn_independent_sampling=bool(cfg.warn_independent_sampling),
            epsilon=0.0,
            epsilon2=float(cfg.epsilon2),
            reduce_trials=reduce_trials,
        )

        # Optional embedded budget policy.
        if policy is not None:
            self._policy = policy
        elif cfg.budget_policy is not None and cfg.budget_policy_enabled:
            # epsilon is configured at the WarpTpeConfig top-level; override any nested value.
            # alpha is optionally overridden by (highest precedence first):
            #   1) WarpTpeSampler(alpha=...)
            #   2) WarpTpeConfig.alpha
            #   3) BudgetPolicyConfig.alpha
            try:
                from dataclasses import replace as _dc_replace

                overrides: dict[str, Any] = {"epsilon": float(cfg.epsilon)}
                if alpha is not None:
                    overrides["alpha"] = float(alpha)
                elif cfg.alpha is not None:
                    overrides["alpha"] = float(cfg.alpha)

                pol_cfg = _dc_replace(cfg.budget_policy, **overrides)
            except Exception:
                pol_cfg = cfg.budget_policy
            self._policy = BudgetedReductionPolicy(pol_cfg)
        else:
            self._policy = None

        self._prev_before_ts: Optional[float] = None
        self._trial_ctx: Optional[dict[str, Any]] = None

        self._last_trial_stats: Optional[Dict[str, Any]] = None
        self._counts: Dict[str, int] = {
            "epsilon": 0,
            "epsilon2": 0,
            "cached": 0,
            "refresh": 0,
            "random": 0,
        }

        # Optional explicit blackbox time override (seconds) for the *next* observe().
        self._override_blackbox_s: Optional[float] = None

    # ---------------- public API ----------------

    def set_last_blackbox_time_s(self, t_blackbox_s: float) -> None:
        """Override blackbox time estimate for the next Observation()."""
        try:
            self._override_blackbox_s = max(0.0, float(t_blackbox_s))
        except Exception:
            self._override_blackbox_s = None

    def get_last_trial_stats(self) -> Optional[Dict[str, Any]]:
        return None if self._last_trial_stats is None else dict(self._last_trial_stats)

    def get_action_counts(self) -> Dict[str, int]:
        return dict(self._counts)

    # ---------------- Optuna hooks ----------------

    def before_trial(self, study: "optuna.study.Study", trial: FrozenTrial) -> None:
        st = self._tls_state()

        # Snapshot presence BEFORE base.before_trial() potentially drops it.
        has_snapshot = st.snapshot is not None

        # Count finished trials cheaply (Optuna cache is ok).
        try:
            done = study._get_trials(  # type: ignore[attr-defined]
                deepcopy=False,
                states=(
                    optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.PRUNED,
                ),
                use_cache=True,
            )
            n_total = int(len(done))
        except Exception:
            n_total = 0

        startup = bool(n_total < int(self.cfg.n_startup_trials))

        decision: Optional[Decision] = None
        if self._policy is not None:
            try:
                decision = self._policy.decide(
                    n_trials_total=n_total, has_snapshot=bool(has_snapshot)
                )
            except Exception:
                decision = _mk_decision(Action.RANDOM, None, "policy_error")

        # Apply decision to base sampler flags.
        if decision is not None and not startup:
            if decision.action == Action.FREEZE:
                # If no snapshot exists, FREEZE is meaningless; fall back to REFRESH.
                if has_snapshot:
                    self.use_cached_snapshot_once()
                else:
                    decision = _mk_decision(
                        Action.REFRESH, decision.reduce_n, "freeze_without_snapshot"
                    )
            if decision.action == Action.RANDOM:
                self.use_random_once()

            # Budget-driven reduction size: forwarded into reduce_trials() during refresh.
            st.reduce_n = (
                decision.reduce_n if decision.action == Action.REFRESH else None
            )
        else:
            st.reduce_n = None

        now = time.perf_counter()
        dt_since_prev_before = None
        if self._prev_before_ts is not None:
            dt_since_prev_before = float(now - self._prev_before_ts)
        self._prev_before_ts = now

        # Capture per-trial context for after_trial() accounting.
        self._trial_ctx = {
            "trial_number": int(trial.number),
            "trial_id": getattr(trial, "_trial_id", None),
            "t_before": now,
            "dt_since_prev_before": dt_since_prev_before,
            "timing_before": self.get_timing_stats(aggregate=False),
            "n_trials_total": n_total,
            "startup": startup,
            "decision": decision,
        }

        # Let the base sampler set up TLS flags and internal bookkeeping.
        super().before_trial(study, trial)

    def after_trial(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[list[float]],
    ) -> None:
        # CachedTPESampler.after_trial clears TLS flags (random_this_trial/freeze_this_trial).
        # Capture them before delegating.
        st = self._tls_state()
        if self._trial_ctx is not None and int(
            self._trial_ctx.get("trial_number", -1)
        ) == int(trial.number):
            self._trial_ctx["was_random"] = bool(st.random_this_trial)
            self._trial_ctx["was_freeze"] = bool(st.freeze_this_trial)
            self._trial_ctx["random_source"] = getattr(st, "random_source", None)

        try:
            super().after_trial(study, trial, state, values)
        finally:
            self._finalize_trial_stats(study, trial, state)

    # ---------------- internals ----------------

    def _make_reduce_trials(self, cfg: WarpTpeConfig):
        kind = cfg.reduce_kind
        tail_frac = float(cfg.reduce_tail_frac)
        tail_frac = max(0.0, min(1.0, tail_frac))
        seed0 = cfg.reduce_seed
        seed_fallback = int(cfg.seed or 0)

        def _reduce(
            trials: list[FrozenTrial],
            *,
            n_keep: Optional[int] = None,
            trial_number: int = 0,
            rng: Any = None,
        ) -> list[FrozenTrial]:
            if n_keep is None:
                return list(trials)
            n_keep_i = int(max(0, n_keep))
            if n_keep_i <= 0:
                return []
            if n_keep_i >= len(trials):
                return list(trials)

            # Always sort by trial.number for determinism.
            ts = sorted(trials, key=lambda t: int(t.number))

            if kind == "last_n":
                return ts[-n_keep_i:]

            # tail_plus_random
            n_tail = int(round(tail_frac * float(n_keep_i)))
            n_tail = max(1, min(n_keep_i, n_tail))
            n_rand = n_keep_i - n_tail

            tail = ts[-n_tail:]
            if n_rand <= 0:
                return tail

            head = ts[: max(0, len(ts) - n_tail)]
            if not head:
                return tail

            # Deterministic sampling; mix in trial_number.
            s = int(seed0 if seed0 is not None else seed_fallback)
            s = int((s * 1_000_003 + int(trial_number)) & 0xFFFFFFFF)

            import random as _random

            rr = _random.Random(s)
            pick = list(head) if n_rand >= len(head) else rr.sample(head, k=int(n_rand))

            out = list(pick) + list(tail)
            out.sort(key=lambda t: int(t.number))
            return out

        return _reduce

    @staticmethod
    def _diff_stats(before: Dict[str, Any], after: Dict[str, Any], key: str) -> float:
        try:
            return float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        except Exception:
            return 0.0

    def _finalize_trial_stats(
        self, study: "optuna.study.Study", trial: FrozenTrial, state: TrialState
    ) -> None:
        ctx = self._trial_ctx
        self._trial_ctx = None
        if not ctx or int(ctx.get("trial_number", -1)) != int(trial.number):
            return

        t_after = time.perf_counter()
        wall_s = float(t_after - float(ctx["t_before"]))

        timing_before: Dict[str, Any] = dict(ctx.get("timing_before") or {})
        timing_after: Dict[str, Any] = self.get_timing_stats(aggregate=False)

        d_fetch = self._diff_stats(timing_before, timing_after, "fetch_trials_s")
        d_reduce = self._diff_stats(timing_before, timing_after, "reduce_trials_s")
        d_split = self._diff_stats(timing_before, timing_after, "split_trials_s")
        d_build = self._diff_stats(timing_before, timing_after, "build_mpe_pairs_s")
        d_draw = self._diff_stats(timing_before, timing_after, "draw_point_s")
        d_refresh_n = int(
            self._diff_stats(timing_before, timing_after, "snapshot_refresh_n")
        )

        t_fetch_s = max(0.0, d_fetch)
        t_sampler_s = max(0.0, d_reduce + d_split + d_build + d_draw)
        overhead_s = t_fetch_s + t_sampler_s

        # Estimate blackbox time by subtracting sampler overhead from the wall time.
        t_bb = max(0.0, wall_s - overhead_s)
        if self._override_blackbox_s is not None:
            t_bb = float(self._override_blackbox_s)
            self._override_blackbox_s = None

        st = self._tls_state()
        n_used = 0
        try:
            if st.snapshot is not None:
                n_used = int(len(st.snapshot.trials_reduced))
        except Exception:
            n_used = 0

        # Determine effective action (TLS flags were captured before super().after_trial).
        was_random = bool(ctx.get("was_random", False))
        was_freeze = bool(ctx.get("was_freeze", False))
        if was_random:
            eff_action = Action.RANDOM
        elif was_freeze:
            eff_action = Action.FREEZE
        else:
            eff_action = Action.REFRESH

        decision: Optional[Decision] = ctx.get("decision")
        startup = bool(ctx.get("startup"))
        reason = (
            "startup"
            if startup
            else (decision.reason if decision is not None else "none")
        )

        snapshot_eps2 = (
            bool(getattr(st.snapshot, "eps2_applied", False))
            if st.snapshot is not None
            else False
        )

        self._counts["random"] += 1 if eff_action == Action.RANDOM else 0
        self._counts["refresh"] += 1 if d_refresh_n > 0 else 0
        self._counts["cached"] += 1 if eff_action == Action.FREEZE else 0
        self._counts["epsilon"] += (
            1 if (eff_action == Action.RANDOM and reason == "epsilon") else 0
        )
        self._counts["epsilon2"] += (
            1 if snapshot_eps2 and eff_action != Action.RANDOM else 0
        )

        # Observe for policy.
        if self._policy is not None and decision is not None and not startup:
            try:
                self._policy.observe(
                    Observation(
                        action=eff_action,
                        t_blackbox_s=t_bb,
                        t_fetch_s=t_fetch_s,
                        t_sampler_s=t_sampler_s,
                        n_trials_total=int(ctx.get("n_trials_total", 0)),
                        n_trials_used=int(n_used),
                        ts=time.time(),
                    )
                )
            except Exception:
                pass

        last = {
            "trial_number": int(trial.number),
            "state": str(getattr(state, "name", state)),
            "action": str(eff_action.value),
            "reason": str(reason),
            "reduce_n": None if decision is None else decision.reduce_n,
            "startup": startup,
            "wall_s": wall_s,
            "t_blackbox_s": t_bb,
            "t_fetch_s": t_fetch_s,
            "t_sampler_s": t_sampler_s,
            "snapshot_refreshed": bool(d_refresh_n > 0),
            "has_snapshot": bool(st.snapshot is not None),
            "eps2_active": bool(snapshot_eps2),
            "n_trials_total": int(ctx.get("n_trials_total", 0)),
            "n_trials_used": int(n_used),
            "dt_since_prev_before_s": ctx.get("dt_since_prev_before"),
            "timing_delta": {
                "fetch_trials_s": t_fetch_s,
                "sampler_s": t_sampler_s,
                "snapshot_refresh_n": int(d_refresh_n),
            },
        }
        self._last_trial_stats = dict(last)

        if self.cfg.trial_attrs != "none":
            _safe_set_trial_user_attr(study, trial, "warp.action", last["action"])
            _safe_set_trial_user_attr(study, trial, "warp.reason", last["reason"])
            _safe_set_trial_user_attr(study, trial, "warp.reduce_n", last["reduce_n"])
            _safe_set_trial_user_attr(
                study, trial, "warp.t_blackbox_s", float(last["t_blackbox_s"])
            )
            _safe_set_trial_user_attr(
                study, trial, "warp.t_fetch_s", float(last["t_fetch_s"])
            )
            _safe_set_trial_user_attr(
                study, trial, "warp.t_sampler_s", float(last["t_sampler_s"])
            )
            _safe_set_trial_user_attr(
                study, trial, "warp.eps2_active", bool(last["eps2_active"])
            )
            if self.cfg.trial_attrs == "full":
                _safe_set_trial_user_attr(
                    study, trial, "warp.n_trials_total", int(last["n_trials_total"])
                )
                _safe_set_trial_user_attr(
                    study, trial, "warp.n_trials_used", int(last["n_trials_used"])
                )
