from __future__ import annotations

from typing import Any, Optional

import pytest


def _mk_dec(action, reduce_n: Optional[int] = None, reason: str = "test"):
    from warp_tpe_sampler import Decision

    return Decision(
        action=action,
        reduce_n=reduce_n,
        freeze_next=(action.value == "freeze"),
        reason=str(reason),
        alerts=(),
        available_budget_s=0.0,
        predicted_bb_eff_s=0.0,
        predicted_overhead_s=0.0,
        predicted_fetch_s=0.0,
        predicted_sampler_s=0.0,
    )


class _StubPolicy:
    """A minimal policy stub for deterministic unit tests."""

    def __init__(self, decisions: list[Any]):
        self._decisions = list(decisions)
        self._i = 0
        self.observations: list[Any] = []

    def decide(self, *, n_trials_total: int, has_snapshot: bool):  # noqa: ANN001
        if not self._decisions:
            raise RuntimeError("No scripted decisions")
        if self._i >= len(self._decisions):
            return self._decisions[-1]
        d = self._decisions[self._i]
        self._i += 1
        return d

    def observe(self, obs):  # noqa: ANN001
        self.observations.append(obs)


def test_budget_random_does_not_increment_internal_epsilon_counter() -> None:
    optuna = pytest.importorskip("optuna")

    from warp_tpe_sampler import Action
    from warp_tpe_sampler import WarpTpeConfig, WarpTpeSampler

    policy = _StubPolicy(
        [
            _mk_dec(Action.RANDOM, None, reason="epsilon"),
            _mk_dec(Action.RANDOM, None, reason="epsilon"),
            _mk_dec(Action.RANDOM, None, reason="epsilon"),
        ]
    )

    sampler = WarpTpeSampler(
        WarpTpeConfig(n_startup_trials=0, seed=0, trial_attrs="none"),
        policy=policy,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def obj(tr):
        tr.suggest_float("x", 0.0, 1.0)
        return 0.0

    sampler.reset_timing_stats()
    study.optimize(obj, n_trials=3, n_jobs=1)
    stats = sampler.get_timing_stats(aggregate=False)

    assert stats["random_applied_n"] >= 1
    assert stats["epsilon_applied_n"] == 0


def test_budget_reduce_n_forwarded_to_reduce_trials_via_n_keep() -> None:
    optuna = pytest.importorskip("optuna")

    from warp_tpe_sampler import Action
    from warp_tpe_sampler import WarpTpeConfig, WarpTpeSampler

    policy = _StubPolicy(
        [
            _mk_dec(Action.RANDOM, None, reason="startup"),
            _mk_dec(Action.REFRESH, 3, reason="refresh_reduced"),
            _mk_dec(Action.REFRESH, 3, reason="refresh_reduced"),
            _mk_dec(Action.REFRESH, 3, reason="refresh_reduced"),
        ]
    )

    sampler = WarpTpeSampler(
        WarpTpeConfig(
            n_startup_trials=0,
            seed=0,
            reduce_kind="last_n",
            trial_attrs="none",
        ),
        policy=policy,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    last_stats = []

    def obj(tr):
        tr.suggest_float("x", 0.0, 1.0)
        return float(tr.number)

    def cb(study_, trial_):  # noqa: ANN001
        st = sampler.get_last_trial_stats()
        if st is not None:
            last_stats.append(st)

    study.optimize(obj, n_trials=4, n_jobs=1, callbacks=[cb])

    assert last_stats
    assert last_stats[-1]["action"] == "REFRESH"
    assert 1 <= int(last_stats[-1]["n_trials_used"]) <= 3


def test_random_trial_does_not_break_following_freeze_snapshot() -> None:
    optuna = pytest.importorskip("optuna")

    from warp_tpe_sampler import Action
    from warp_tpe_sampler import WarpTpeConfig, WarpTpeSampler

    policy = _StubPolicy(
        [
            _mk_dec(Action.RANDOM, None, reason="startup"),
            _mk_dec(Action.REFRESH, 5, reason="refresh_full"),
            _mk_dec(Action.RANDOM, None, reason="epsilon"),
            _mk_dec(Action.FREEZE, None, reason="freeze"),
        ]
    )

    sampler = WarpTpeSampler(
        WarpTpeConfig(n_startup_trials=0, seed=0, trial_attrs="none"),
        policy=policy,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    per = []

    def obj(tr):
        tr.suggest_float("x", 0.0, 1.0)
        return 0.0

    def cb(study_, trial_):  # noqa: ANN001
        st = sampler.get_last_trial_stats()
        if st is not None:
            per.append(st)

    study.optimize(obj, n_trials=4, n_jobs=1, callbacks=[cb])

    assert len(per) == 4
    assert per[3]["action"] == "FREEZE"
    assert per[3]["snapshot_refreshed"] is False


def test_warp_tpe_config_epsilon_overrides_budget_policy_epsilon() -> None:
    """WarpTpeConfig.epsilon must override BudgetPolicyConfig.epsilon.

    Rationale: epsilon is now promoted to the WarpTpeConfig top-level to avoid a split
    source-of-truth between the sampler config and the embedded budget policy config.
    """

    optuna = pytest.importorskip("optuna")

    from warp_tpe_sampler import BudgetPolicyConfig
    from warp_tpe_sampler import WarpTpeConfig, WarpTpeSampler

    sampler = WarpTpeSampler(
        WarpTpeConfig(
            n_startup_trials=0,
            seed=0,
            trial_attrs="none",
            # New top-level epsilon.
            epsilon=1.0,
            # Intentionally conflicting value: should be ignored/overridden.
            budget_policy=BudgetPolicyConfig(epsilon=0.0, warmup_steps=0, seed=0),
            budget_policy_enabled=True,
        )
    )

    study = optuna.create_study(direction="minimize", sampler=sampler)

    def obj(tr):
        tr.suggest_float("x", 0.0, 1.0)
        return 0.0

    study.optimize(obj, n_trials=3, n_jobs=1)

    st = sampler.get_last_trial_stats()
    assert st is not None
    assert st["reason"] == "epsilon"

    counts = sampler.get_action_counts()
    assert counts["epsilon"] >= 1


def test_warp_tpe_config_alpha_overrides_budget_policy_alpha() -> None:
    """WarpTpeConfig.alpha must override BudgetPolicyConfig.alpha when policy is embedded."""

    optuna = pytest.importorskip("optuna")

    from warp_tpe_sampler import BudgetPolicyConfig
    from warp_tpe_sampler import WarpTpeConfig, WarpTpeSampler

    cfg = WarpTpeConfig(
        n_startup_trials=0,
        seed=0,
        trial_attrs="none",
        alpha=0.10,
        budget_policy=BudgetPolicyConfig(alpha=0.33, warmup_steps=0, seed=0),
        budget_policy_enabled=True,
    )

    sampler = WarpTpeSampler(cfg)
    assert sampler._policy is not None
    assert sampler._policy.cfg.alpha == pytest.approx(0.10)


def test_warp_tpe_sampler_alpha_param_overrides_warp_tpe_config_alpha() -> None:
    """WarpTpeSampler(alpha=...) must have highest precedence."""

    pytest.importorskip("optuna")

    from warp_tpe_sampler import BudgetPolicyConfig
    from warp_tpe_sampler import WarpTpeConfig, WarpTpeSampler

    cfg = WarpTpeConfig(
        n_startup_trials=0,
        seed=0,
        trial_attrs="none",
        alpha=0.20,
        budget_policy=BudgetPolicyConfig(alpha=0.33, warmup_steps=0, seed=0),
        budget_policy_enabled=True,
    )

    sampler = WarpTpeSampler(cfg, alpha=0.05)
    assert sampler._policy is not None
    assert sampler._policy.cfg.alpha == pytest.approx(0.05)
