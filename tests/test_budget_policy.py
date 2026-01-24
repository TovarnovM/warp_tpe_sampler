from __future__ import annotations

import math

from warp_tpe_sampler import (
    Action,
    BudgetPolicyConfig,
    BudgetedReductionPolicy,
    Observation,
)


def test_beta_equivalence():
    cfg = BudgetPolicyConfig(alpha=0.2)
    assert math.isclose(cfg.beta, 0.2 / 0.8)


def test_reduced_refresh_respects_n_min():
    cfg = BudgetPolicyConfig(
        alpha=0.2, epsilon=0.0, t_min_sec=1.0, n_min=100, warmup_steps=0, seed=0
    )
    pol = BudgetedReductionPolicy(cfg)

    # seed EWMAs
    pol.state.bb_ema_s = 10.0
    pol.state.fetch_per_trial_ema_s = 1e-6
    pol.state.refresh_per_trial_ema_s = 0.01
    pol.state.bank_s = 0.0

    d = pol.decide(n_trials_total=10000, has_snapshot=True)
    assert d.action == Action.REFRESH
    assert d.reduce_n is not None
    assert d.reduce_n >= cfg.n_min


def test_freeze_streak_limit_blocks_freeze():
    cfg = BudgetPolicyConfig(
        alpha=0.2, epsilon=0.0, warmup_steps=0, max_freeze_streak=2, seed=0
    )
    pol = BudgetedReductionPolicy(cfg)

    # force tiny budget so only freeze fits
    pol.state.bb_ema_s = 1.0
    pol.state.bank_s = 0.0
    pol.state.freeze_cost_ema_s = 1e-6
    pol.state.fetch_per_trial_ema_s = 1.0
    pol.state.refresh_per_trial_ema_s = 1.0

    d1 = pol.decide(n_trials_total=1000, has_snapshot=True)
    assert d1.action == Action.FREEZE
    pol.observe(
        Observation(
            action=Action.FREEZE,
            t_blackbox_s=1.0,
            t_fetch_s=0.0,
            t_sampler_s=1e-6,
            n_trials_total=1000,
            n_trials_used=0,
        )
    )

    d2 = pol.decide(n_trials_total=1000, has_snapshot=True)
    assert d2.action == Action.FREEZE
    pol.observe(
        Observation(
            action=Action.FREEZE,
            t_blackbox_s=1.0,
            t_fetch_s=0.0,
            t_sampler_s=1e-6,
            n_trials_total=1000,
            n_trials_used=0,
        )
    )

    d3 = pol.decide(n_trials_total=1000, has_snapshot=True)
    assert d3.action != Action.FREEZE
