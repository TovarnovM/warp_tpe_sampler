from __future__ import annotations

from typing import Any, Tuple

import pytest


def _numpy_rng_state_repr(rng: Any) -> Tuple[str, Any]:
    """Return a stable, comparable representation of a numpy RNG state."""

    # numpy.random.RandomState
    if hasattr(rng, "get_state"):
        name, keys, pos, has_gauss, cached_gaussian = rng.get_state()
        # keys is an ndarray; compare bytes to avoid element-wise array comparison.
        return (
            "RandomState",
            (name, keys.tobytes(), int(pos), int(has_gauss), float(cached_gaussian)),
        )

    # numpy.random.Generator
    if hasattr(rng, "bit_generator"):
        return ("Generator", repr(rng.bit_generator.state))

    # Fallback
    return (type(rng).__name__, repr(rng))


def test_cached_tpe_sampler_clone_preserves_config_and_is_independent() -> None:
    pytest.importorskip("optuna")

    from warp_tpe_sampler import CachedTPESampler

    sampler = CachedTPESampler(
        n_startup_trials=3,
        n_ei_candidates=17,
        seed=0,
        multivariate=False,
        constant_liar=True,
        epsilon=0.123,
        epsilon2=0.456,
    )

    sampler._debug_set_marker("orig")

    clone = sampler.clone()
    assert isinstance(clone, CachedTPESampler)
    assert clone is not sampler

    # Preserve key configuration.
    assert clone._epsilon == sampler._epsilon
    assert clone._epsilon2 == sampler._epsilon2
    assert clone._n_startup_trials == sampler._n_startup_trials
    assert clone._n_ei_candidates == sampler._n_ei_candidates
    assert clone._constant_liar == sampler._constant_liar

    # Thread-local state must not be shared.
    assert clone._debug_get_marker() is None
    assert sampler._debug_get_marker() == "orig"


def test_cached_tpe_sampler_reseed_rng_changes_rng_state() -> None:
    pytest.importorskip("optuna")

    from warp_tpe_sampler import CachedTPESampler

    sampler = CachedTPESampler(n_startup_trials=1, seed=0)
    before = _numpy_rng_state_repr(sampler._rng.rng)
    sampler.reseed_rng()
    after = _numpy_rng_state_repr(sampler._rng.rng)
    assert after != before


def test_warp_tpe_sampler_clone_clones_policy_when_internal() -> None:
    optuna = pytest.importorskip("optuna")
    _ = optuna  # silence unused

    from warp_tpe_sampler import BudgetPolicyConfig, WarpTpeConfig, WarpTpeSampler

    cfg = WarpTpeConfig(
        n_startup_trials=0,
        seed=0,
        trial_attrs="none",
        budget_policy=BudgetPolicyConfig(epsilon=0.0, warmup_steps=0, seed=0),
        budget_policy_enabled=True,
    )

    sampler = WarpTpeSampler(cfg)
    assert sampler._policy is not None

    clone = sampler.clone()
    assert isinstance(clone, WarpTpeSampler)
    assert clone is not sampler
    assert clone.cfg == sampler.cfg

    assert clone._policy is not None
    assert clone._policy is not sampler._policy
    assert clone._policy.cfg == sampler._policy.cfg


def test_warp_tpe_sampler_clone_clones_explicit_policy() -> None:
    pytest.importorskip("optuna")

    from warp_tpe_sampler import WarpTpeConfig, WarpTpeSampler

    class StubPolicy:
        def __init__(self) -> None:
            self.v = 123

    cfg = WarpTpeConfig(n_startup_trials=0, seed=0, trial_attrs="none")
    pol = StubPolicy()
    sampler = WarpTpeSampler(cfg, policy=pol)

    clone = sampler.clone()
    assert clone is not sampler
    assert clone._policy is not None
    assert clone._policy is not pol
    assert getattr(clone._policy, "v") == 123


def test_warp_tpe_sampler_reseed_rng_also_reseeds_policy_rng() -> None:
    pytest.importorskip("optuna")

    from warp_tpe_sampler import BudgetPolicyConfig, WarpTpeConfig, WarpTpeSampler

    cfg = WarpTpeConfig(
        n_startup_trials=0,
        seed=0,
        trial_attrs="none",
        budget_policy=BudgetPolicyConfig(epsilon=0.5, warmup_steps=0, seed=0),
        budget_policy_enabled=True,
    )
    sampler = WarpTpeSampler(cfg)
    assert sampler._policy is not None

    # Capture both RNG states.
    before_tpe = _numpy_rng_state_repr(sampler._rng.rng)
    before_pol = sampler._policy._rng.getstate()

    sampler.reseed_rng()

    after_tpe = _numpy_rng_state_repr(sampler._rng.rng)
    after_pol = sampler._policy._rng.getstate()

    assert after_tpe != before_tpe
    assert after_pol != before_pol
