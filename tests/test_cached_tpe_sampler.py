# tests/test_cached_tpe_sampler.py
from __future__ import annotations

import threading
from typing import Any, Dict, Tuple

import pytest


def _nonstartup(n_trials: int, n_startup: int) -> int:
    # Trials 0..n_trials-1; after n_startup trials are finished,
    # remaining trials are non-startup (TPE-capable).
    return max(0, n_trials - n_startup)


def test_split_is_computed_once_per_nonstartup_trial_for_independent_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = pytest.importorskip("optuna")

    from warp_tpe_sampler import CachedTPESampler
    import warp_tpe_sampler.cached_tpe_sampler as mod

    split_calls = {"n": 0}
    orig_split = mod._split_trials

    def wrapped_split(*args: Any, **kwargs: Any):
        split_calls["n"] += 1
        return orig_split(*args, **kwargs)

    monkeypatch.setattr(mod, "_split_trials", wrapped_split)

    n_startup = 1
    n_trials = 4

    sampler = CachedTPESampler(
        n_startup_trials=n_startup, seed=0, multivariate=False, constant_liar=False
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        trial.suggest_float("y", 0.0, 1.0)
        trial.suggest_float("z", 0.0, 1.0)
        return 0.0

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    assert split_calls["n"] == _nonstartup(n_trials, n_startup)


def test_mpe_pair_is_cached_within_same_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    optuna = pytest.importorskip("optuna")
    from optuna.distributions import FloatDistribution

    from warp_tpe_sampler import CachedTPESampler

    sampler = CachedTPESampler(
        n_startup_trials=1, seed=0, multivariate=False, constant_liar=False
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(
        lambda tr: (tr.suggest_float("x", 0.0, 1.0) - 0.3) ** 2, n_trials=1, n_jobs=1
    )

    t = study.ask()
    frozen = study._storage.get_trial(t._trial_id)

    build_calls = {"n": 0}
    orig_build = sampler._build_parzen_estimator

    def wrapped_build(*args: Any, **kwargs: Any):
        build_calls["n"] += 1
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(sampler, "_build_parzen_estimator", wrapped_build)

    search_space = {"x": FloatDistribution(0.0, 1.0)}

    sampler._get_or_build_mpe_pair(study, frozen, search_space)
    sampler._get_or_build_mpe_pair(study, frozen, search_space)

    assert build_calls["n"] == 2  # below + above once


def test_snapshot_refresh_clears_mpe_cache_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = pytest.importorskip("optuna")
    from optuna.distributions import FloatDistribution

    from warp_tpe_sampler import CachedTPESampler

    sampler = CachedTPESampler(
        n_startup_trials=1, seed=0, multivariate=False, constant_liar=False
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(
        lambda tr: (tr.suggest_float("x", 0.0, 1.0) - 0.1) ** 2, n_trials=1, n_jobs=1
    )

    search_space = {"x": FloatDistribution(0.0, 1.0)}

    build_calls = {"n": 0}
    orig_build = sampler._build_parzen_estimator

    def wrapped_build(*args: Any, **kwargs: Any):
        build_calls["n"] += 1
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(sampler, "_build_parzen_estimator", wrapped_build)

    # Trial A: build once
    ta = study.ask()
    fa = study._storage.get_trial(ta._trial_id)
    sampler._get_or_build_mpe_pair(study, fa, search_space)
    assert build_calls["n"] == 2

    # Trial B: default refresh => clear mpe => rebuild
    tb = study.ask()
    fb = study._storage.get_trial(tb._trial_id)
    sampler._get_or_build_mpe_pair(study, fb, search_space)
    assert build_calls["n"] == 4


def test_freeze_next_trial_reuses_snapshot_and_mpe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = pytest.importorskip("optuna")
    from optuna.distributions import FloatDistribution

    from warp_tpe_sampler import CachedTPESampler

    sampler = CachedTPESampler(
        n_startup_trials=1, seed=0, multivariate=False, constant_liar=True
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(
        lambda tr: (tr.suggest_float("x", 0.0, 1.0) - 0.2) ** 2, n_trials=1, n_jobs=1
    )

    search_space = {"x": FloatDistribution(0.0, 1.0)}

    build_calls = {"n": 0}
    orig_build = sampler._build_parzen_estimator

    def wrapped_build(*args: Any, **kwargs: Any):
        build_calls["n"] += 1
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(sampler, "_build_parzen_estimator", wrapped_build)

    # Trial A
    ta = study.ask()
    fa = study._storage.get_trial(ta._trial_id)
    sampler._get_or_build_mpe_pair(study, fa, search_space)
    assert build_calls["n"] == 2

    # Freeze for next trial (same thread)
    sampler.use_cached_snapshot_once()

    # Trial B: should reuse snapshot and existing mpe (no rebuild)
    tb = study.ask()
    fb = study._storage.get_trial(tb._trial_id)
    sampler._get_or_build_mpe_pair(study, fb, search_space)
    assert build_calls["n"] == 2


def test_thread_local_isolation_of_flags_and_state() -> None:
    pytest.importorskip("optuna")

    from warp_tpe_sampler import CachedTPESampler

    sampler = CachedTPESampler(n_startup_trials=1, seed=0)

    results: Dict[str, Any] = {}

    def worker(name: str) -> None:
        sampler._debug_set_marker(name)
        sampler.use_cached_snapshot_once()
        results[name] = {"marker": sampler._debug_get_marker()}

    t1 = threading.Thread(target=worker, args=("t1",))
    t2 = threading.Thread(target=worker, args=("t2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["t1"]["marker"] == "t1"
    assert results["t2"]["marker"] == "t2"
    assert sampler._debug_get_marker() is None


# ---------------------------
# epsilon2 / below2 mechanism tests
# ---------------------------


def test_epsilon2_disabled_keeps_split_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = pytest.importorskip("optuna")

    import warp_tpe_sampler.cached_tpe_sampler as mod
    from warp_tpe_sampler import CachedTPESampler

    def fake_split(study, trials, n_below, constraints_enabled):
        # Deterministic split for test; ignore n_below.
        if len(trials) <= 1:
            return list(trials), []
        return list(trials[:1]), list(trials[1:])

    monkeypatch.setattr(mod, "_split_trials", fake_split)

    sampler = CachedTPESampler(
        n_startup_trials=0,
        seed=0,
        multivariate=False,
        constant_liar=False,
        epsilon2=0.0,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(lambda tr: float(tr.number), n_trials=5, n_jobs=1)

    t = study.ask()
    frozen = study._storage.get_trial(t._trial_id)
    snap = sampler._ensure_snapshot(study, frozen)

    trials_all = study._get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
        use_cache=True,
    )
    expected_below, expected_above = fake_split(study, trials_all, 1, False)

    assert [tt.number for tt in snap.below_trials] == [
        tt.number for tt in expected_below
    ]
    assert [tt.number for tt in snap.above_trials] == [
        tt.number for tt in expected_above
    ]
    assert snap.eps2_applied is False


def test_epsilon2_always_applies_and_preserves_partition_invariants() -> None:
    optuna = pytest.importorskip("optuna")

    import warp_tpe_sampler.cached_tpe_sampler as mod
    from warp_tpe_sampler import CachedTPESampler

    # Fixed gamma for predictable below size.
    sampler = CachedTPESampler(
        n_startup_trials=0,
        seed=0,
        multivariate=False,
        constant_liar=False,
        epsilon2=1.0,
        gamma=lambda n: min(2, n),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Deterministic objective values by trial number.
    study.optimize(lambda tr: float(tr.number), n_trials=10, n_jobs=1)

    t = study.ask()
    frozen = study._storage.get_trial(t._trial_id)
    snap = sampler._ensure_snapshot(study, frozen)

    assert snap.eps2_applied is True

    # Recompute base split exactly as in sampler._refresh_snapshot.
    reduced = study._get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
        use_cache=True,
    )
    n_finished_reduced = len(reduced)
    base_below, base_above = mod._split_trials(
        study,
        reduced,
        sampler._gamma(n_finished_reduced),
        sampler._constraints_func is not None,
    )

    below2_nums = {tt.number for tt in snap.below_trials}
    above2_nums = {tt.number for tt in snap.above_trials}
    base_below_nums = {tt.number for tt in base_below}
    base_above_nums = {tt.number for tt in base_above}

    # Invariants
    assert below2_nums.isdisjoint(above2_nums)
    assert below2_nums.union(above2_nums) == base_below_nums.union(base_above_nums)
    assert len(below2_nums) == len(base_below)
    assert below2_nums.issubset(base_above_nums)
    assert base_below_nums.issubset(above2_nums)

    # Should be a *different* set than the original below by construction (below2 ⊆ above).
    assert below2_nums != base_below_nums

    # Exact recomposition per spec: above2 = base_below ∪ (base_above \ below2)
    expected_above2 = base_below_nums.union(base_above_nums - below2_nums)
    assert above2_nums == expected_above2


def test_epsilon2_gating_not_enough_above_keeps_original_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = pytest.importorskip("optuna")

    import warp_tpe_sampler.cached_tpe_sampler as mod
    from warp_tpe_sampler import CachedTPESampler

    def fake_split(study, trials, n_below, constraints_enabled):
        # All trials go to below, above is empty => epsilon2 can't apply.
        return list(trials), []

    monkeypatch.setattr(mod, "_split_trials", fake_split)

    sampler = CachedTPESampler(
        n_startup_trials=0,
        seed=0,
        multivariate=False,
        constant_liar=False,
        epsilon2=1.0,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(lambda tr: float(tr.number), n_trials=1, n_jobs=1)

    t = study.ask()
    frozen = study._storage.get_trial(t._trial_id)
    snap = sampler._ensure_snapshot(study, frozen)

    assert snap.eps2_applied is False
    assert [tt.number for tt in snap.above_trials] == []
    assert [tt.number for tt in snap.below_trials] == [0]


# ---------------------------
# Hierarchical space matrix tests: multivariate/group
# ---------------------------


def _hier_objective(trial) -> float:
    # Always present param(s)
    x = trial.suggest_float("x", 0.0, 1.0)

    # Define-by-run hierarchical branching (deterministic by trial number)
    if trial.number % 2 == 0:
        y = trial.suggest_float("y", 0.0, 1.0)
        return (x - 0.2) ** 2 + (y - 0.8) ** 2
    else:
        z = trial.suggest_float("z", 0.0, 1.0)
        return (x - 0.7) ** 2 + (z - 0.1) ** 2


@pytest.mark.parametrize(
    "multivariate,group",
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_hierarchical_space_split_once_per_nonstartup_trial_all_modes(
    monkeypatch: pytest.MonkeyPatch, multivariate: bool, group: bool
) -> None:
    optuna = pytest.importorskip("optuna")
    import warp_tpe_sampler.cached_tpe_sampler as mod
    from warp_tpe_sampler import CachedTPESampler

    if group and not multivariate:
        pytest.skip("Optuna requires multivariate=True for group=True.")

    split_calls = {"n": 0}
    orig_split = mod._split_trials

    def wrapped_split(*args: Any, **kwargs: Any):
        split_calls["n"] += 1
        return orig_split(*args, **kwargs)

    monkeypatch.setattr(mod, "_split_trials", wrapped_split)

    n_startup = 2
    n_trials = 8

    sampler = CachedTPESampler(
        n_startup_trials=n_startup,
        seed=0,
        multivariate=multivariate,
        group=group,
        constant_liar=False,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_hier_objective, n_trials=n_trials, n_jobs=1)

    assert split_calls["n"] == _nonstartup(n_trials, n_startup)


def test_hierarchical_space_builds_distinct_mpe_pairs_for_distinct_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = pytest.importorskip("optuna")
    from optuna.distributions import FloatDistribution

    from warp_tpe_sampler.cached_tpe_sampler import CachedTPESampler

    sampler = CachedTPESampler(
        n_startup_trials=2, seed=0, multivariate=False, constant_liar=False
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Ensure we have both branches in history.
    study.optimize(_hier_objective, n_trials=6, n_jobs=1)

    # Use ask() to create a new trial context.
    t = study.ask()
    frozen = study._storage.get_trial(t._trial_id)

    build_calls = {"n": 0}
    orig_build = sampler._build_parzen_estimator

    def wrapped_build(*args: Any, **kwargs: Any):
        build_calls["n"] += 1
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(sampler, "_build_parzen_estimator", wrapped_build)

    ss_y = {"y": FloatDistribution(0.0, 1.0)}
    ss_z = {"z": FloatDistribution(0.0, 1.0)}

    # First time for each key => 2 estimators per key.
    sampler._get_or_build_mpe_pair(study, frozen, ss_y)
    sampler._get_or_build_mpe_pair(study, frozen, ss_z)
    assert build_calls["n"] == 4

    # Repeat => cache hit, no new builds.
    sampler._get_or_build_mpe_pair(study, frozen, ss_y)
    sampler._get_or_build_mpe_pair(study, frozen, ss_z)
    assert build_calls["n"] == 4


# ---------------------------
# Multi-objective smoke with hierarchical + group
# ---------------------------


def _hier_objective_mo(trial) -> Tuple[float, float]:
    x = trial.suggest_float("x", 0.0, 1.0)
    if trial.number % 2 == 0:
        y = trial.suggest_float("y", 0.0, 1.0)
        return ((x - 0.2) ** 2 + (y - 0.8) ** 2, (x - 0.9) ** 2)
    else:
        z = trial.suggest_float("z", 0.0, 1.0)
        return ((x - 0.7) ** 2 + (z - 0.1) ** 2, (x - 0.1) ** 2)


def test_multiobjective_hierarchical_group_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = pytest.importorskip("optuna")
    import warp_tpe_sampler.cached_tpe_sampler as mod
    from warp_tpe_sampler import CachedTPESampler

    split_calls = {"n": 0}
    orig_split = mod._split_trials

    def wrapped_split(*args: Any, **kwargs: Any):
        split_calls["n"] += 1
        return orig_split(*args, **kwargs)

    monkeypatch.setattr(mod, "_split_trials", wrapped_split)

    n_startup = 2
    n_trials = 7

    sampler = CachedTPESampler(
        n_startup_trials=n_startup,
        seed=0,
        multivariate=True,
        group=True,
        constant_liar=False,
    )
    study = optuna.create_study(directions=("minimize", "minimize"), sampler=sampler)
    study.optimize(_hier_objective_mo, n_trials=n_trials, n_jobs=1)

    assert split_calls["n"] == _nonstartup(n_trials, n_startup)


# ---------------------------
# Timing stats tests
# ---------------------------


def test_timing_stats_are_recorded_and_nonnegative() -> None:
    optuna = pytest.importorskip("optuna")

    from warp_tpe_sampler import CachedTPESampler

    sampler = CachedTPESampler(
        n_startup_trials=1, seed=0, multivariate=True, group=False, constant_liar=False
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        y = trial.suggest_float("y", 0.0, 1.0)
        return (x - 0.2) ** 2 + (y - 0.8) ** 2

    sampler.reset_timing_stats()
    study.optimize(objective, n_trials=5, n_jobs=1)

    stats = sampler.get_timing_stats()
    # Fetch must happen at least for startup gating and/or snapshot.
    assert stats["fetch_trials_n"] > 0
    assert stats["fetch_trials_s"] >= 0.0

    # Past startup, we should have at least one split and at least one draw.
    assert stats["split_trials_n"] >= 1
    assert stats["split_trials_s"] >= 0.0
    assert stats["draw_point_n"] >= 1
    assert stats["draw_point_s"] >= 0.0


def test_timing_stats_freeze_avoids_refresh_and_rebuild() -> None:
    optuna = pytest.importorskip("optuna")
    from optuna.distributions import FloatDistribution

    from warp_tpe_sampler import CachedTPESampler

    sampler = CachedTPESampler(
        n_startup_trials=1, seed=0, multivariate=False, constant_liar=True
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Seed history
    study.optimize(
        lambda tr: (tr.suggest_float("x", 0.0, 1.0) - 0.2) ** 2, n_trials=2, n_jobs=1
    )

    ss_x = {"x": FloatDistribution(0.0, 1.0)}

    sampler.reset_timing_stats()

    # Trial A: build snapshot + mpe
    ta = study.ask()
    fa = study._storage.get_trial(ta._trial_id)
    sampler._get_or_build_mpe_pair(study, fa, ss_x)
    s1 = sampler.get_timing_stats()

    assert s1["snapshot_refresh_n"] >= 1
    assert s1["build_mpe_pairs_n"] >= 1

    # Freeze for next trial
    sampler.use_cached_snapshot_once()

    # Trial B: reuse snapshot+mpe (no refresh, no build)
    tb = study.ask()
    fb = study._storage.get_trial(tb._trial_id)
    sampler._get_or_build_mpe_pair(study, fb, ss_x)
    s2 = sampler.get_timing_stats()

    assert s2["snapshot_refresh_n"] == s1["snapshot_refresh_n"]
    assert s2["build_mpe_pairs_n"] == s1["build_mpe_pairs_n"]
    assert s2["mpe_cache_hits"] >= s1["mpe_cache_hits"] + 1


# ---------------------------
# epsilon (fast random point) tests
# ---------------------------


def test_epsilon_always_random_avoids_snapshot_and_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = pytest.importorskip("optuna")

    import warp_tpe_sampler.cached_tpe_sampler as mod
    from warp_tpe_sampler import CachedTPESampler

    split_calls = {"n": 0}
    orig_split = mod._split_trials

    def wrapped_split(*args: Any, **kwargs: Any):
        split_calls["n"] += 1
        return orig_split(*args, **kwargs)

    monkeypatch.setattr(mod, "_split_trials", wrapped_split)

    sampler = CachedTPESampler(
        n_startup_trials=1,
        seed=0,
        multivariate=False,
        constant_liar=False,
        epsilon=1.0,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial) -> float:
        # Multiple params to ensure sampler is exercised.
        trial.suggest_float("x", 0.0, 1.0)
        trial.suggest_float("y", 0.0, 1.0)
        trial.suggest_float("z", 0.0, 1.0)
        return 0.0

    sampler.reset_timing_stats()
    study.optimize(objective, n_trials=6, n_jobs=1)
    stats = sampler.get_timing_stats()

    # When epsilon=1.0, all non-startup trials should be random and must not
    # build a snapshot or split below/above.
    assert split_calls["n"] == 0
    assert stats["snapshot_refresh_n"] == 0
    assert stats["build_mpe_pairs_n"] == 0


def test_use_random_once_applies_to_next_trial_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = pytest.importorskip("optuna")

    import warp_tpe_sampler.cached_tpe_sampler as mod
    from warp_tpe_sampler import CachedTPESampler

    split_calls = {"n": 0}
    orig_split = mod._split_trials

    def wrapped_split(*args: Any, **kwargs: Any):
        split_calls["n"] += 1
        return orig_split(*args, **kwargs)

    monkeypatch.setattr(mod, "_split_trials", wrapped_split)

    sampler = CachedTPESampler(
        n_startup_trials=1,
        seed=0,
        multivariate=False,
        constant_liar=False,
        epsilon=0.0,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # One startup trial.
    study.optimize(
        lambda tr: float(tr.suggest_float("x", 0.0, 1.0)), n_trials=1, n_jobs=1
    )

    sampler.reset_timing_stats()
    split_calls["n"] = 0

    # Force next trial to be random (fast path).
    sampler.use_random_once()

    # Trial A (random) + Trial B (TPE) => split must run exactly once.
    study.optimize(
        lambda tr: float(tr.suggest_float("x", 0.0, 1.0)), n_trials=2, n_jobs=1
    )

    assert split_calls["n"] == 1
