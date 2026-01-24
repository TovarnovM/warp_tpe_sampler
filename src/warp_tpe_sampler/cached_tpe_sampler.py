from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

try:
    import optuna
    from optuna.distributions import BaseDistribution
    from optuna.samplers import TPESampler as _OptunaTPESampler
    from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
    from optuna.samplers._tpe.sampler import _split_trials
    from optuna.trial import FrozenTrial, TrialState
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "CachedTPESampler requires optuna to be installed and importable."
    ) from e


SearchSpaceKey = Tuple[str, ...]
"""Type of trials reduction function.

We keep this intentionally permissive to allow passing additional context without
breaking older callables.

Supported call patterns:
  - reduce(trials: list[FrozenTrial]) -> list[FrozenTrial]
  - reduce(trials: list[FrozenTrial], *, n_keep: int|None, trial_number: int, rng: random.Random) -> list[FrozenTrial]
"""

ReduceTrialsFunc = Callable[..., list[FrozenTrial]]


def _default_reduce_trials(trials: list[FrozenTrial], **_: Any) -> list[FrozenTrial]:
    return trials


def _make_search_space_key(search_space: Dict[str, BaseDistribution]) -> SearchSpaceKey:
    return tuple(sorted(search_space.keys()))


@dataclass
class _Snapshot:
    trials_reduced: list[FrozenTrial]
    below_trials: list[FrozenTrial]
    above_trials: list[FrozenTrial]
    finished_total_all: int  # COMPLETE+PRUNED count (startup gate)

    eps2_applied: bool = False


@dataclass
class TimingStats:
    # random_applied_n counts *all* fast-path RandomSampler trials, regardless of source.
    # epsilon_applied_n counts only those RandomSampler trials that were triggered by
    # the internal epsilon coin-flip (i.e. not by use_random_once).
    random_applied_n: int = 0

    epsilon_applied_n: int = 0
    eps2_applied_n: int = 0

    fetch_trials_n: int = 0
    fetch_trials_s: float = 0.0

    reduce_trials_n: int = 0
    reduce_trials_s: float = 0.0

    split_trials_n: int = 0
    split_trials_s: float = 0.0

    build_mpe_pairs_n: int = 0
    build_mpe_pairs_s: float = 0.0

    draw_point_n: int = 0
    draw_point_s: float = 0.0

    snapshot_refresh_n: int = 0

    mpe_cache_hits: int = 0
    mpe_cache_misses: int = 0

    def add_inplace(self, other: "TimingStats") -> None:
        self.random_applied_n += other.random_applied_n

        self.epsilon_applied_n += other.epsilon_applied_n
        self.eps2_applied_n += other.eps2_applied_n

        self.fetch_trials_n += other.fetch_trials_n
        self.fetch_trials_s += other.fetch_trials_s

        self.reduce_trials_n += other.reduce_trials_n
        self.reduce_trials_s += other.reduce_trials_s

        self.split_trials_n += other.split_trials_n
        self.split_trials_s += other.split_trials_s

        self.build_mpe_pairs_n += other.build_mpe_pairs_n
        self.build_mpe_pairs_s += other.build_mpe_pairs_s

        self.draw_point_n += other.draw_point_n
        self.draw_point_s += other.draw_point_s

        self.snapshot_refresh_n += other.snapshot_refresh_n

        self.mpe_cache_hits += other.mpe_cache_hits
        self.mpe_cache_misses += other.mpe_cache_misses

    def to_dict(self) -> Dict[str, Any]:
        return {
            "random_applied_n": self.random_applied_n,
            "epsilon_applied_n": self.epsilon_applied_n,
            "eps2_applied_n": self.eps2_applied_n,
            "fetch_trials_n": self.fetch_trials_n,
            "fetch_trials_s": self.fetch_trials_s,
            "reduce_trials_n": self.reduce_trials_n,
            "reduce_trials_s": self.reduce_trials_s,
            "split_trials_n": self.split_trials_n,
            "split_trials_s": self.split_trials_s,
            "build_mpe_pairs_n": self.build_mpe_pairs_n,
            "build_mpe_pairs_s": self.build_mpe_pairs_s,
            "draw_point_n": self.draw_point_n,
            "draw_point_s": self.draw_point_s,
            "snapshot_refresh_n": self.snapshot_refresh_n,
            "mpe_cache_hits": self.mpe_cache_hits,
            "mpe_cache_misses": self.mpe_cache_misses,
        }


@dataclass
class _TLSState:
    active_study_id: Optional[int] = None
    active_trial_number: Optional[int] = None

    freeze_next_trial_once: bool = False
    freeze_this_trial: bool = False

    random_next_trial_once: bool = False
    random_this_trial: bool = False
    # Optional diagnostic marker for the current trial.
    # - "forced"  : use_random_once()
    # - "epsilon" : internal epsilon coin-flip
    random_source: Optional[str] = None
    finished_count_cached: Optional[int] = None
    # Optional per-trial reduction size forwarded to reduce_trials().
    # Intended to be used by higher-level controllers (e.g. budget policy).
    reduce_n: Optional[int] = None
    snapshot: Optional[_Snapshot] = None
    mpe_cache: Dict[SearchSpaceKey, Tuple[_ParzenEstimator, _ParzenEstimator]] = field(
        default_factory=dict
    )

    timing: TimingStats = field(default_factory=TimingStats)

    # debug hook
    debug_marker: Optional[str] = None


class CachedTPESampler(_OptunaTPESampler):
    """
    CachedTPESampler

    Goals:
      - Thread-local snapshot cache: trials fetch (+ optional reduction) + below/above split.
      - Thread-local MPE cache: (mpe_below, mpe_above) per search_space_key, built from snapshot.
      - On snapshot refresh -> clear MPE cache.
      - One-shot flag: use_cached_snapshot_once() freezes snapshot+MPE for the *next* trial
        in the *current thread*.

    Intended trade-off (explicit):
      - constant_liar snapshot includes RUNNING and freezing also freezes RUNNING.
    """

    def __init__(
        self,
        *args: Any,
        reduce_trials: Optional[ReduceTrialsFunc] = None,
        epsilon: float = 0.0,
        epsilon2: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduce_trials: ReduceTrialsFunc = reduce_trials or _default_reduce_trials

        epsilon = float(epsilon)
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon!r}")
        self._epsilon = epsilon

        epsilon2 = float(epsilon2)
        if not (0.0 <= epsilon2 <= 1.0):
            raise ValueError(f"epsilon2 must be in [0, 1], got {epsilon2!r}")
        self._epsilon2 = epsilon2

        self._tls = threading.local()

        # Aggregated timings across threads (for benchmarks with n_jobs>1)
        self._agg_lock = threading.Lock()
        self._agg_timing = TimingStats()

    # ---------------- TLS ----------------
    def _tls_state(self) -> _TLSState:
        st = getattr(self._tls, "state", None)
        if st is None:
            st = _TLSState()
            self._tls.state = st
        return st

    def __getstate__(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("_tls", None)
        d.pop("_agg_lock", None)
        return d

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._tls = threading.local()
        self._agg_lock = threading.Lock()
        self._agg_timing = TimingStats()

    # ---------------- timing API ----------------
    def reset_timing_stats(self, *, aggregate: bool = True) -> None:
        st = self._tls_state()
        st.timing = TimingStats()
        if aggregate:
            with self._agg_lock:
                self._agg_timing = TimingStats()

    def get_timing_stats(self, *, aggregate: bool = True) -> Dict[str, Any]:
        if not aggregate:
            return self._tls_state().timing.to_dict()
        with self._agg_lock:
            return self._agg_timing.to_dict()

    def _record_stage(self, name: str, dt: float) -> None:
        st = self._tls_state()
        t = st.timing

        if name == "fetch_trials":
            t.fetch_trials_n += 1
            t.fetch_trials_s += dt
        elif name == "reduce_trials":
            t.reduce_trials_n += 1
            t.reduce_trials_s += dt
        elif name == "split_trials":
            t.split_trials_n += 1
            t.split_trials_s += dt
        elif name == "build_mpe_pairs":
            t.build_mpe_pairs_n += 1
            t.build_mpe_pairs_s += dt
        elif name == "draw_point":
            t.draw_point_n += 1
            t.draw_point_s += dt
        else:  # pragma: no cover
            return

        with self._agg_lock:
            a = self._agg_timing
            if name == "fetch_trials":
                a.fetch_trials_n += 1
                a.fetch_trials_s += dt
            elif name == "reduce_trials":
                a.reduce_trials_n += 1
                a.reduce_trials_s += dt
            elif name == "split_trials":
                a.split_trials_n += 1
                a.split_trials_s += dt
            elif name == "build_mpe_pairs":
                a.build_mpe_pairs_n += 1
                a.build_mpe_pairs_s += dt
            elif name == "draw_point":
                a.draw_point_n += 1
                a.draw_point_s += dt

    def _timed(self, name: str):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._record_stage(name, time.perf_counter() - t0)

        return _cm()

    # ---------------- control API ----------------
    def use_cached_snapshot_once(self) -> None:
        st = self._tls_state()
        st.freeze_next_trial_once = True

    def use_random_once(self) -> None:
        """Force the next trial in the current thread to be sampled by RandomSampler.

        This is a fast path: it bypasses snapshot refresh, trial splitting, and MPE construction.
        """
        st = self._tls_state()
        st.random_next_trial_once = True

    # ---------------- hooks ----------------
    def before_trial(self, study: "optuna.study.Study", trial: FrozenTrial) -> None:
        super().before_trial(study, trial)
        self._on_new_trial_context(study, trial)

    def after_trial(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        try:
            super().after_trial(study, trial, state, values)
        finally:
            self._tls_state().freeze_this_trial = False
            self._tls_state().random_this_trial = False
            self._tls_state().random_source = None
            # Per-trial reduction size is advisory; clear it after the trial.
            self._tls_state().reduce_n = None

    # ---------------- caching core ----------------
    def _on_new_trial_context(
        self, study: "optuna.study.Study", trial: FrozenTrial
    ) -> None:
        st = self._tls_state()
        study_id = getattr(study, "_study_id", id(study))

        is_new = (st.active_study_id != study_id) or (
            st.active_trial_number != trial.number
        )
        if not is_new:
            return

        st.active_study_id = study_id
        st.active_trial_number = trial.number
        coinflip_random = False

        if st.freeze_next_trial_once:
            st.freeze_this_trial = True
            st.freeze_next_trial_once = False
        else:
            st.freeze_this_trial = False

        if st.random_next_trial_once:
            st.random_this_trial = True
            st.random_source = "forced"
            st.random_next_trial_once = False
        else:
            coinflip_random = (self._epsilon > 0.0) and (
                float(self._rng.rng.random()) < self._epsilon
            )
            st.random_this_trial = coinflip_random
            st.random_source = "epsilon" if coinflip_random else None
        if st.random_this_trial:
            st.timing.random_applied_n += 1
            with self._agg_lock:
                self._agg_timing.random_applied_n += 1

        # epsilon_applied_n is reserved for *internal* epsilon coin-flips only.
        if coinflip_random:
            st.timing.epsilon_applied_n += 1
            with self._agg_lock:
                self._agg_timing.epsilon_applied_n += 1

        st.finished_count_cached = None
        # Important: do not drop the snapshot when the current trial is RANDOM.
        # This keeps the "random trial" fast-path compatible with a following
        # use_cached_snapshot_once() (FREEZE) decision.
        if not st.freeze_this_trial and not st.random_this_trial:
            st.snapshot = None
            st.mpe_cache.clear()

    def _sample_random_relative(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        # NOTE: RandomSampler returns external representations.
        params: Dict[str, Any] = {}
        for name, dist in search_space.items():
            params[name] = self._random_sampler.sample_independent(
                study, trial, name, dist
            )
        return params

    def _get_finished_count_all(
        self, study: "optuna.study.Study", trial: FrozenTrial
    ) -> int:
        self._on_new_trial_context(study, trial)
        st = self._tls_state()

        if st.snapshot is not None:
            st.finished_count_cached = st.snapshot.finished_total_all
            return st.snapshot.finished_total_all

        if st.finished_count_cached is not None:
            return st.finished_count_cached

        with self._timed("fetch_trials"):
            done = study._get_trials(
                deepcopy=False,
                states=(TrialState.COMPLETE, TrialState.PRUNED),
                use_cache=True,
            )
        st.finished_count_cached = len(done)
        return st.finished_count_cached

    def _ensure_snapshot(
        self, study: "optuna.study.Study", trial: FrozenTrial
    ) -> _Snapshot:
        self._on_new_trial_context(study, trial)
        st = self._tls_state()

        if st.snapshot is not None:
            return st.snapshot

        snap = self._refresh_snapshot(study, trial)
        st.snapshot = snap
        st.finished_count_cached = snap.finished_total_all
        st.mpe_cache.clear()
        return snap

    def _refresh_snapshot(
        self, study: "optuna.study.Study", trial: FrozenTrial
    ) -> _Snapshot:
        st = self._tls_state()
        st.timing.snapshot_refresh_n += 1
        with self._agg_lock:
            self._agg_timing.snapshot_refresh_n += 1

        if self._constant_liar:
            states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING)
        else:
            states = (TrialState.COMPLETE, TrialState.PRUNED)

        with self._timed("fetch_trials"):
            trials_all = study._get_trials(
                deepcopy=False, states=states, use_cache=True
            )

        finished_total_all = int(
            sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in trials_all)
        )

        if self._constant_liar:
            trials_all = [t for t in trials_all if t.number != trial.number]
        # Allow reduction policy to use more context (e.g. budget-provided n_keep).
        n_keep = st.reduce_n
        with self._timed("reduce_trials"):
            try:
                reduced = self._reduce_trials(
                    trials_all,
                    n_keep=n_keep,
                    trial_number=int(trial.number),
                    rng=self._rng.rng,
                )
            except TypeError:
                # Backward compatible: old reduce(trials) callables.
                reduced = self._reduce_trials(trials_all)
        if not isinstance(reduced, list):
            reduced = list(reduced)

        n_finished_reduced = int(
            sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in reduced)
        )

        with self._timed("split_trials"):
            below, above = _split_trials(
                study,
                reduced,
                self._gamma(n_finished_reduced),
                self._constraints_func is not None,
            )

        below, above, eps2_applied = self._maybe_apply_epsilon2_split(
            study, below, above
        )

        if eps2_applied:
            st.timing.eps2_applied_n += 1
            with self._agg_lock:
                self._agg_timing.eps2_applied_n += 1

        return _Snapshot(
            trials_reduced=reduced,
            below_trials=below,
            above_trials=above,
            finished_total_all=finished_total_all,
            eps2_applied=eps2_applied,
        )

    def _get_or_build_mpe_pair(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Tuple[_ParzenEstimator, _ParzenEstimator]:
        st = self._tls_state()
        snap = self._ensure_snapshot(study, trial)
        key = _make_search_space_key(search_space)

        if key in st.mpe_cache:
            st.timing.mpe_cache_hits += 1
            with self._agg_lock:
                self._agg_timing.mpe_cache_hits += 1
            return st.mpe_cache[key]

        st.timing.mpe_cache_misses += 1
        with self._agg_lock:
            self._agg_timing.mpe_cache_misses += 1

        with self._timed("build_mpe_pairs"):
            mpe_below = self._build_parzen_estimator(
                study, search_space, snap.below_trials, handle_below=True
            )
            mpe_above = self._build_parzen_estimator(
                study, search_space, snap.above_trials, handle_below=False
            )

        st.mpe_cache[key] = (mpe_below, mpe_above)
        return mpe_below, mpe_above

    # ---------------- epsilon2 / below2 mechanism ----------------
    def _weighted_sample_without_replacement(
        self,
        items: list[FrozenTrial],
        weights: list[float],
        k: int,
    ) -> list[FrozenTrial]:
        """Weighted sample without replacement using the sampler RNG.

        This is intentionally simple (O(k*m)) because it runs once per snapshot refresh.
        """
        if k <= 0:
            return []
        if k > len(items):
            raise ValueError("k must be <= len(items)")

        rng = self._rng.rng
        pool_items = list(items)
        pool_w = list(weights)
        chosen: list[FrozenTrial] = []

        for _ in range(k):
            total = float(sum(pool_w))
            if total <= 0.0:
                j = int(rng.random() * len(pool_items))
            else:
                r = float(rng.random()) * total
                acc = 0.0
                j = 0
                for j, w in enumerate(pool_w):
                    acc += float(w)
                    if r <= acc:
                        break
            chosen.append(pool_items.pop(j))
            pool_w.pop(j)

        return chosen

    def _maybe_apply_epsilon2_split(
        self,
        study: "optuna.study.Study",
        below: list[FrozenTrial],
        above: list[FrozenTrial],
    ) -> tuple[list[FrozenTrial], list[FrozenTrial], bool]:
        """Apply epsilon2/below2 mechanism.

        Spec:
          - below2 is sampled from above
          - |below2| == |below| (i.e. inherits gamma/split size)
          - above2 = below + (above - below2)

        Gating:
          - disabled when epsilon2 == 0
          - disabled for multi-objective studies
          - disabled when not enough eligible trials in above
        """
        if self._epsilon2 <= 0.0:
            return below, above, False

        # Only single-objective. (MO ranking is undefined here by design.)
        if hasattr(study, "directions") and len(study.directions) != 1:
            return below, above, False

        # Coin flip
        if float(self._rng.rng.random()) >= self._epsilon2:
            return below, above, False

        k = len(below)
        if k <= 0:
            return below, above, False

        # Eligible: finite scalar value only.
        eligible: list[tuple[float, FrozenTrial]] = []
        for t in above:
            v = getattr(t, "value", None)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if fv != fv or fv in (float("inf"), float("-inf")):
                continue
            eligible.append((fv, t))

        if len(eligible) < k:
            return below, above, False

        # Direction
        direction = getattr(study, "direction", None)
        dname = str(getattr(direction, "name", direction)).lower()
        minimize = "minimize" in dname

        # Rank-bias: higher weight for better trials within above.
        eligible.sort(key=lambda x: x[0], reverse=not minimize)
        ranked = [t for _, t in eligible]
        m = len(ranked)
        weights = [float(m - i) for i in range(m)]

        below2 = self._weighted_sample_without_replacement(ranked, weights, k)
        below2_nums = {t.number for t in below2}

        above2: list[FrozenTrial] = list(below)
        above2.extend([t for t in above if t.number not in below2_nums])

        return below2, above2, True

    # ---------------- overrides for gating + draw timing ----------------
    def _sample_relative(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        if search_space == {}:
            return {}

        self._on_new_trial_context(study, trial)
        if self._tls_state().random_this_trial:
            return self._sample_random_relative(study, trial, search_space)

        finished = self._get_finished_count_all(study, trial)
        if finished < self._n_startup_trials:
            return {}
        return self._sample(study, trial, search_space, use_trial_cache=True)

    def sample_independent(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._on_new_trial_context(study, trial)
        if self._tls_state().random_this_trial:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        finished = self._get_finished_count_all(study, trial)
        if finished < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        # Keep upstream warning behavior approximately (without additional fetch)
        if self._warn_independent_sampling and self._multivariate:
            snap = self._ensure_snapshot(study, trial)
            if any(param_name in t.params for t in snap.trials_reduced):
                tpl = None
                try:
                    from optuna.samplers._base import (
                        _INDEPENDENT_SAMPLING_WARNING_TEMPLATE as tpl,
                    )
                except Exception:
                    tpl = None

                if tpl is None:
                    msg = (
                        f"The parameter `{param_name}` in Trial#{trial.number} is sampled independently "
                        f"using `{self._random_sampler.__class__.__name__}` instead of `{self.__class__.__name__}`. "
                        "This can degrade optimization performance. "
                        "You can suppress this warning by setting `warn_independent_sampling=False`."
                    )
                else:
                    msg = tpl.format(
                        param_name=param_name,
                        trial_number=trial.number,
                        independent_sampler_name=self._random_sampler.__class__.__name__,
                        sampler_name=self.__class__.__name__,
                        fallback_reason="the parameter is not in the relative search space",
                    )
                from optuna.logging import get_logger

                get_logger(__name__).warning(msg, stacklevel=2)

        params = self._sample(
            study, trial, {param_name: param_distribution}, use_trial_cache=True
        )
        return params[param_name]

    def _sample(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
        use_trial_cache: bool,  # ignored by design (snapshot governs)
    ) -> Dict[str, Any]:
        _ = self._ensure_snapshot(study, trial)
        mpe_below, mpe_above = self._get_or_build_mpe_pair(study, trial, search_space)

        with self._timed("draw_point"):
            samples_below = mpe_below.sample(self._rng.rng, self._n_ei_candidates)
            acq = self._compute_acquisition_func(samples_below, mpe_below, mpe_above)
            ret = _OptunaTPESampler._compare(samples_below, acq)

            for name, dist in search_space.items():
                ret[name] = dist.to_external_repr(ret[name])

        return ret

    # ---------------- debug helpers ----------------
    def _debug_set_marker(self, value: str) -> None:
        self._tls_state().debug_marker = value

    def _debug_get_marker(self) -> Optional[str]:
        return self._tls_state().debug_marker
