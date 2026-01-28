# warp-tpe-sampler

A fast, Optuna-compatible implementation of **cached TPE** with optional **budget-aware control** (“Warp TPE”) to reduce per-trial sampler overhead (storage fetch, TPE refresh) under wall-clock constraints.

This repository provides two layers:

1. **`CachedTPESampler`** — a drop-in replacement for Optuna’s `TPESampler` that **caches** expensive intermediate state and avoids repeated work across parameter suggestions within the same trial.
2. **`WarpTpeSampler`** — `CachedTPESampler` + an embedded **budget policy** that decides *per trial* whether to:

   * **REFRESH** TPE state using all (or reduced) history,
   * **FREEZE** and reuse a cached snapshot to avoid refresh cost,
   * **RANDOM** sample when the time budget is too tight (or for exploration).

This README avoids LaTeX so all “formulas” render correctly in GitHub preview.

---

## Installation

```bash
pip install warp-tpe-sampler
```

Runtime dependency: **Optuna 4.x**.

---

## Quick start

### Cached TPE (drop-in)

```python
import optuna
from warp_tpe_sampler import CachedTPESampler


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_int("y", 0, 10)
    return (x - 1.23) ** 2 + (y - 7) ** 2


sampler = CachedTPESampler(
    n_startup_trials=20,
    seed=0,
    multivariate=True,
    group=True,
    constant_liar=False,
)

study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=200)
```

### Warp TPE (budget-aware)

```python
import optuna
from warp_tpe_sampler import WarpTpeSampler, WarpTpeConfig
from warp_tpe_sampler.pbt_funcs.budget_policy import BudgetPolicyConfig

budget = BudgetPolicyConfig(
    warmup_trials=5,
    warmup_steps=5,
    safety=0.9,
    beta=0.25,
    ema_halflife=16,
    max_bank_s=30.0,
    n_min=16,
    n_max=512,
    epsilon=0.05,
    seed=0,
)

cfg = WarpTpeConfig(
    n_startup_trials=20,
    seed=0,
    multivariate=True,
    group=True,
    constant_liar=False,

    # Budget-policy integration
    budget_policy_enabled=True,
    budget_policy=budget,

    # Reduction strategy used when the policy asks for reduced refresh
    reduce_kind="tail_plus_random",
    reduce_tail_frac=0.7,

    # Exploration & diversification
    epsilon=0.05,   # overrides budget_policy.epsilon
    epsilon2=0.02,  # “below2” diversification

    # Optional trial annotations
    trial_attrs="basic",
)

sampler = WarpTpeSampler(cfg)


### Custom trial.user_attrs hook

If you want to write your own `trial.user_attrs` (e.g. to log domain-specific metrics), pass a callable via `WarpTpeSampler(..., trial_user_attrs_fn=...)` or set it on `WarpTpeConfig.trial_user_attrs_fn`.

The callable is invoked from `after_trial` and receives a `set_user_attr(key, value)` helper so it can write into Optuna storage even though it is called with a `FrozenTrial`.

```python
def my_attrs_writer(*, trial, study, sampler, set_user_attr, **kwargs):
    set_user_attr("custom.trial", trial.number)
    set_user_attr("custom.action", str(kwargs.get("action")))

sampler = WarpTpeSampler(cfg, trial_user_attrs_fn=my_attrs_writer)
```


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)

    # Optional: if you measure black-box runtime yourself, feed it here.
    # sampler.set_last_blackbox_time_s(measured_seconds)

    return (x - 1.23) ** 2


study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=200)

print("Action counts:", sampler.get_action_counts())
print("Last trial stats:", sampler.get_last_trial_stats())
```

---

## Why caching matters

A TPE sampler commonly does (often per parameter suggestion):

1. Fetch trials from storage
2. Split trials into “good” vs “bad” sets (`below` / `above`)
3. Build Parzen estimators for each parameter distribution
4. Sample candidates, score acquisition, pick best

In hierarchical and/or multivariate settings, steps (2–3) can repeat **many times per trial** even though the set of completed trials hasn’t changed during the trial.

`CachedTPESampler` introduces a **snapshot** that is computed once (or reused) and shared across all `suggest_*` calls within a trial.

---

# `CachedTPESampler`

## What it changes vs Optuna `TPESampler`

### 1) Snapshot caching per trial

For each new trial, the sampler can build and cache:

* `trials_all`: all trials fetched from study storage
* `trials_reduced`: a subset after applying the configured reduction function
* split cache keyed by *search-space key* (important for hierarchical spaces):

  * `below_trials`, `above_trials`
* estimator cache keyed by subspace

This prevents repeated:

* storage fetch
* splitting
* estimator rebuild

across repeated parameter suggestions inside the same trial.

### 2) Reduction hook (`reduce_trials`) + dynamic `reduce_n`

The sampler accepts a user-supplied function:

```python
reduce_trials(
    trials: list[FrozenTrial],
    n_keep: int | None,
    trial_number: int,
    rng: np.random.RandomState,
) -> list[FrozenTrial]
```

* If `n_keep is None`, reducers should typically behave as “no reduction”.
* A higher-level controller (e.g., `WarpTpeSampler`) can set a thread-local `reduce_n` that is forwarded as `n_keep`.

### 3) Two exploration controls: `epsilon` and `epsilon2`

* `epsilon`: with probability `epsilon` random-sample *this trial* (post-startup)
* `epsilon2`: “below2” diversification that changes how `below` is formed (post-startup)

### 4) One-shot controls

* `use_cached_snapshot_once()` — force using the cached snapshot for the next trial (if any)
* `use_random_once()` — force the next trial to be random (fast-path exploration)

---

## Parameter reference (CachedTPESampler)

`CachedTPESampler` mostly mirrors Optuna’s `TPESampler`, plus additional knobs.

### Sampling / TPE behavior

* `n_startup_trials: int` — number of completed trials required before using TPE; before that, random sampling.
* `n_ei_candidates: int` — number of candidates used to approximate acquisition.
* `gamma: Callable[[int], int]` — maps number of finished trials → number of “below” trials.
* `weights: Callable[[int], Sequence[float]]` — trial weights for Parzen estimation.
* `multivariate: bool`, `group: bool` — Optuna experimental multivariate/grouped sampling flags.
* `constant_liar: bool` — Optuna experimental constant liar.
* `warn_independent_sampling: bool` — warn when independent sampling happens while `multivariate=True`.

### Caching / reduction / exploration

* `reduce_trials: Callable[...]` — reduction hook.
* `epsilon: float` — probability to random-sample a trial (post-startup).
* `epsilon2: float` — probability to apply “below2” diversification (post-startup).

---

## TPE objective and key formulas (GitHub-friendly)

Optuna’s TPE builds two density models for a parameter vector `x`:

* `l(x)`: density estimated from “good” trials (`below`)
* `g(x)`: density estimated from “bad” trials (`above`)

A common acquisition is to sample candidates from `l(x)` and pick one maximizing the density ratio:

* `x* = argmax_x ( l(x) / g(x) )`
* equivalently maximize: `log l(x) - log g(x)`

### Below/above split

Let `n` be the number of completed trials (after reduction). Define:

* `n_below = gamma(n)`
* `below = best n_below trials` (direction-aware)
* `above = remaining trials`

In multi-objective mode, Optuna’s internal MO logic is used, but caching applies the same way.

---

## “below2” (`epsilon2`) diversification

Hierarchical spaces and aggressive reduction can make `below` too narrow. With probability `epsilon2`, the sampler transforms the split:

* Keep `above` unchanged.
* Replace `below` with a weighted sample from `above` of size `k = len(below)`.

### Weighted sampling scheme

Let `m = len(above)` and index `above` as `above[0], ..., above[m-1]`. Define descending integer weights:

* `w_i = m - i` for `i = 0..m-1`

Then sample `k` elements **without replacement** from `above` using weights `w`.

Intuition: earlier elements in `above` (closer to the split boundary) are more likely to be chosen, widening what counts as “good” while staying near the decision surface.

---

## Trial reduction strategies (concept)

Reduction is externalized via `reduce_trials`. Common patterns:

### “Keep last N”

* `reduced = trials[-N:]`

### “Tail + random”

Keep a recent “tail” and fill the rest with random picks from older trials.

Let:

* `N` = total to keep
* `f` = tail fraction in `(0, 1)`
* `n_tail = floor(f * N)`
* `n_rand = N - n_tail`

Then:

* `reduced = tail(n_tail) + random_sample(older, n_rand)`

This is the default idea used by `WarpTpeSampler` when `reduce_kind="tail_plus_random"`.

---

## Timing and internal stats

`CachedTPESampler` tracks wall-clock timing for major components (names may vary by version):

* trial fetch
* reduction
* split
* estimator build
* candidate draw / acquisition scoring

These stats are consumed by `WarpTpeSampler`’s budget policy.

---

# `WarpTpeSampler` (Cached TPE + Budget Policy)

## Motivation

Sampler overhead can be significant when:

* storage is remote (RDB / gRPC proxy)
* the study has many trials (fetch/refresh becomes expensive)
* you have strict wall-clock limits

`WarpTpeSampler` embeds a **Budgeted Reduction Policy** that tries to keep sampler overhead bounded relative to the black-box evaluation time.

---

## `WarpTpeConfig` reference

### TPE settings

* `n_startup_trials: int`
* `n_ei_candidates: int`
* `seed: int | None`
* `multivariate: bool`
* `group: bool`
* `constant_liar: bool`
* `consider_endpoints: bool`
* `consider_magic_clip: bool`
* `prior_weight: float`
* `warn_independent_sampling: bool`

### Reduction settings

* `reduce_kind: "last_n" | "tail_plus_random"`
* `reduce_tail_frac: float` (only for `tail_plus_random`)

### Exploration and diversification

* `epsilon: float` (policy-level exploration; overrides nested policy epsilon)
* `epsilon2: float` (passed down to `CachedTPESampler` as “below2” probability)

### Budget policy integration

* `budget_policy_enabled: bool`
* `budget_policy: BudgetPolicyConfig | None`
* `alpha: float | None` (optional override for `BudgetPolicyConfig.alpha`; you can also pass it to `WarpTpeSampler(alpha=...)`)

### Optional trial annotations

* `trial_attrs: "none" | "basic" | "full"`
* `trial_user_attrs_fn: callable | None` (custom hook called from `after_trial` to set trial user attrs; executed regardless of `trial_attrs`)

  * `basic`: store action label + small stats payload
  * `full`: store extended internal stats (can be large)

---

## Custom trial user attrs hook

You can pass a custom function that writes into `trial.user_attrs` from the sampler (executed in `after_trial`).

This is useful when you want to annotate trials with custom metrics, debugging info, or policy internals.

The hook is called like:

```python
def writer(*, trial, study, sampler, cfg, decision, action, reason, last_trial_stats, ctx, state, set_user_attr, **kwargs):
    # Use `set_user_attr(key, value)` to write to the current trial.
    set_user_attr("my.key", "value")
```

Usage:

```python
cfg = WarpTpeConfig(trial_attrs="none", trial_user_attrs_fn=writer)
sampler = WarpTpeSampler(cfg)
# OR: sampler = WarpTpeSampler(cfg, trial_user_attrs_fn=writer)  # overrides cfg hook
```

---

## How WarpTpeSampler maps policy decisions into sampler behavior

Per trial, the policy returns a decision:

* `Action.REFRESH` (optionally with `reduce_n`)
* `Action.FREEZE`
* `Action.RANDOM`

WarpTpeSampler applies it as follows:

* **REFRESH**:

  * clears snapshot (forces refresh)
  * sets `reduce_n` so `reduce_trials(..., n_keep=reduce_n, ...)` is applied
* **FREEZE**:

  * reuses the cached snapshot (no refresh)
* **RANDOM**:

  * triggers one-shot random mode (`use_random_once()`)

WarpTpeSampler typically disables internal `CachedTPESampler(epsilon=...)` to keep exploration logic centralized in the policy.

---

# Budget Policy (`BudgetPolicyConfig` + `BudgetedReductionPolicy`)

## Overview

The policy maintains a **time bank** (seconds), updated once per completed trial.

Definitions:

* `T_bb` = black-box evaluation time (seconds)
* `T_ov` = sampler overhead time (seconds)

  * typically `T_fetch + T_sampler`

### Effective black-box time floor

To avoid instability for very fast objectives:

* `T_bb_eff = max(T_bb, t_min_sec)`

---

## Bank update equation

Income is proportional to black-box time:

* `income = beta * T_bb_eff`

Spending is measured overhead:

* `spend = T_fetch + T_sampler`

Bank update:

* `bank_next = clip(bank + income - spend, -max_bank_s, +max_bank_s)`

Interpretation:

* `beta` controls the target overhead fraction (e.g., `beta=0.25` targets about 25% overhead vs black-box time).
* `max_bank_s` bounds accumulated credit/debt.

---

## Available budget for the next decision

The policy predicts next-step income using an EMA of `T_bb_eff`:

* `T_hat = EMA(T_bb_eff)`

Then:

* `available = max(0, bank + beta * T_hat)`
* `available_safe = safety * available`

`safety` is a margin factor in `(0, 1)`.

---

## Prediction model for overhead

The policy maintains EMA estimates for per-unit costs:

* `fetch_per_trial ≈ EMA(T_fetch / n_total)`
* `refresh_per_trial ≈ EMA(T_refresh / n_used)`
* `freeze_cost ≈ EMA(T_freeze)`

Predicted costs:

* `T_fetch_hat(n_total) = fetch_per_trial * n_total`
* `T_refresh_hat(n_used) = refresh_per_trial * n_used`
* `T_freeze_hat = freeze_cost`

`ema_halflife` controls smoothing.

---

## Decision logic (high-level)

Given:

* `n_total = number of finished trials`
* `has_snapshot = whether a cached snapshot exists`

The policy selects an action using this high-level structure:

1. Exploration (epsilon-greedy)

   * with probability `epsilon` → `RANDOM`

2. Startup / warmup

   * if `n_total < warmup_trials` → `RANDOM`
   * else if `warmup_steps > 0` and still warming up → `REFRESH(full)`

3. Prefer full refresh if it fits

   * if `T_fetch_hat(n_total) + T_refresh_hat(n_total) <= available_safe` → `REFRESH(full)`

4. Otherwise, try reduced refresh

   * compute maximum feasible `n_maxfit`:

     * `n_maxfit = floor((available_safe - T_fetch_hat(n_total)) / refresh_per_trial)`
   * clamp:

     * `n_used = clamp(n_maxfit, n_min, n_max)`
   * if feasible → `REFRESH(reduce_n=n_used)`

5. Otherwise, freeze if affordable

   * if `has_snapshot` and `freeze_streak < max_freeze_streak` and `T_freeze_hat <= available_safe` → `FREEZE`

6. Fallback

   * `RANDOM`

---

## Policy configuration reference (`BudgetPolicyConfig`)

* `warmup_trials: int` — minimum finished trials before the policy may use non-random decisions.
* `warmup_steps: int` — number of forced refresh steps after warmup.
* `randomize_every: int` — force `RANDOM` every K steps (optional).
* `enforce_randomize_when_snapshot: bool` — optional forcing behavior when a snapshot exists.
* `safety: float` — safety margin multiplier `< 1`.
* `beta: float` — income fraction: overhead budget per unit black-box time.
* `max_bank_s: float` — bank clamp.
* `ema_halflife: int` — EMA responsiveness.
* `max_freeze_streak: int` — cap on consecutive FREEZE actions.
* `n_min: int`, `n_max: int` — bounds for reduced refresh size (`reduce_n`).
* `epsilon: float` — exploration probability.
* `seed: int | None` — policy RNG seed.
* `t_min_sec: float` — floor for effective black-box time.

---

## Introspection API (WarpTpeSampler)

* `set_last_blackbox_time_s(t: float)` — provide measured black-box runtime (seconds) for policy updates.
* `get_last_trial_stats() -> dict | None` — returns last recorded action/stats snapshot (depends on `trial_attrs`).
* `get_action_counts() -> dict[str, int]` — counts of actions/events (refresh/freeze/random/epsilon/epsilon2 etc.).

---

# Compatibility and version pinning

This project is Optuna-compatible at the public API level, but relies on some internal Optuna components for TPE behavior.

Recommended:

* Pin Optuna to a validated minor series in your downstream projects (e.g., `optuna==4.1.*`).
* Keep CI running against that pinned series and bump intentionally.

---

# Development

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Lint

```bash
ruff check .
```

---

# License

MIT (see `LICENSE`).
