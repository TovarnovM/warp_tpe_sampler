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
from warp_tpe_sampler import BudgetPolicyConfig, WarpTpeConfig, WarpTpeSampler

budget = BudgetPolicyConfig(
    warmup_steps=5,
    safety=0.9,
    alpha=0.20,
    ema_halflife_s=16.0,
    max_bank_s=30.0,
    t_min_s=0.01,
    t_max_s=30.0,
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
    alpha=0.25,     # overrides budget_policy.alpha
    epsilon2=0.02,  # “below2” diversification

    # Optional trial annotations
    trial_attrs="basic",
)

sampler = WarpTpeSampler(cfg)


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

* `epsilon: float` (policy-level exploration; overrides `budget_policy.epsilon`)
* `alpha: float | None` (optional; overrides `budget_policy.alpha`)
* `epsilon2: float` (passed down to `CachedTPESampler` as “below2” probability)

### Budget policy integration

* `budget_policy_enabled: bool`
* `budget_policy: BudgetPolicyConfig | None`

### Optional trial annotations

* `trial_attrs: "none" | "basic" | "full"`

  * `basic`: store action label + small stats payload
  * `full`: store extended internal stats (can be large)

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

`WarpTpeSampler` can embed a lightweight *time-budget controller* that decides, per trial, whether to:

* `RUN` — rebuild (or reuse) the cached TPE snapshot normally and sample via TPE.
* `FREEZE` — reuse the previous cached snapshot for this trial (avoids expensive refresh).
* `REDUCE` — refresh the cached snapshot, but keep only `N` finished trials.
* `RANDOM` — force random sampling for the whole trial.

The policy is intentionally minimal. It does **not** attempt to model Optuna's internal costs in detail; instead it uses a *time bank* with two decision thresholds.

## Time bank model

You can feed the policy with the measured black-box runtime via `WarpTpeSampler.set_last_blackbox_time_s(t)`.

Let:

* $t_{bb}$ be the measured black-box runtime of the last completed trial (seconds).
* $t_{bb,eff} = \max(t_{bb}, t_{min})$ be the effective runtime floor.
* $\alpha \in (0,1)$ be the feedback parameter and $\beta = \frac{\alpha}{1-\alpha}$.
* $b$ be the internal bank (seconds), clipped to $[0, \text{max\_bank\_s}]$.

The policy maintains an EMA $\hat{t}_{bb}$ of $t_{bb,eff}$ using a wall-clock half-life:

$$
\text{decay} = 0.5^{\Delta t / \text{ema\_halflife\_s}}, \qquad
\hat{t}_{bb} \leftarrow \text{decay}\,\hat{t}_{bb} + (1-\text{decay})\,t_{bb,eff}
$$

For the *next* decision, the policy uses a conservative benchmark prediction:

$$
t_{bench} = \text{safety} \cdot \hat{t}_{bb}
$$

The *available budget* is:

$$
\text{available} = \max(0, b + \beta\, t_{bench})
$$

After an action is chosen, the policy records a predicted time for that action $t_{pred}$. It then updates the bank:

$$
\text{overhead} = \max(0, t_{bench} - t_{pred})
$$

$$
b \leftarrow \operatorname{clip}_{[0,\text{max\_bank\_s}]}\bigl(b + \beta\, t_{bb,eff} - \text{overhead}\bigr)
$$

Interpretation:

* $\beta\,t_{bb,eff}$ is an income term proportional to how long the objective took.
* $\text{overhead}$ charges the policy for choosing actions predicted to be faster than the benchmark. This makes the bank represent slack *relative to the benchmark*, rather than raw wall time.

## Decision logic

Let `available` be computed as above, and let $t_{min}$ / $t_{max}$ be thresholds in seconds.

1. Warmup
   * if `total_steps < warmup_steps` → `RUN`.

2. Threshold-based action
   * if `available >= t_max` → `RUN`.
   * else if `available >= t_min` → `FREEZE`.
   * else → `REDUCE` if `n_keep > n_min`, otherwise `RANDOM`.

   For `REDUCE`, the policy outputs:

   $$
   n_{used} = \operatorname{clip}_{[n_{min}, n_{max}]}(n_{keep})
   $$

3. Exploration (epsilon-greedy)
   * with probability `epsilon` → override the chosen action with `RANDOM`.

## Policy configuration reference (`BudgetPolicyConfig`)

* `warmup_steps: int` — number of initial decisions forced to `RUN`.
* `safety: float` — conservative multiplier for the benchmark prediction in $(0,1]$.
* `alpha: float` — feedback parameter in $(0,1)$ controlling $\beta = \frac{\alpha}{1-\alpha}$.
* `ema_halflife_s: float` — half-life (seconds) for the EMA of black-box runtime.
* `max_bank_s: float` — clip limit for the time bank (seconds).
* `t_min_s: float` — threshold (seconds) below which the policy must `REDUCE` or `RANDOM`.
* `t_max_s: float` — threshold (seconds) above which the policy can safely `RUN`.
* `n_min: int`, `n_max: int` — bounds for the `reduce_n` produced by `REDUCE`.
* `epsilon: float` — epsilon-greedy override probability.
* `seed: int | None` — RNG seed for epsilon-greedy decisions.

### Where to set `alpha` and `epsilon`

If you embed the policy via `WarpTpeSampler` / `WarpTpeConfig`, you can override two parameters at the top level:

* `WarpTpeConfig.epsilon` (always overrides `budget_policy.epsilon`)
* `WarpTpeConfig.alpha` (if set, overrides `budget_policy.alpha`)
* `WarpTpeSampler(alpha=...)` (highest priority override; beats both config levels)

## Introspection API (WarpTpeSampler)

* `set_last_blackbox_time_s(t: float)` — provide the measured black-box runtime (seconds).
* `get_last_trial_stats() -> dict | None` — returns the last recorded decision/telemetry snapshot.
* `get_action_counts() -> dict[str, int]` — counts of actions/events (freeze/reduce/random/epsilon/epsilon2 etc.).

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
