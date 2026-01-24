# warp-tpe-sampler

WarpTpeSampler for Optuna: CachedTPESampler + embedded budget policy (refresh/freeze/random).

## Install
```bash
pip install warp-tpe-sampler
```

## Quickstart
```python
import optuna
from warp_tpe_sampler import WarpTpeSampler, WarpTpeConfig

sampler = WarpTpeSampler(WarpTpeConfig(seed=0, n_startup_trials=20))
study = optuna.create_study(direction="minimize", sampler=sampler)

def objective(trial):
    x = trial.suggest_float("x", 0.0, 1.0)
    return (x - 0.37) ** 2

study.optimize(objective, n_trials=200, n_jobs=1)
```

## Notes

Intended usage: `n_jobs == 1`.

Depends on Optuna internals; see supported Optuna versions in pyproject.toml.