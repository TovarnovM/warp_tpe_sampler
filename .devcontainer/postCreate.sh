#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install -e ".[dev]"

# sanity checks
python -c "import optuna; print('optuna', optuna.__version__)"
pytest -q
