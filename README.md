# Stock Price Prediction (Graduation Research)

This repository contains **four trinary classification** stock-price prediction experiments:

- Transformer (time-series window)
- LSTM (time-series window)
- Logistic Regression (pooled window features)
- LightGBM (pooled window features)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run an experiment (backward-compatible entrypoints)

```bash
python transformer/trinary_transformer_old.py
python LSTM/trinary_LSTM.py
python Logistic_Regression/trinary_logistic_regression.py
python LightGBM/trinary_LightGBM.py
```

Each script writes figures and a `run_summary.json` under the same paths as before
(e.g. `transformer/run_summary.json`).

### Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Repo layout

- `src/stock_pred/`: canonical library code
  - `common_shared.py`: shared utilities (seeding, plots, metrics, PWFE splitter, run summary)
  - `dataset_pipeline.py`: data fetching, feature engineering, labeling, windowing
  - `models/`: experiment scripts, moved into a package with minimal edits
- Top-level `common_shared.py` and `dataset_pipeline.py` are **compatibility shims** for old imports.
- `transformer/`, `LSTM/`, `Logistic_Regression/`, `LightGBM/` contain thin wrappers that preserve original entrypoints.

## Reproducibility notes

- `set_seeds(42)` is called to match the original scripts.
- Deep-learning runs may still be non-deterministic depending on TensorFlow/CUDA versions and GPU kernels.
  See `tests/test_reproducibility_notes.py` for what is (and is not) guaranteed.

