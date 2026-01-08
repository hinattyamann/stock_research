from __future__ import annotations

import numpy as np
import pandas as pd

from stock_pred.dataset_pipeline import (
    add_log1p_features,
    build_tabular_dataset,
    make_features,
    make_trinary_labels,
    replace_inf_and_dropna,
    to_windows,
)

def _toy_ohlcv(n: int = 120) -> pd.DataFrame:
    # deterministic toy OHLCV (not market-realistic; for regression checks only)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    base = np.linspace(100.0, 120.0, n)
    df = pd.DataFrame(
        {
            "open": base + 0.1,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "volume": np.linspace(1e6, 2e6, n),
        },
        index=idx,
    )
    return df

def test_make_features_has_expected_core_columns():
    df = make_features(_toy_ohlcv())
    for col in ["ret1", "sigma20", "rsi14", "macd", "atr", "bb_width", "vol_ratio"]:
        assert col in df.columns

def test_trinary_labels_value_range_and_alignment():
    df = make_features(_toy_ohlcv())
    y, abs_metric = make_trinary_labels(df, horizon=1, k_tau=0.3)
    # labels can contain NA at the tail; drop for range check
    ys = y.dropna().astype(int)
    assert set(np.unique(ys.values)).issubset({0, 1, 2})
    assert len(abs_metric) == len(df)

def test_windowing_dtype_and_shape():
    X = np.random.RandomState(0).randn(100, 4).astype(np.float32)
    y = np.random.RandomState(1).randint(0, 3, size=100).astype(int)
    Xw, yw = to_windows(X, y, win=10)
    assert Xw.dtype == np.float32
    assert yw.dtype == int
    assert Xw.shape == (90, 10, 4)
    assert yw.shape == (90,)

def test_build_tabular_dataset_smoke():
    df = make_features(_toy_ohlcv(200))
    df = add_log1p_features(df, candidates=["sigma20", "atr"])
    y, abs_metric = make_trinary_labels(df, horizon=1, k_tau=0.3)

    feature_cols = [c for c in df.columns if c not in {"open","high","low","close","volume"}]
    X_all, y_np, abs_all, feat_final, data = build_tabular_dataset(
        df,
        y=y,
        abs_metric=abs_metric,
        feature_cols=feature_cols,
        use_topix_features=False,
        use_log1p=True,
        log1p_candidates=["sigma20", "atr"],
        extra_log1p_cols=(),
        nan_tail_days=10,
    )
    assert X_all.ndim == 2
    assert len(y_np) == X_all.shape[0]
    assert len(abs_all) == X_all.shape[0]
    assert len(feat_final) == X_all.shape[1]
    assert len(data) == X_all.shape[0]
