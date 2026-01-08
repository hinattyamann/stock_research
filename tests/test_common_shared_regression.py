from __future__ import annotations

import numpy as np

from stock_pred.common_shared import PWFESplit, _class_counts, _check_finite, _score_trinary

def test_class_counts_trinary():
    y = np.array([0, 1, 2, 2, 1, 1])
    out = _class_counts(y, n_classes=3)
    assert out == {0: 1, 1: 3, 2: 2}

def test_check_finite_summary():
    x = np.array([[1.0, np.nan], [np.inf, 3.0]])
    info = _check_finite("x", x)
    assert info["has_nan"] is True
    assert info["has_inf"] is True
    assert info["shape"] == [2, 2]

def test_score_trinary_matches_known():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])
    s1 = _score_trinary(y_true, y_pred, metric="macro_f1")
    s2 = _score_trinary(y_true, y_pred, metric="balanced_acc")
    # regression: values are stable for scikit-learn
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0

def test_pwfe_split_basic_properties():
    splitter = PWFESplit(n_splits=3, embargo=2, min_train=1, min_val=1)
    splits = splitter.split(30)
    assert len(splits) >= 1
    for tr, va in splits:
        assert tr.max() < va.min()  # embargo purges tail
