# common_shared.py
from __future__ import annotations

import os
import json
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

@dataclass(frozen=True)
class LogConfig:
    """
    4モデルで共通のログ/保存設定。
    既存コードの LOGCFG.save_json / LOGCFG.json_path などをそのまま使えるように
    属性名は統一する。
    """
    save_json: bool = True
    json_path: str = ""
    print_fold_line: bool = True
    print_confusion: bool = False
    # DL（Transformer/LSTM）だけで使う想定。未使用でも害はないので共通側に置く。
    quiet_fit: bool = True


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _class_counts(y: Any, n_classes: int = 3) -> Dict[int, int]:
    """Count labels in y (trinary by default)."""
    out = {int(k): 0 for k in range(n_classes)}
    y = np.asarray(y).reshape(-1)
    for k, c in zip(*np.unique(y, return_counts=True)):
        out[int(k)] = int(c)
    return out


def _check_finite(name: str, arr: Any) -> Dict[str, Any]:
    """Return a small finite-check summary without dumping full arrays."""
    arr = np.asarray(arr)
    return {
        "name": name,
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def set_seeds(seed: int = 42) -> None:
    """Best-effort seed fixing across common libs."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    # TensorFlow is used by Transformer/LSTM but not by LGBM/LogReg
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except Exception:
        pass

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _score_trinary(y_true, y_pred, metric: str = "macro_f1") -> float:
    """三値分類用の単純スコア関数（macro F1 or balanced accuracy）。"""
    return f1_score(y_true, y_pred, average="macro") if metric == "macro_f1" else balanced_accuracy_score(y_true, y_pred)


def compute_mda_importance(
    model: Any,
    Xva_win: np.ndarray,
    yva_win: np.ndarray,
    metric_name: str,
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 3,
    random_state: int = 42,
) -> "pd.DataFrame":
    """
    時系列窓のPermutation Importance (MDA)。
    - baseline: 検証窓での推論スコア（macro_f1/balanced acc）
    - 特徴jを時系列を保ったままpermuteし、スコア低下量を重要度とする
    """
    import pandas as pd

    rng = np.random.RandomState(random_state)
    proba = model.predict(Xva_win, verbose=0)
    yhat = np.argmax(proba, axis=1)
    base = _score_trinary(yva_win, yhat, metric=metric_name)

    n_feat = Xva_win.shape[-1]
    names = feature_names if (feature_names and len(feature_names) == n_feat) else [f"f{j}" for j in range(n_feat)]
    drops, stds = [], []

    for j in range(n_feat):
        scores = []
        for _ in range(n_repeats):
            Xp = Xva_win.copy()
            for t in range(Xp.shape[1]):
                idx = rng.permutation(Xp.shape[0])
                Xp[:, t, j] = Xp[idx, t, j]

            yp = np.argmax(model.predict(Xp, verbose=0), axis=1)
            sc = _score_trinary(yva_win, yp, metric=metric_name)
            scores.append(sc)

        scores = np.asarray(scores, float)
        drops.append(base - scores.mean())
        stds.append(scores.std())

    imp = pd.DataFrame({"feature": names, "importance": drops, "std": stds}).sort_values("importance", ascending=False)
    return imp.reset_index(drop=True)


def compute_mda_importance_tabular(
    model: Any,
    X_va: np.ndarray,
    y_va: np.ndarray,
    metric_name: str = "macro_f1",
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 3,
    random_state: int = 42,
) -> "pd.DataFrame":
    """タブular用Permutation Importance (LightGBM向けの元実装と同一ロジック)。"""
    import pandas as pd

    rng = np.random.RandomState(random_state)
    proba = model.predict_proba(X_va)
    yhat = np.argmax(proba, axis=1)
    base = f1_score(y_va, yhat, average="macro") if metric_name == "macro_f1" else balanced_accuracy_score(y_va, yhat)

    n_feat = X_va.shape[-1]
    names = feature_names if (feature_names and len(feature_names) == n_feat) else [f"f{j}" for j in range(n_feat)]
    drops, stds = [], []
    for j in range(n_feat):
        scores = []
        for _ in range(n_repeats):
            Xp = X_va.copy()
            idx = rng.permutation(Xp.shape[0])
            Xp[:, j] = Xp[idx, j]
            yp = np.argmax(model.predict_proba(Xp), axis=1)
            sc = f1_score(y_va, yp, average="macro") if metric_name == "macro_f1" else balanced_accuracy_score(y_va, yp)
            scores.append(sc)
        scores = np.asarray(scores, float)
        drops.append(base - scores.mean())
        stds.append(scores.std())

    imp = pd.DataFrame({"feature": names, "importance": drops, "std": stds}).sort_values("importance", ascending=False)
    return imp.reset_index(drop=True)


def plot_importance(df_imp, save_path: str, topk: int = 20, title: str = "Permutation Importance(MDA)"):
    """MDAの上位重要特徴を横棒グラフに保存。元実装の挙動を維持。"""
    import matplotlib.pyplot as plt

    k = min(topk, len(df_imp))
    sub = df_imp.iloc[:k].iloc[::-1]
    plt.figure(figsize=(8, 0.35 * k + 1.5))
    plt.barh(sub["feature"], sub["importance"], xerr=sub.get("std", None))
    plt.xlabel("Score drop vs baseline")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def finalize_run_summary(
    run_summary: Dict[str, Any],
    *,
    pooled_main: Dict[str, Any],
    pooled_subset: Dict[str, Any],
    logcfg: Any,
) -> Dict[str, Any]:
    """
    各モデル末尾に散らばっている
    - finished_at の付与
    - pooled(main/subset) の付与
    - JSON保存（LOGCFG.save_json / LOGCFG.json_path）
    を共通化する。
    既存の挙動を変えないため、logcfgはduck-typingで受ける。
    """
    run_summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_summary["pooled"] = {"main": pooled_main, "subset": pooled_subset}

    save_json = bool(getattr(logcfg, "save_json", False))
    json_path = str(getattr(logcfg, "json_path", "") or "")
    if save_json and json_path:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)
        print(f"[LOG] saved run summary to {json_path}")
    return run_summary

def _plot_history(hist, figdir: str, fold: int, task: str, extra_title: str = "") -> None:
    """Kerasの学習履歴（loss/val_loss, 可能ならval_accuracy）をPNGで保存。"""
    if hist is None:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    h = getattr(hist, "history", None)
    if not isinstance(h, dict):
        return

    _ensure_dir(figdir)
    plt.figure(figsize=(8, 4))
    ax1 = plt.gca()
    ax1.plot(h.get("loss", []), label="train_loss")
    if "val_loss" in h:
        ax1.plot(h["val_loss"], label="val_loss")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")

    key = "val_accuracy" if "val_accuracy" in h else None
    if key:
        ax2 = ax1.twinx()
        ax2.plot(h[key], label=key, alpha=0.7)
        ax2.set_ylabel(key)
        ax2.legend(loc="lower right")

    ax1.grid(True, alpha=0.25)
    plt.title(f"[{task}] Fold{fold} Training History {extra_title}")
    fn = f"{figdir}/hist_fold{fold}.png"
    plt.tight_layout()
    plt.savefig(fn, dpi=160)
    plt.close()


def _plot_lgb_history(eval_result: dict, figdir: str, fold: int, task: str, extra_title: str = "") -> None:
    """LightGBM eval historyをPNGで保存（validation_0/1の一般形に対応）。"""
    if not eval_result:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    keys = list(eval_result.keys())
    if not keys:
        return

    metric_name = None
    series = {}
    for ds in keys:
        metrics = eval_result.get(ds, {})
        if not metrics:
            continue
        if metric_name is None:
            metric_name = next(iter(metrics.keys()))
        if metric_name in metrics:
            series[ds] = metrics[metric_name]
    if not series or metric_name is None:
        return

    _ensure_dir(figdir)
    plt.figure(figsize=(8, 4))
    for ds, vals in series.items():
        label = ds
        if ds.lower().startswith("validation_0"):
            label = f"val_{metric_name}"
        elif ds.lower().startswith("validation_1") or ds.lower().startswith("training"):
            label = f"train_{metric_name}"
        plt.plot(vals, label=label)
    plt.ylabel(metric_name)
    plt.xlabel("iteration")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.title(f"[{task}] Fold{fold} LightGBM History {extra_title}")
    fn = f"{figdir}/hist_fold{fold}.png"
    plt.tight_layout()
    plt.savefig(fn, dpi=160)
    plt.close()

def _plot_cm(cm, labels, save_path: str, normalize: bool = False, title_prefix: str = "") -> None:
    """混同行列をPNG保存（行正規化オプション、視認性向上の縁取りつき）。"""
    if cm is None:
        return
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import patheffects
    except Exception:
        return

    cm = np.asarray(cm)
    cm_to_plot = (cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)) if normalize else cm

    plt.figure(figsize=(6.0, 4.8))
    vmax = 1.0 if normalize else (float(cm_to_plot.max()) if cm_to_plot.size else 1.0)
    im = plt.imshow(cm_to_plot, cmap="viridis", vmin=0.0, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    fmt = ".2f" if normalize else ".0f"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_to_plot[i, j]
            disp_val = val if normalize else np.rint(val)
            txt = f"{disp_val:{fmt}}"
            normed = im.norm(val)
            fg = "white" if normed > 0.55 else "black"
            txt_obj = plt.text(j, i, txt, ha="center", va="center", color=fg, fontsize=10)
            edge = "black" if fg == "white" else "white"
            txt_obj.set_path_effects([
                patheffects.Stroke(linewidth=1.6, foreground=edge),
                patheffects.Normal()
            ])

    plt.title(f"{title_prefix} Confusion Matrix" + (" (norm)" if normalize else ""))
    plt.tight_layout()
    _ensure_dir(str(__import__("pathlib").Path(save_path).parent))
    plt.savefig(save_path, dpi=180)
    plt.close()

@dataclass
class PWFESplit:
    """Purged Walk-Forward with Embargo."""
    n_splits: int = 5
    embargo: int = 5
    min_train: int = 252
    min_val: int = 63

    def split(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        cuts = np.cumsum(fold_sizes)

        splits = []
        start = 0
        for k in range(self.n_splits):
            va_start = start
            va_end = cuts[k]

            tr_end = max(0, va_start - self.embargo)
            tr_idx = idx[:tr_end]
            va_idx = idx[va_start:va_end]

            emb_len = max(0, va_start - tr_end)
            tr_len, va_len = len(tr_idx), len(va_idx)
            print(f"[SPLIT] Fold {k+1}: "
                  f"TRAIN[0..{tr_end-1}] len={tr_len} | "
                  f"Embargo={emb_len} | "
                  f"VAL[{va_start}..{va_end-1}] len={va_len}")

            if tr_len < self.min_train or va_len < self.min_val:
                reason = []
                if tr_len < self.min_train:
                    reason.append(f"train<{self.min_train}")
                if va_len < self.min_val:
                    reason.append(f"val<{self.min_val}")
                print(f"[SPLIT]  -> skip Fold {k+1} ({', '.join(reason)})")
                start = va_end
                continue

            splits.append((tr_idx, va_idx))
            start = va_end

        return splits
