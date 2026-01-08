# \dataset_pipeline.py
from __future__ import annotations

import math
from typing import Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ta indicators（あなたのmake_featuresが使っているもの）
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volume import OnBalanceVolumeIndicator


# 既存コードでEPSを使っていたので、ここに集約（値は既存に合わせて調整可）
EPS = 1e-12


def _log_span(df, tag: str):
    """行数と日付レンジを簡易表示（インデックスがDatetimeIndex想定）。"""
    try:
        n = len(df)
        i0 = df.index[0] if n else None
        i1 = df.index[-1] if n else None
        if hasattr(i0, "date"):  # DatetimeIndex 前提
            s = f"{i0.date()}" if i0 is not None else "NA"
            e = f"{i1.date()}" if i1 is not None else "NA"
        else:
            s = str(i0); e = str(i1)
        print(f"[DATA] {tag}: n={n}, range={s} .. {e}")
    except Exception as ex:
        print(f"[DATA] {tag}: n={len(df)} (range print failed: {ex})")


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """対象銘柄のOHLCVをyfinanceで取得し、最低限のクレンジングをして返す。"""
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                threads=False,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(
                columns={"Close": "close", "Volume": "volume", "Open": "open", "High": "high", "Low": "low"}
            )
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df = df.dropna()
            if len(df) < 50:
                raise ValueError(f"too few rows: {len(df)}")
            return df
        except Exception as ex:
            last_exc = ex
    raise RuntimeError(f"fetch_ohlcv failed: {last_exc}")


def _rolling_slope(series: pd.Series, win: int) -> pd.Series:
    """直近win点の単回帰（時刻→値）の傾きをrollingで求める。
    - トレンド方向の強さ（勾配）を時点ごとに数値化
    - 各windowで線形回帰の傾き（OLSのβ1）を簡便計算
    """
    t = np.arange(win).astype(float)
    t_mean = t.mean()
    denom = ((t - t_mean) ** 2).sum() + EPS

    def _slope(arr):
        x = np.asarray(arr, dtype=float)
        return ((t - t_mean) * (x - x.mean())).sum() / denom
    return series.rolling(win, min_periods=win).apply(_slope, raw=True)


def _rolling_resid_last(series: pd.Series, win: int) -> pd.Series:
    """直近win点の線形回帰に対し「最後の点の残差」をrollingで算出。
    - トレンドに対する現在値の乖離（上振れ/下振れ）を捉える
    - これをZ化（残差/残差のstd）してスケールを揃えて使用
    """
    t = np.arange(win).astype(float)
    t_mean = t.mean()
    denom = ((t - t_mean) ** 2).sum() + EPS

    def _resid(arr):
        x = np.asarray(arr, dtype=float)
        slope = ((t - t_mean) * (x - x.mean())).sum() / denom
        intercept = x.mean() - slope * t_mean
        fitted_last = slope * (win - 1) + intercept # 回帰直線上の最終点
        return x[-1] - fitted_last                  # 実測 - 直線予測
    
    return series.rolling(win, min_periods=win).apply(_resid, raw=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCVから学習に使う派生特徴をまとめて作成。
    - 「未来を見ない」形で作る（rollingは現在まで）
    - 価格/リターン、ボラ、テクニカル、出来高、ギャップ、ローソク比率、トレンド構造、異常フラグなど
    """
    if df is None or len(df) < 20:  # ATR/BBが最低で使う幅
        raise ValueError(f"make_features: insufficient rows ({0 if df is None else len(df)})")
    df = df.copy()

    # 1) リターン系
    df["ret1"]  = df["close"].pct_change()
    df["ret5"]  = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    df["ret20"] = df["close"].pct_change(20)

    # 移動平均と乖離（devMA）：トレンド方向の相対位置
    for n in (5, 20, 60):
        ma = df["close"].rolling(n).mean()
        df[f"ma{n}"] = ma
        df[f"devMA{n}"] = df["close"]/ma - 1.0

    # 2) ボラティリティ系：直近の荒さを多面的に
    df["sigma5"]  = df["ret1"].rolling(5).std()
    df["sigma20"] = df["ret1"].rolling(20).std()
    bb = BollingerBands(df["close"], window=20)
    # バンド幅をmavgで正規化（価格水準の違いを吸収）
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-12)
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"] = atr.average_true_range()
    df["atr_ratio"] = df["atr"] / (df["close"] + 1e-12)
    df["range_ma_ratio"] = (df["high"] - df["low"]) / (df["close"].rolling(20).mean() + 1e-12)
    df["atr_diff"] = df["atr_ratio"].diff() # ボラの変化方向

    # 3) テクニカル：モメンタム/オシレータ
    df["rsi14"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["mom10"] = df["close"] - df["close"].shift(10)
    df["mom20"] = df["close"] - df["close"].shift(20)
    ema_short = EMAIndicator(df["close"], window=12).ema_indicator()
    ema_long  = EMAIndicator(df["close"], window=26).ema_indicator()
    df["ema_diff"] = ema_short - ema_long   # EMA短長差

    # 4) 出来高：出来高の水準変化とOBV系
    df["vol_chg"] = df["volume"].pct_change()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma20"] + 1e-12)
    obv = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["obv"] = obv
    df["obv_ma20"] = df["obv"].rolling(20).mean()
    df["obv_ratio"] = df["obv"] / (df["obv_ma20"] + 1e-12)

    # 5) ギャップ（寄り付き跳ね）
    df["gap"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-12)

    # 6) 市場要因は add_market_factors() で付与（順序上ここでは触れない）

    # 7) 簡易異常フラグ（バイナリ数値化）：極端日をモデルに明示
    vol_sigma20 = df["volume"].rolling(20).std()    # 出来高のローリング標準偏差（20日）
    gap_sigma60 = df["gap"].rolling(60).std()       # ギャップのローリング標準偏差（60日）
    df["price_anom"] = (df["ret1"].abs() > 2 * df["sigma20"]).astype(int)               # |日次リターン| > 2σ20
    df["vol_anom"]   = (df["volume"] > (df["vol_ma20"] + 2 * vol_sigma20)).astype(int)  # 出来高が平均+2σ20を超過
    df["gap_anom"]   = (df["gap"].abs() > 2 * gap_sigma60).astype(int)                  # ギャップが60日σの2倍超

    # 8) ローソク足の形状比率：ヒゲ比/実体比 → プライスアクションの微妙な差異を拾う
    df["body_len"]     = (df["close"] - df["open"]).abs()
    df["upper_shadow"] = df["high"] - pd.concat([df["close"], df["open"]], axis=1).max(axis=1)
    df["lower_shadow"] = pd.concat([df["close"], df["open"]], axis=1).min(axis=1) - df["low"]
    df["shadow_ratio"] = df["upper_shadow"] / (df["lower_shadow"].abs() + 1e-6)
    df["body_ratio"]   = df["body_len"] / ((df["high"] - df["low"]).abs() + 1e-6)
    df["shadow_ratio_chg"] = df["shadow_ratio"].pct_change()
    rng = (df["high"] - df["low"]).abs()
    df["upper_shadow_ratio"] = (df["upper_shadow"] / (rng + EPS)).clip(-5, 5)
    df["lower_shadow_ratio"] = (df["lower_shadow"] / (rng + EPS)).clip(-5, 5)

    # 9) ボラレジーム指標：短期/中期ボラの比率で局面（静/動）を粗く表現
    df["sigma_ratio"] = (df["sigma5"] / (df["sigma20"] + EPS)).clip(0, 10)

    # 10) トレンド構造：回帰傾きと最終点残差（30/60）
    for w in (30, 60):
        slope_col = f"slope{w}"
        resid_col = f"resid{w}"
        residz_col = f"{resid_col}_z"

        df[slope_col] = _rolling_slope(df["close"], w)
        df[resid_col] = _rolling_resid_last(df["close"], w)
        # 残差のZ化（スケール合わせ・外れ値耐性の強化）
        df[residz_col] = (df[resid_col] / (df[resid_col].rolling(w).std() + EPS)).clip(-5, 5)

    # 11) 出来高の週次比（5日平均比）：短期の相対増減
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_wk_ratio"] = (df["volume"] / (df["vol_ma5"] + EPS)).clip(0, 20)

    return df


def _dl(ticker, start, end):
    """補助ダウンローダ：指数などを単純取得して列正規化。"""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _ema(s: pd.Series, span: int) -> pd.Series:
    """単純なEMA。"""
    return s.ewm(span=span, adjust=False).mean()


def add_market_factors(df: pd.DataFrame) -> pd.DataFrame:
    """市場因子（TOPIX、日経225、S&P500、VIX）を安全に付与。"""
    start, end = df.index.min(), df.index.max()
    out = df.copy()

    topix = _dl("^TOPX", start, end)
    proxy_used = False
    if topix.empty or "Close" not in topix.columns or df.index.intersection(topix.index).empty:
        print("[WARN] ^TOPX unavailable → fallback to 1306.T (TOPIX ETF)")
        topix = _dl("1306.T", start, end)
        proxy_used = True
    if topix.empty or df.index.intersection(topix.index).empty:
        print("[WARN] skip ^TOPX (empty or no overlap)")
    else:
        topix = topix[["Close"]].rename(columns={"Close": "topix_close"})
        topix = topix.reindex(out.index).ffill(limit=5)
        topix["topix_ret1"] = topix["topix_close"].pct_change(fill_method=None)
        topix["topix_sigma20"] = topix["topix_ret1"].rolling(20).std()
        out = out.merge(topix[["topix_ret1", "topix_sigma20"]], left_index=True, right_index=True, how="left")
        if proxy_used:
            print("[INFO] TOPIX features sourced from 1306.T (proxy)")

    for name, ticker in {"nikkei": "^N225", "sp500": "^GSPC"}.items():
        ix = _dl(ticker, start, end)
        if ix.empty or out.index.intersection(ix.index).empty:
            print(f"[WARN] skip {ticker} (empty or no overlap)")
            continue
        ix = ix[["Close"]].rename(columns={"Close": f"{name}_close"})
        ix[f"{name}_ret1"] = ix[f"{name}_close"].pct_change(fill_method=None)
        ix[f"{name}_sigma20"] = ix[f"{name}_ret1"].rolling(20).std()
        out = out.merge(ix[[f"{name}_ret1", f"{name}_sigma20"]], left_index=True, right_index=True, how="left")

    vix = _dl("^VIX", start, end)
    if not vix.empty and not out.index.intersection(vix.index).empty:
        vix = vix[["Close"]].rename(columns={"Close": "vix"})
        vix["Date"] = vix.index
        out = out.copy()
        out["Date"] = out.index
        out = (
            out.reset_index(drop=True)
            .merge(vix.reset_index(drop=True), on="Date", how="left")
            .set_index("Date")
        )
        out.index = pd.to_datetime(out.index)
        out["vix_ret1"] = out["vix"].pct_change(fill_method=None)
    else:
        print("[WARN] skip ^VIX (empty or no overlap)")

    return out


def add_rel_strength10(df: pd.DataFrame) -> pd.DataFrame:
    """TOPIXに対する10日相対強度（銘柄10日リターン － TOPIX10日リターン）。"""
    out = df.copy()
    if "topix_ret1" in out.columns:
        topix_1p = (1.0 + out["topix_ret1"].fillna(0.0))
        out["topix_ret10"] = topix_1p.rolling(10, min_periods=10).apply(np.prod, raw=True) - 1.0
        out["rel_strength10"] = (out["ret10"] - out["topix_ret10"]).replace([np.inf, -np.inf], np.nan)
    return out


def make_binary_excess_labels(
    df: pd.DataFrame,
    *,
    horizon: int,
    k_tau: float,
    bench_col: str = "topix_ret1",
    use_k_tau_margin: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    H日先“超過収益（銘柄 - ベンチ）”による2値ラベル（0/1）を作成。
    元の各モデルと同じロジックを horizon/k_tau を引数化して提供。
    """
    H = horizon

    r_stock_H = df["close"].pct_change(H).shift(-H)

    if bench_col not in df.columns:
        raise ValueError(f"Benchmark return column '{bench_col}' not found in df.columns")

    bench_1p = (1.0 + df[bench_col].fillna(0.0))
    R_bench = bench_1p.rolling(window=H, min_periods=H).apply(np.prod, raw=True) - 1.0
    r_bench_H = R_bench.shift(-H)

    excess_H = r_stock_H - r_bench_H

    if use_k_tau_margin:
        if "ret1" not in df.columns:
            raise ValueError("df must contain 'ret1' (stock 1-day return) for margin calc.")
        excess1 = (df["ret1"] - df[bench_col]).astype(float)
        std_excess20 = excess1.rolling(20).std()
        margin = k_tau * std_excess20 * math.sqrt(H)
    else:
        margin = 0.0

    if isinstance(margin, (pd.Series, np.ndarray)):
        y = (excess_H > margin).astype("Int64")
    else:
        y = (excess_H > float(margin)).astype("Int64")

    abs_metric = excess_H.abs()
    return y, abs_metric


def to_windows(X: np.ndarray, y: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    """(時系列, 特徴) 配列を (B, T=win, F), (B,) に変換。"""
    xs, ys = [], []
    for t in range(win, len(X)):
        xs.append(X[t - win : t])
        ys.append(y[t])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=int)


def add_log1p_features(df: pd.DataFrame, candidates: Iterable[str], eps: float = 1e-8) -> pd.DataFrame:
    """非負の連続量に log1p 変換を追加（<col>_log1p）。元列は保持。"""
    out = df.copy()
    cols = [c for c in candidates if c in out.columns]
    if not cols:
        return out
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce").clip(lower=0)
        out[f"{c}_log1p"] = np.log1p(s + eps)
    return out


def make_trinary_labels(df: pd.DataFrame, horizon: int, k_tau: float) -> Tuple[pd.Series, pd.Series]:
    """DATA依存を排除し、horizon/k_tauを引数で受け取る版（ロジックは同じ）。"""
    r_H = df["close"].pct_change(horizon).shift(-horizon)
    sigma20 = df["ret1"].rolling(20).std()
    tau = k_tau * sigma20 * math.sqrt(horizon)

    y = pd.Series(index=df.index, dtype="Int64")
    y[r_H < -tau]       = 0
    y[r_H.abs() <= tau] = 1
    y[r_H >  tau]       = 2
    return y.astype("Int64"), r_H.abs()


def finalize_feature_columns(
    df: pd.DataFrame,
    feature_cols: List[str],
    use_topix_features: bool,
) -> List[str]:
    """存在する列だけ残し、必要ならTOPIX系(prefix topix_)を除外。"""
    cols = [c for c in feature_cols if (c in df.columns and df[c].notna().any())]
    if not use_topix_features:
        cols = [c for c in cols if not c.startswith("topix_")]
    return cols


def replace_inf_and_dropna(df: pd.DataFrame) -> pd.DataFrame:
    """inf→nan置換してdropna（ログは呼び出し側の責務）。"""
    out = df.replace([np.inf, -np.inf], np.nan)
    out = out.dropna()
    return out

def build_tabular_dataset(
    df: pd.DataFrame,
    *,
    y: pd.Series,
    abs_metric: pd.Series,
    feature_cols: List[str],
    use_topix_features: bool,
    use_log1p: bool,
    log1p_candidates: Iterable[str],
    extra_log1p_cols: Iterable[str] = ("vix_log1p",),
    nan_tail_days: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    feature_cols -> 実在列へ絞り込み -> log1p列を取り込み -> NaN/Inf処理 -> dropna -> X/y/abs を返す。
    返り値: (X_all, y_all, abs_all, feature_cols_final, data_df)
    """
    # 実在する列だけ採用（期間等により欠ける可能性に備える）
    base_cols = [c for c in feature_cols if (c in df.columns and df[c].notna().any())]

    log1p_cols: List[str] = []
    if use_log1p:
        for c in log1p_candidates:
            name = f"{c}_log1p"
            if name in df.columns and df[name].notna().any():
                log1p_cols.append(name)

    for name in list(extra_log1p_cols):
        if name in df.columns and df[name].notna().any() and name not in log1p_cols:
            log1p_cols.append(name)

    feature_cols_final = list(dict.fromkeys(base_cols + log1p_cols))
    feature_cols_final = finalize_feature_columns(df, feature_cols_final, use_topix_features=use_topix_features)
    if not feature_cols_final:
        raise RuntimeError("[ERR] No usable features after assembly.")

    # 特徴＋教師＋サブセット指標を縦に揃え、NaNとInfを安全に弾く（dropna前にNaNログ）
    data_before = pd.concat([df[feature_cols_final], y.rename("y"), abs_metric.rename("abs_next")], axis=1)
    data_before = data_before.replace([np.inf, -np.inf], np.nan)
    required = feature_cols_final + ["y", "abs_next"]

    na_total = data_before[required].isna().sum().sort_values(ascending=False)
    print("[NA] total NaNs per column (top 30):", na_total.head(30).to_dict())
    na_tail = data_before[required].tail(nan_tail_days).isna().sum().sort_values(ascending=False)
    print(f"[NA] last{nan_tail_days}d NaNs per column (top 30):", na_tail.head(30).to_dict())

    data = data_before.dropna(subset=required)

    X_all = data[feature_cols_final].values.astype(np.float32)
    y_all = data["y"].values.astype(int)
    abs_all = data["abs_next"].values.astype(np.float32)
    if X_all.shape[1] != len(feature_cols_final):
        raise RuntimeError(f"Dim mismatch: X_all {X_all.shape}, cols {len(feature_cols_final)}")

    return X_all, y_all, abs_all, feature_cols_final, data
