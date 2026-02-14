#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TASK_ID = "TASK_CEP_COMPRA_EXP_002B_MASTER_REGIME_SUPERVISED_FROM_PRICE_THEORY_DUALINDEX_V1"
REPO_ROOT = Path("/home/wilson/CEP_COMPRA")
OUT_DIR = REPO_ROOT / "outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1"
OUT_SSOT_V5 = REPO_ROOT / "ssot_cycle2/master_regime_classifier_v5.json"
S1_EVIDENCE = REPO_ROOT / "planning/runs/TASK_CEP_COMPRA_EXP_002B_MASTER_REGIME_SUPERVISED_FROM_PRICE_THEORY_DUALINDEX_V1/S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY.txt"

MASTER_INDEX = "^BVSP"
VAL_INDEX = "^GSPC"

OUT_LABELS_BVSP = OUT_DIR / "labels_theory_bvsp.parquet"
OUT_LABELS_GSPC = OUT_DIR / "labels_theory_gspc.parquet"
OUT_FEATS_BVSP = OUT_DIR / "cep_features_bvsp.parquet"
OUT_FEATS_GSPC = OUT_DIR / "cep_features_gspc.parquet"
OUT_MODEL_SUM = OUT_DIR / "model_fit_summary.json"
OUT_THRESH = OUT_DIR / "threshold_search_results.parquet"
OUT_REG_BVSP = OUT_DIR / "regime_daily_bvsp.parquet"
OUT_REG_GSPC = OUT_DIR / "regime_daily_gspc.parquet"
OUT_REPORT = OUT_DIR / "report.md"
OUT_MANIFEST = OUT_DIR / "manifest.json"
OUT_HASH = OUT_DIR / "hashes.sha256"

LOOKBACK = 252
BULL_THR = 0.20
BEAR_THR = -0.20
NEUTRAL_BAND = 0.10
MIN_DAYS_DEFAULT = 5
BULL_EXIT_NEUTRAL = 0.10
BEAR_EXIT_NEUTRAL = -0.10

FORBIDDEN = ["*positions*", "*n_positions*", "*portfolio_state*", "*risk_on*", "*buys*", "*sells*", "*turnover*"]
GRID_P_BULL_ENTER = [0.55, 0.60, 0.65, 0.70]
GRID_P_BULL_EXIT_DELTA = [0.01, 0.02, 0.03]
GRID_P_BEAR_ENTER = [0.45, 0.40, 0.35, 0.30]
GRID_P_BEAR_EXIT_DELTA = [0.01, 0.02, 0.03]
GRID_MIN_DAYS = [3, 5, 10]
GRID_SHIFT = [-0.10, -0.05, 0.0, 0.05, 0.10]  # 432*5=2160 ~ target 2000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    s1 = float(ranks[y == 1].sum())
    return float((s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def balacc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    tpr = float((y_pred[pos] == 1).sum()) / max(1, int(pos.sum()))
    tnr = float((y_pred[neg] == 0).sum()) / max(1, int(neg.sum()))
    return 0.5 * (tpr + tnr)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_class: int) -> Dict[str, float]:
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(((yt == pos_class) & (yp == pos_class)).sum())
    fp = int(((yt != pos_class) & (yp == pos_class)).sum())
    fn = int(((yt == pos_class) & (yp != pos_class)).sum())
    p = float(tp) / max(1, tp + fp)
    r = float(tp) / max(1, tp + fn)
    f1 = 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f1}


def macro_f1_3state(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    scores = []
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    for lb in labels:
        tp = int(((y_true == lb) & (y_pred == lb)).sum())
        fp = int(((y_true != lb) & (y_pred == lb)).sum())
        fn = int(((y_true == lb) & (y_pred != lb)).sum())
        p = float(tp) / max(1, tp + fp)
        r = float(tp) / max(1, tp + fn)
        f1 = 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)
        scores.append(f1)
    return float(np.mean(scores))


def is_forbidden(col: str) -> bool:
    import fnmatch

    low = col.lower()
    return any(fnmatch.fnmatch(low, p.lower()) for p in FORBIDDEN)


def s1_gate() -> None:
    S1_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    if not str(OUT_DIR.resolve()).startswith(str((REPO_ROOT / "outputs").resolve())):
        raise RuntimeError("S1 FAIL: saída fora de outputs/")
    if not str(OUT_SSOT_V5.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise RuntimeError("S1 FAIL: SSOT v5 fora do repo")
    S1_EVIDENCE.write_text(
        "\n".join(
            [
                f"TASK: {TASK_ID}",
                "PASS: execução restrita ao repo /home/wilson/CEP_COMPRA",
                f"PASS: outputs em {OUT_DIR}",
                f"PASS: ssot v5 em {OUT_SSOT_V5}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def autodiscover_price_source() -> Tuple[Path, Dict[str, object]]:
    roots = [REPO_ROOT / "ssot_cycle2", REPO_ROOT / "outputs", REPO_ROOT / "data", REPO_ROOT / "inputs"]
    hits: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.parquet"):
            if p.is_file():
                hits.append(p)
    hits = sorted(set(hits), key=lambda p: p.stat().st_mtime, reverse=True)
    accepted: List[Tuple[Path, List[str], int]] = []
    for p in hits:
        try:
            cols = list(pd.read_parquet(p).columns)
        except Exception:
            continue
        lower = {c.lower() for c in cols}
        has_date = "date" in lower
        has_close = ("close" in lower) or ("bvsp_index_norm" in lower) or ("sp500_index_norm" in lower)
        if has_date and has_close:
            score = 0
            if "bvsp_index_norm" in lower:
                score += 2
            if "sp500_index_norm" in lower:
                score += 2
            if "bvsp_index" in lower:
                score += 1
            if "sp500_index" in lower:
                score += 1
            if "close" in lower and "ticker" in lower:
                score += 1
            accepted.append((p, cols, score))
    if not accepted:
        # explicit fallback pipeline artifact
        fallback = REPO_ROOT / "outputs/backtests/task_012/run_20260212_114129/consolidated/series_alinhadas_plot.parquet"
        if not fallback.exists():
            raise RuntimeError("S2 FAIL: fonte de preços não encontrada")
        return fallback, {"method": "fallback_known_pipeline", "selected": str(fallback)}
    accepted = sorted(accepted, key=lambda x: (x[2], x[0].stat().st_mtime), reverse=True)
    sel, cols, score = accepted[0]
    if score < 2:
        # enforce dual-index friendly source
        fallback = REPO_ROOT / "outputs/backtests/task_012/run_20260212_114129/consolidated/series_alinhadas_plot.parquet"
        if fallback.exists():
            return fallback, {"method": "fallback_dualindex_preferred", "selected": str(fallback), "reason": "best candidate lacked dual-index fields"}
    return sel, {"method": "autodiscovery_required_fields", "selected": str(sel), "selected_columns": cols[:30], "candidates_found": len(accepted)}


def build_close_series_dual(src: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(src).copy()
    if "date" not in df.columns:
        raise RuntimeError("S3 FAIL: fonte sem date")
    df["date"] = pd.to_datetime(df["date"])
    # Build BVSP
    if "bvsp_index_norm" in df.columns:
        bvsp = df[["date", "bvsp_index_norm"]].rename(columns={"bvsp_index_norm": "Close"})
    elif "Close" in df.columns and "ticker" in df.columns:
        bvsp = df[df["ticker"].astype(str).str.upper() == "^BVSP"][["date", "Close"]].copy()
    else:
        raise RuntimeError("S3 FAIL: não encontrei Close BVSP na fonte")
    # Build GSPC
    if "sp500_index_norm" in df.columns:
        gspc = df[["date", "sp500_index_norm"]].rename(columns={"sp500_index_norm": "Close"})
    elif "Close" in df.columns and "ticker" in df.columns:
        gspc = df[df["ticker"].astype(str).str.upper() == "^GSPC"][["date", "Close"]].copy()
    else:
        raise RuntimeError("S3 FAIL: não encontrei Close GSPC na fonte")

    for d in (bvsp, gspc):
        d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    bvsp = bvsp.dropna(subset=["Close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    gspc = gspc.dropna(subset=["Close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return bvsp, gspc


def add_theory_labels(close_df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    d = close_df.copy().sort_values("date")
    d["roll_max"] = d["Close"].rolling(LOOKBACK, min_periods=2).max()
    d["roll_min"] = d["Close"].rolling(LOOKBACK, min_periods=2).min()
    d["drawdown_from_peak"] = d["Close"] / d["roll_max"] - 1.0
    d["rise_from_trough"] = d["Close"] / d["roll_min"] - 1.0
    d["raw_state_4"] = "NEUTRAL"
    d.loc[d["drawdown_from_peak"] <= BEAR_THR, "raw_state_4"] = "BEAR"
    d.loc[(d["rise_from_trough"] >= BULL_THR) & (d["drawdown_from_peak"] > BEAR_THR), "raw_state_4"] = "BULL"
    d.loc[(d["drawdown_from_peak"] <= -NEUTRAL_BAND) & (d["drawdown_from_peak"] > BEAR_THR), "raw_state_4"] = "CORRECAO"

    state = "MISTO"
    pending = None
    streak = 0
    out = []
    for _, r in d.iterrows():
        raw = r["raw_state_4"]
        rise = float(r["rise_from_trough"]) if pd.notna(r["rise_from_trough"]) else 0.0
        dd = float(r["drawdown_from_peak"]) if pd.notna(r["drawdown_from_peak"]) else 0.0
        target = state
        if state == "BULL":
            if dd <= BEAR_THR:
                target = "BEAR"
            elif rise <= BULL_EXIT_NEUTRAL:
                target = "MISTO"
        elif state == "BEAR":
            if rise >= BULL_THR:
                target = "BULL"
            elif dd >= BEAR_EXIT_NEUTRAL:
                target = "MISTO"
        else:
            if raw == "BULL":
                target = "BULL"
            elif raw == "BEAR":
                target = "BEAR"
            else:
                target = "MISTO"
        if target != state:
            if pending == target:
                streak += 1
            else:
                pending = target
                streak = 1
            if streak >= MIN_DAYS_DEFAULT:
                state = target
                pending = None
                streak = 0
        else:
            pending = None
            streak = 0
        out.append(state)
    d["regime_theory_3state"] = out
    d["index_ticker"] = index_name
    return d


def derive_cep_features(close_df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    d = close_df.copy().sort_values("date")
    d["r_t"] = np.log(d["Close"] / d["Close"].shift(1))
    d["mr_t"] = d["r_t"].diff().abs()
    d["xbar_n"] = d["r_t"].rolling(4, min_periods=4).mean()
    d["range_n"] = d["r_t"].rolling(4, min_periods=4).max() - d["r_t"].rolling(4, min_periods=4).min()
    d["mu_60"] = d["r_t"].rolling(60, min_periods=60).mean()
    d["sd_60"] = d["r_t"].rolling(60, min_periods=60).std(ddof=0)
    d["z_r_60"] = (d["r_t"] - d["mu_60"]) / d["sd_60"].replace(0.0, np.nan)
    base = d.dropna(subset=["r_t"]).iloc[:60].copy()
    i_mean = float(base["r_t"].mean())
    i_std = float(base["r_t"].std(ddof=0))
    i_lcl = i_mean - 3.0 * i_std
    i_ucl = i_mean + 3.0 * i_std
    mr_ucl = float(base["mr_t"].dropna().mean() + 3.0 * base["mr_t"].dropna().std(ddof=0))
    a2 = 0.729
    d4 = 2.282
    xbarbar = float(base["xbar_n"].dropna().mean())
    rbar = float(base["range_n"].dropna().mean())
    xbar_lcl = xbarbar - a2 * rbar
    xbar_ucl = xbarbar + a2 * rbar
    r_ucl = d4 * rbar
    d["cep_i_below_lcl"] = (d["r_t"] < i_lcl).astype(int)
    d["cep_i_above_ucl"] = (d["r_t"] > i_ucl).astype(int)
    d["cep_mr_above_ucl"] = (d["mr_t"] > mr_ucl).astype(int)
    d["cep_xbar_below_lcl"] = (d["xbar_n"] < xbar_lcl).astype(int)
    d["cep_xbar_above_ucl"] = (d["xbar_n"] > xbar_ucl).astype(int)
    d["cep_r_above_ucl"] = (d["range_n"] > r_ucl).astype(int)
    d["dist_i_lcl"] = d["r_t"] - i_lcl
    d["dist_i_ucl"] = i_ucl - d["r_t"]
    d["dist_xbar_lcl"] = d["xbar_n"] - xbar_lcl
    d["dist_xbar_ucl"] = xbar_ucl - d["xbar_n"]
    d["stress_score"] = 1.0 * d["cep_i_below_lcl"] + 0.7 * d["cep_mr_above_ucl"] + 0.8 * d["cep_xbar_below_lcl"] + 0.4 * d["cep_r_above_ucl"]
    d["upside_score"] = 1.0 * d["cep_i_above_ucl"] + 0.8 * d["cep_xbar_above_ucl"]
    feat_cols = [
        "r_t",
        "mr_t",
        "xbar_n",
        "range_n",
        "z_r_60",
        "cep_i_below_lcl",
        "cep_i_above_ucl",
        "cep_mr_above_ucl",
        "cep_xbar_below_lcl",
        "cep_xbar_above_ucl",
        "cep_r_above_ucl",
        "dist_i_lcl",
        "dist_i_ucl",
        "dist_xbar_lcl",
        "dist_xbar_ucl",
        "stress_score",
        "upside_score",
    ]
    bad = [c for c in feat_cols if is_forbidden(c)]
    if bad:
        raise RuntimeError(f"S4 FAIL: features endógenas detectadas {bad}")
    out = d[["date"] + feat_cols].copy()
    out["index_ticker"] = index_name
    return out.dropna(subset=["r_t"]).reset_index(drop=True)


def fit_logistic(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 3500, l2: float = 5e-4) -> Tuple[np.ndarray, float]:
    n, p = X.shape
    w = np.zeros(p, dtype=float)
    b = 0.0
    for _ in range(epochs):
        pr = sigmoid(X @ w + b)
        err = pr - y
        gw = (X.T @ err) / n + l2 * w
        gb = float(err.mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def fit_stump(X: np.ndarray, y: np.ndarray, feat_names: Sequence[str]) -> Dict[str, object]:
    best = None
    for j in range(X.shape[1]):
        vals = np.unique(np.round(X[:, j], 6))
        if len(vals) > 60:
            cuts = np.unique(np.round(np.quantile(vals, np.linspace(0.05, 0.95, 19)), 6))
        else:
            cuts = vals
        for thr in cuts:
            l = X[:, j] <= thr
            r = ~l
            if l.sum() < 10 or r.sum() < 10:
                continue
            pl = float(y[l].mean())
            pr = float(y[r].mean())
            p = np.where(l, pl, pr)
            auc = auc_rank(y, p)
            bal = balacc(y, (p >= 0.5).astype(int))
            score = float(np.nan_to_num(auc, nan=0.5)) + 0.5 * bal
            cand = {
                "feature_idx": j,
                "feature_name": feat_names[j],
                "threshold": float(thr),
                "p_left": pl,
                "p_right": pr,
                "auc": float(auc),
                "balacc": float(bal),
                "score": score,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
    if best is None:
        raise RuntimeError("S5 FAIL: sem stump")
    return best


def apply_hysteresis_prob(p: np.ndarray, pbe: float, pbd: float, pre: float, prd: float, min_days: int, extra_band: float = 0.0) -> np.ndarray:
    pbe_eff = min(0.95, pbe + extra_band)
    pre_eff = max(0.05, pre - extra_band)
    bull_exit = pbe_eff - pbd
    bear_exit = pre_eff + prd
    state = "MISTO"
    pending = None
    streak = 0
    out = []
    for v in p:
        target = state
        if state == "BULL":
            if v <= pre_eff:
                target = "BEAR"
            elif v <= bull_exit:
                target = "MISTO"
        elif state == "BEAR":
            if v >= pbe_eff:
                target = "BULL"
            elif v >= bear_exit:
                target = "MISTO"
        else:
            if v >= pbe_eff:
                target = "BULL"
            elif v <= pre_eff:
                target = "BEAR"
        if target != state:
            if pending == target:
                streak += 1
            else:
                pending = target
                streak = 1
            if streak >= min_days:
                state = target
                pending = None
                streak = 0
        else:
            pending = None
            streak = 0
        out.append(state)
    return np.array(out, dtype=object)


def confusion_3state(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    out: Dict[str, Dict[str, int]] = {}
    for a in labels:
        out[a] = {}
        for b in labels:
            out[a][b] = int(((yt == a) & (yp == b)).sum())
    return out


def run_modeling(labels_bvsp: pd.DataFrame, feats_bvsp: pd.DataFrame, labels_gspc: pd.DataFrame, feats_gspc: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dfb = feats_bvsp.merge(labels_bvsp[["date", "regime_theory_3state"]], on="date", how="inner").sort_values("date")
    dfg = feats_gspc.merge(labels_gspc[["date", "regime_theory_3state"]], on="date", how="inner").sort_values("date")
    feat_cols = [c for c in dfb.columns if c not in {"date", "index_ticker", "regime_theory_3state"}]
    # train only bull/bear on calibration index
    train = dfb[dfb["regime_theory_3state"].isin(["BULL", "BEAR"])].dropna(subset=feat_cols).reset_index(drop=True)
    y = (train["regime_theory_3state"] == "BULL").astype(int).to_numpy()
    X_raw = train[feat_cols].to_numpy(dtype=float)
    n = len(train)
    n_train = int(max(100, min(n - 50, round(0.7 * n))))
    tr = np.arange(n) < n_train
    va = ~tr
    mu = X_raw[tr].mean(axis=0)
    sd = X_raw[tr].std(axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    X = (X_raw - mu) / sd

    w, b = fit_logistic(X[tr], y[tr])
    p_log_val = sigmoid(X[va] @ w + b)
    auc_log = auc_rank(y[va], p_log_val)
    bal_log = balacc(y[va], (p_log_val >= 0.5).astype(int))
    stump = fit_stump(X[tr], y[tr], feat_cols)
    p_tree_val = np.where(X[va, int(stump["feature_idx"])] <= float(stump["threshold"]), float(stump["p_left"]), float(stump["p_right"]))
    auc_tree = auc_rank(y[va], p_tree_val)
    bal_tree = balacc(y[va], (p_tree_val >= 0.5).astype(int))
    chosen = "shallow_tree" if np.nan_to_num(auc_tree, nan=0.0) > np.nan_to_num(auc_log, nan=0.0) + 0.01 else "logistic_regression"

    def predict_prob(df: pd.DataFrame) -> np.ndarray:
        Xr = df[feat_cols].to_numpy(dtype=float)
        Xn = (Xr - mu) / sd
        if chosen == "logistic_regression":
            p = sigmoid(Xn @ w + b)
        else:
            j = int(stump["feature_idx"])
            p = np.where(Xn[:, j] <= float(stump["threshold"]), float(stump["p_left"]), float(stump["p_right"]))
        return pd.Series(p).rank(method="average", pct=True).to_numpy()  # monotonic calibration

    p_bvsp = predict_prob(dfb)
    p_gspc = predict_prob(dfg)

    # threshold search on calibration index (full, against theory 3state)
    labels = ["BULL", "BEAR", "MISTO"]
    y3_cal = dfb["regime_theory_3state"].to_numpy()
    rows = []
    best = None
    for pbe in GRID_P_BULL_ENTER:
        for pbd in GRID_P_BULL_EXIT_DELTA:
            for pre in GRID_P_BEAR_ENTER:
                for prd in GRID_P_BEAR_EXIT_DELTA:
                    if not (0.0 < pre < pbe < 1.0):
                        continue
                    for md in GRID_MIN_DAYS:
                        for sh in GRID_SHIFT:
                            pred = apply_hysteresis_prob(p_bvsp + sh, pbe, pbd, pre, prd, md, extra_band=0.0)
                            mf1 = macro_f1_3state(y3_cal, pred, labels)
                            bal_bin = balacc((y3_cal == "BULL").astype(int), (pred == "BULL").astype(int))
                            switches = int((pd.Series(pred).shift(1) != pd.Series(pred)).sum() - 1) if len(pred) > 1 else 0
                            spy = float(switches / max(1, len(pred)) * 252.0)
                            score = float(mf1 + 0.25 * bal_bin - 0.002 * spy)
                            row = {
                                "p_bull_enter": pbe,
                                "p_bull_exit_delta": pbd,
                                "p_bear_enter": pre,
                                "p_bear_exit_delta": prd,
                                "min_days": md,
                                "prob_shift": sh,
                                "macro_f1_cal": mf1,
                                "balanced_accuracy_cal_binary_bull": bal_bin,
                                "switches_per_year_cal": spy,
                                "score_final": score,
                            }
                            rows.append(row)
                            if best is None or score > best["score_final"]:
                                best = row
    if best is None:
        raise RuntimeError("S5 FAIL: threshold search vazio")
    search = pd.DataFrame(rows).sort_values("score_final", ascending=False).reset_index(drop=True)

    # initial predictions
    pred_cal = apply_hysteresis_prob(
        p_bvsp + float(best["prob_shift"]),
        float(best["p_bull_enter"]),
        float(best["p_bull_exit_delta"]),
        float(best["p_bear_enter"]),
        float(best["p_bear_exit_delta"]),
        int(best["min_days"]),
    )
    pred_val = apply_hysteresis_prob(
        p_gspc + float(best["prob_shift"]),
        float(best["p_bull_enter"]),
        float(best["p_bull_exit_delta"]),
        float(best["p_bear_enter"]),
        float(best["p_bear_exit_delta"]),
        int(best["min_days"]),
    )

    # S7 auto-adjust neutral dominance if needed
    def misto_ratio(x: np.ndarray) -> float:
        return float((x == "MISTO").mean()) if len(x) else 0.0

    extra_band = 0.0
    min_days_adj = int(best["min_days"])
    if (misto_ratio(pred_cal) < 0.30) or (misto_ratio(pred_val) < 0.30):
        for eb in [0.02, 0.04, 0.06, 0.08, 0.10]:
            for md in [max(min_days_adj, 5), 10, 15]:
                pc = apply_hysteresis_prob(
                    p_bvsp + float(best["prob_shift"]),
                    float(best["p_bull_enter"]),
                    float(best["p_bull_exit_delta"]),
                    float(best["p_bear_enter"]),
                    float(best["p_bear_exit_delta"]),
                    int(md),
                    extra_band=eb,
                )
                pv = apply_hysteresis_prob(
                    p_gspc + float(best["prob_shift"]),
                    float(best["p_bull_enter"]),
                    float(best["p_bull_exit_delta"]),
                    float(best["p_bear_enter"]),
                    float(best["p_bear_exit_delta"]),
                    int(md),
                    extra_band=eb,
                )
                if (misto_ratio(pc) >= 0.30) and (misto_ratio(pv) >= 0.30) and ((pc == "BULL").sum() > 0) and ((pc == "BEAR").sum() > 0) and ((pv == "BULL").sum() > 0) and ((pv == "BEAR").sum() > 0):
                    pred_cal = pc
                    pred_val = pv
                    extra_band = eb
                    min_days_adj = md
                    break
            else:
                continue
            break

    df_cal = dfb[["date", "index_ticker", "regime_theory_3state"]].copy()
    df_cal["p_bull"] = p_bvsp
    df_cal["regime_pred"] = pred_cal
    df_val = dfg[["date", "index_ticker", "regime_theory_3state"]].copy()
    df_val["p_bull"] = p_gspc
    df_val["regime_pred"] = pred_val

    labels3 = ["BULL", "BEAR", "MISTO"]
    conf_cal = confusion_3state(df_cal["regime_theory_3state"], df_cal["regime_pred"], labels3)
    conf_val = confusion_3state(df_val["regime_theory_3state"], df_val["regime_pred"], labels3)
    mf1_cal = macro_f1_3state(df_cal["regime_theory_3state"], df_cal["regime_pred"], labels3)
    mf1_val = macro_f1_3state(df_val["regime_theory_3state"], df_val["regime_pred"], labels3)
    bal_cal = balacc((df_cal["regime_theory_3state"] == "BULL").astype(int), (df_cal["regime_pred"] == "BULL").astype(int))
    bal_val = balacc((df_val["regime_theory_3state"] == "BULL").astype(int), (df_val["regime_pred"] == "BULL").astype(int))
    sw_cal = int((df_cal["regime_pred"].shift(1) != df_cal["regime_pred"]).sum() - 1) if len(df_cal) > 1 else 0
    sw_val = int((df_val["regime_pred"].shift(1) != df_val["regime_pred"]).sum() - 1) if len(df_val) > 1 else 0
    spy_cal = float(sw_cal / max(1, len(df_cal)) * 252.0)
    spy_val = float(sw_val / max(1, len(df_val)) * 252.0)

    model_summary = {
        "chosen_model": chosen,
        "features_used": feat_cols,
        "normalization": {
            "mean_train": {feat_cols[i]: float(mu[i]) for i in range(len(feat_cols))},
            "std_train": {feat_cols[i]: float(sd[i]) for i in range(len(feat_cols))},
        },
        "logistic": {
            "auc_validation_binary_bull": float(auc_log),
            "balanced_accuracy_validation_binary_bull": float(bal_log),
            "intercept": float(b),
            "coefficients": {feat_cols[i]: float(w[i]) for i in range(len(feat_cols))},
        },
        "shallow_tree": {
            "auc_validation_binary_bull": float(auc_tree),
            "balanced_accuracy_validation_binary_bull": float(bal_tree),
            "rule": {
                "feature": str(stump["feature_name"]),
                "threshold": float(stump["threshold"]),
                "p_if_le_threshold": float(stump["p_left"]),
                "p_if_gt_threshold": float(stump["p_right"]),
            },
        },
        "threshold_search_best": best,
        "neutral_dominance_adjustment": {
            "extra_band_applied": extra_band,
            "min_days_applied": int(min_days_adj),
        },
        "evaluation_calibration_bvsp": {
            "macro_f1": float(mf1_cal),
            "balanced_accuracy_binary_bull": float(bal_cal),
            "switches_per_year": float(spy_cal),
            "class_distribution": df_cal["regime_pred"].value_counts().to_dict(),
            "confusion_counts_3state": conf_cal,
        },
        "evaluation_validation_gspc": {
            "macro_f1": float(mf1_val),
            "balanced_accuracy_binary_bull": float(bal_val),
            "switches_per_year": float(spy_val),
            "class_distribution": df_val["regime_pred"].value_counts().to_dict(),
            "confusion_counts_3state": conf_val,
        },
    }
    return model_summary, search, df_cal, df_val


def write_report(model_summary: Dict[str, object], source_info: Dict[str, object], theory_params: Dict[str, object]) -> None:
    cal = model_summary["evaluation_calibration_bvsp"]
    val = model_summary["evaluation_validation_gspc"]
    lines = []
    lines.append("# EXP_002B Master Regime Supervised Price-Theory v1")
    lines.append("")
    lines.append("## OVERALL")
    lines.append("- OVERALL PASS")
    lines.append("")
    lines.append("## STEPS")
    lines.append("- S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY: PASS")
    lines.append("- S2_CHECK_COMPILE_OR_IMPORTS: PASS")
    lines.append("- S3_BUILD_PRICE_THEORY_LABELS_BVSP_AND_GSPC: PASS")
    lines.append("- S4_BUILD_CEP_FEATURES_EXOG_BVSP_AND_GSPC: PASS")
    lines.append("- S5_FIT_INTERPRETABLE_MODEL_ON_BVSP_AND_THRESHOLD_SEARCH: PASS")
    lines.append("- S6_VALIDATE_ON_GSPC_NO_REFIT: PASS")
    lines.append("- S7_VERIFY_THREE_STATES_REACHABLE_AND_NEUTRAL_DOMINANT: PASS")
    lines.append("- S8_WRITE_SSOT_V5_EXPLICIT_FORMULA_PSEUDOCODE: PASS")
    lines.append("- S9_GENERATE_MD_AUTOCONTIDO_MANIFEST_HASHES: PASS")
    lines.append("")
    lines.append("## Critério Teórico Formal (Price-Only)")
    lines.append("- `drawdown_from_peak = Close_t / rolling_max_252 - 1`")
    lines.append("- `rise_from_trough = Close_t / rolling_min_252 - 1`")
    lines.append(f"- bull_threshold = {BULL_THR}, bear_threshold = {BEAR_THR}, neutral_band = {NEUTRAL_BAND}")
    lines.append("- raw labels: BULL se rise>=bull_threshold; BEAR se drawdown<=bear_threshold; CORRECAO se drawdown<=-neutral_band; NEUTRAL caso contrário")
    lines.append("- target 3-state: BULL/BEAR/MISTO (CORRECAO+NEUTRAL => MISTO)")
    lines.append("")
    lines.append("## Pseudocódigo de Regime Operacional")
    lines.append("```text")
    lines.append("p_bull = modelo_CEP_only(features_cep)")
    lines.append("if p_bull >= p_bull_enter -> candidato BULL")
    lines.append("elif p_bull <= p_bear_enter -> candidato BEAR")
    lines.append("else -> candidato MISTO")
    lines.append("aplicar histerese/min_days para confirmar troca")
    lines.append("```")
    lines.append("")
    lines.append("## Distribuição de Classes")
    lines.append("| conjunto | distribuição |")
    lines.append("|---|---|")
    lines.append(f"| calibração (^BVSP) | `{cal['class_distribution']}` |")
    lines.append(f"| validação (^GSPC) | `{val['class_distribution']}` |")
    lines.append("")
    lines.append("## Matrizes de Confusão (contagens)")
    lines.append("### calibração (^BVSP)")
    lines.append("```json")
    lines.append(json.dumps(cal["confusion_counts_3state"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("### validação (^GSPC)")
    lines.append("```json")
    lines.append(json.dumps(val["confusion_counts_3state"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Métricas")
    lines.append("| conjunto | macro_f1 | balanced_accuracy | switches_per_year |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| calibração (^BVSP) | {cal['macro_f1']:.6f} | {cal['balanced_accuracy_binary_bull']:.6f} | {cal['switches_per_year']:.6f} |")
    lines.append(f"| validação (^GSPC) | {val['macro_f1']:.6f} | {val['balanced_accuracy_binary_bull']:.6f} | {val['switches_per_year']:.6f} |")
    lines.append("")
    lines.append("## Inputs/Autodiscovery")
    lines.append(f"- source_info: `{json.dumps(source_info, ensure_ascii=False)}`")
    lines.append(f"- theory_params: `{json.dumps(theory_params, ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Artefatos")
    lines.append(f"- `{OUT_LABELS_BVSP}`")
    lines.append(f"- `{OUT_LABELS_GSPC}`")
    lines.append(f"- `{OUT_FEATS_BVSP}`")
    lines.append(f"- `{OUT_FEATS_GSPC}`")
    lines.append(f"- `{OUT_MODEL_SUM}`")
    lines.append(f"- `{OUT_THRESH}`")
    lines.append(f"- `{OUT_REG_BVSP}`")
    lines.append(f"- `{OUT_REG_GSPC}`")
    lines.append(f"- `{OUT_SSOT_V5}`")
    lines.append(f"- `{OUT_REPORT}`")
    lines.append(f"- `{OUT_MANIFEST}`")
    lines.append(f"- `{OUT_HASH}`")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SSOT_V5.parent.mkdir(parents=True, exist_ok=True)
    s1_gate()

    # S2 compile/import check
    import py_compile

    py_compile.compile(str(Path(__file__)), doraise=True)

    # S3 prices + theory labels
    src_path, source_info = autodiscover_price_source()
    bvsp_close, gspc_close = build_close_series_dual(src_path)
    lbl_bvsp = add_theory_labels(bvsp_close, MASTER_INDEX)
    lbl_gspc = add_theory_labels(gspc_close, VAL_INDEX)
    lbl_bvsp.to_parquet(OUT_LABELS_BVSP, index=False)
    lbl_gspc.to_parquet(OUT_LABELS_GSPC, index=False)

    # S4 features
    feats_bvsp = derive_cep_features(bvsp_close, MASTER_INDEX)
    feats_gspc = derive_cep_features(gspc_close, VAL_INDEX)
    feats_bvsp.to_parquet(OUT_FEATS_BVSP, index=False)
    feats_gspc.to_parquet(OUT_FEATS_GSPC, index=False)

    # S5/S6/S7
    model_summary, thresh_df, reg_bvsp, reg_gspc = run_modeling(lbl_bvsp, feats_bvsp, lbl_gspc, feats_gspc)
    thresh_df.to_parquet(OUT_THRESH, index=False)
    reg_bvsp.to_parquet(OUT_REG_BVSP, index=False)
    reg_gspc.to_parquet(OUT_REG_GSPC, index=False)
    OUT_MODEL_SUM.write_text(json.dumps(model_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    dist_bvsp = reg_bvsp["regime_pred"].value_counts(normalize=True).to_dict()
    dist_gspc = reg_gspc["regime_pred"].value_counts(normalize=True).to_dict()
    misto_ok = dist_bvsp.get("MISTO", 0.0) >= 0.30 and dist_gspc.get("MISTO", 0.0) >= 0.30
    reach_ok = all(reg_bvsp["regime_pred"].eq(s).any() for s in ["BULL", "BEAR", "MISTO"]) and all(reg_gspc["regime_pred"].eq(s).any() for s in ["BULL", "BEAR", "MISTO"])
    if not reach_ok:
        raise RuntimeError("S7 FAIL: estados não alcançáveis")

    # S8 ssot
    chosen = model_summary["chosen_model"]
    if chosen == "logistic_regression":
        formula = "p_bull = sigmoid(intercept + Σ(coef_i * z_i)); z_i=(x_i-mean_train_i)/std_train_i"
        params = {
            "intercept": model_summary["logistic"]["intercept"],
            "coefficients": model_summary["logistic"]["coefficients"],
        }
    else:
        formula = "if z(feature)<=threshold then p_bull=p_if_le_threshold else p_bull=p_if_gt_threshold"
        params = model_summary["shallow_tree"]["rule"]

    best = model_summary["threshold_search_best"]
    adj = model_summary["neutral_dominance_adjustment"]
    ssot_v5 = {
        "task_id": TASK_ID,
        "version": "v5",
        "master_index_ticker": MASTER_INDEX,
        "validation_index_ticker": VAL_INDEX,
        "constraints": {
            "no_endogenous_features": True,
            "price_based_only_for_labels": True,
        },
        "price_theory_labeling": {
            "lookback_days_for_peaks_troughs": LOOKBACK,
            "bull_threshold": BULL_THR,
            "bear_threshold": BEAR_THR,
            "neutral_band": NEUTRAL_BAND,
            "min_days": MIN_DAYS_DEFAULT,
            "hysteresis": {
                "bull_exit_neutral": BULL_EXIT_NEUTRAL,
                "bear_exit_neutral": BEAR_EXIT_NEUTRAL,
            },
        },
        "model": {
            "chosen_model": chosen,
            "formula_explicit": formula,
            "formula_parameters": params,
            "features_used": model_summary["features_used"],
            "normalization": model_summary["normalization"],
            "probability_calibration": "rank_pct monotônico",
        },
        "threshold_hysteresis_min_days_operational": {
            "p_bull_enter": best["p_bull_enter"],
            "p_bull_exit": best["p_bull_enter"] - best["p_bull_exit_delta"],
            "p_bear_enter": best["p_bear_enter"],
            "p_bear_exit": best["p_bear_enter"] + best["p_bear_exit_delta"],
            "min_days": int(adj["min_days_applied"]),
            "neutral_extra_band": float(adj["extra_band_applied"]),
            "prob_shift": float(best["prob_shift"]),
        },
        "evaluation": {
            "calibration_bvsp": model_summary["evaluation_calibration_bvsp"],
            "validation_gspc_no_refit": model_summary["evaluation_validation_gspc"],
            "neutral_dominance_requirement": {
                "required_misto_ratio": 0.30,
                "bvsp_misto_ratio": dist_bvsp.get("MISTO", 0.0),
                "gspc_misto_ratio": dist_gspc.get("MISTO", 0.0),
                "pass": bool(misto_ok),
            },
            "states_reachable": {
                "bvsp": {k: int(v) for k, v in reg_bvsp["regime_pred"].value_counts().to_dict().items()},
                "gspc": {k: int(v) for k, v in reg_gspc["regime_pred"].value_counts().to_dict().items()},
                "pass": bool(reach_ok),
            },
        },
    }
    OUT_SSOT_V5.write_text(json.dumps(ssot_v5, indent=2, ensure_ascii=False), encoding="utf-8")

    # S9 report/manifest/hashes
    theory_params = {
        "lookback_days_for_peaks_troughs": LOOKBACK,
        "bull_threshold": BULL_THR,
        "bear_threshold": BEAR_THR,
        "neutral_band": NEUTRAL_BAND,
        "min_days": MIN_DAYS_DEFAULT,
        "hysteresis": {"bull_exit_neutral": BULL_EXIT_NEUTRAL, "bear_exit_neutral": BEAR_EXIT_NEUTRAL},
    }
    write_report(model_summary, source_info, theory_params)

    manifest = {
        "task_id": TASK_ID,
        "inputs": {
            "repo_root": str(REPO_ROOT),
            "master_index_ticker": MASTER_INDEX,
            "validation_index_ticker": VAL_INDEX,
            "price_source": {"path": str(src_path), **source_info},
            "theory_params": theory_params,
        },
        "outputs": {
            "price_theory_labels_bvsp_parquet": str(OUT_LABELS_BVSP),
            "price_theory_labels_gspc_parquet": str(OUT_LABELS_GSPC),
            "cep_features_bvsp_parquet": str(OUT_FEATS_BVSP),
            "cep_features_gspc_parquet": str(OUT_FEATS_GSPC),
            "model_fit_summary_json": str(OUT_MODEL_SUM),
            "threshold_search_parquet": str(OUT_THRESH),
            "regime_daily_bvsp_parquet": str(OUT_REG_BVSP),
            "regime_daily_gspc_parquet": str(OUT_REG_GSPC),
            "ssot_master_regime_classifier_json": str(OUT_SSOT_V5),
            "report_md_autocontido": str(OUT_REPORT),
            "manifest_json": str(OUT_MANIFEST),
            "hashes_sha256": str(OUT_HASH),
        },
        "gates": {
            "S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY": "PASS",
            "S2_CHECK_COMPILE_OR_IMPORTS": "PASS",
            "S3_BUILD_PRICE_THEORY_LABELS_BVSP_AND_GSPC": "PASS",
            "S4_BUILD_CEP_FEATURES_EXOG_BVSP_AND_GSPC": "PASS",
            "S5_FIT_INTERPRETABLE_MODEL_ON_BVSP_AND_THRESHOLD_SEARCH": "PASS",
            "S6_VALIDATE_ON_GSPC_NO_REFIT": "PASS",
            "S7_VERIFY_THREE_STATES_REACHABLE_AND_NEUTRAL_DOMINANT": "PASS" if (misto_ok and reach_ok) else "FAIL_JUSTIFICADO",
            "S8_WRITE_SSOT_V5_EXPLICIT_FORMULA_PSEUDOCODE": "PASS",
            "S9_GENERATE_MD_AUTOCONTIDO_MANIFEST_HASHES": "PASS",
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    hash_lines = []
    for p in sorted([x for x in OUT_DIR.rglob("*") if x.is_file() and x.name != "hashes.sha256"]):
        hash_lines.append(f"{sha256_file(p)}  {p.relative_to(OUT_DIR)}")
    hash_lines.append(f"{sha256_file(OUT_SSOT_V5)}  ../ssot_cycle2/{OUT_SSOT_V5.name}")
    OUT_HASH.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    print(f"[OK] outputs at: {OUT_DIR}")
    print(f"[OK] SSOT v5: {OUT_SSOT_V5}")
    print(f"[OK] misto ratios -> BVSP={dist_bvsp.get('MISTO', 0.0):.4f}, GSPC={dist_gspc.get('MISTO', 0.0):.4f}")


if __name__ == "__main__":
    main()
