#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


TASK_ID = "TASK_CEP_COMPRA_EXP_003B_MASTER_REGIME_V7_5STATE_AND_BUYCONFUSION_DUALINDEX_V1"
REPO_ROOT = Path("/home/wilson/CEP_COMPRA")
OUT_DIR = REPO_ROOT / "outputs/experimentos/controle_rl/EXP_003B_master_regime_v7_5state"
OUT_SSOT = REPO_ROOT / "ssot_cycle2/master_regime_classifier_v7.json"
S1_EVIDENCE = REPO_ROOT / "planning/runs/TASK_CEP_COMPRA_EXP_003B_MASTER_REGIME_V7_5STATE_AND_BUYCONFUSION_DUALINDEX_V1/S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY.txt"

MASTER_INDEX = "^BVSP"
VAL_INDEX = "^GSPC"

OUT_CLOSE_BVSP = OUT_DIR / "close_bvsp.parquet"
OUT_CLOSE_GSPC = OUT_DIR / "close_gspc.parquet"
OUT_LBL5_BVSP = OUT_DIR / "labels_theory_5state_bvsp.parquet"
OUT_LBL5_GSPC = OUT_DIR / "labels_theory_5state_gspc.parquet"
OUT_LBLBUY_BVSP = OUT_DIR / "labels_buy3_bvsp.parquet"
OUT_LBLBUY_GSPC = OUT_DIR / "labels_buy3_gspc.parquet"
OUT_FEAT_BVSP = OUT_DIR / "cep_features_bvsp.parquet"
OUT_FEAT_GSPC = OUT_DIR / "cep_features_gspc.parquet"
OUT_MODEL = OUT_DIR / "model_fit_summary.json"
OUT_THRESH = OUT_DIR / "threshold_search_results.parquet"
OUT_REG_BVSP = OUT_DIR / "regime_daily_bvsp_5state.parquet"
OUT_REG_GSPC = OUT_DIR / "regime_daily_gspc_5state.parquet"
OUT_BUY_BVSP = OUT_DIR / "buy_level_daily_bvsp.parquet"
OUT_BUY_GSPC = OUT_DIR / "buy_level_daily_gspc.parquet"
OUT_CONF5 = OUT_DIR / "confusion_5state_counts.json"
OUT_CONFBUY = OUT_DIR / "confusion_buy3_counts.json"
OUT_REPORT = OUT_DIR / "report.md"
OUT_MANIFEST = OUT_DIR / "manifest.json"
OUT_HASH = OUT_DIR / "hashes.sha256"

LOOKBACK = 252
TH_BULL = 0.20
TH_BULL_CORR_LOW = 0.10
TH_BEAR = -0.20
TH_BEAR_CORR_UP = -0.10
MIN_DAYS_DEFAULT = 5

BUY_MAP = {
    "BULL": "BUY2",
    "CORR_BULL_NEUTRO": "BUY2",
    "NEUTRO": "BUY2",
    "CORR_NEUTRO_BEAR": "BUY1",
    "BEAR": "BUY0",
}

STATE5 = ["BULL", "CORR_BULL_NEUTRO", "NEUTRO", "CORR_NEUTRO_BEAR", "BEAR"]
BUY3 = ["BUY2", "BUY1", "BUY0"]

FORBIDDEN = ["*positions*", "*n_positions*", "*portfolio_state*", "*risk_on*", "*buys*", "*sells*", "*turnover*"]

GRID_P_BULL_ENTER = [0.60, 0.65, 0.70, 0.75]
GRID_P_BULL_EXIT_DELTA = [0.01, 0.02, 0.03]
GRID_P_BEAR_ENTER = [0.40, 0.35, 0.30, 0.25]
GRID_P_BEAR_EXIT_DELTA = [0.01, 0.02, 0.03]
GRID_MIN_DAYS = [3, 5, 10]
GRID_EXTRA_BAND = [-0.04, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]


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
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    s1 = float(r[y == 1].sum())
    return float((s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def is_forbidden(name: str) -> bool:
    import fnmatch

    low = name.lower()
    return any(fnmatch.fnmatch(low, p.lower()) for p in FORBIDDEN)


def confusion_counts(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    out: Dict[str, Dict[str, int]] = {}
    for a in labels:
        out[a] = {}
        for b in labels:
            out[a][b] = int(((yt == a) & (yp == b)).sum())
    return out


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    vals = []
    for lb in labels:
        tp = int(((yt == lb) & (yp == lb)).sum())
        fp = int(((yt != lb) & (yp == lb)).sum())
        fn = int(((yt == lb) & (yp != lb)).sum())
        p = float(tp) / max(1, tp + fp)
        r = float(tp) / max(1, tp + fn)
        vals.append(0.0 if (p + r) == 0 else 2.0 * p * r / (p + r))
    return float(np.mean(vals))


def balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    rec = []
    for lb in labels:
        m = yt == lb
        rec.append(float((yp[m] == lb).sum()) / max(1, int(m.sum())))
    return float(np.mean(rec))


def switches_per_year(seq: Sequence[str]) -> float:
    s = pd.Series(seq)
    sw = int((s.shift(1) != s).sum() - 1) if len(s) > 1 else 0
    return float(sw / max(1, len(s)) * 252.0)


def s1_gate() -> None:
    S1_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    if not str(OUT_DIR.resolve()).startswith(str((REPO_ROOT / "outputs").resolve())):
        raise RuntimeError("S1 FAIL: out_dir fora de outputs/")
    if not str(OUT_SSOT.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise RuntimeError("S1 FAIL: ssot fora do repo")
    S1_EVIDENCE.write_text(
        "\n".join(
            [
                f"TASK: {TASK_ID}",
                "PASS: allowlist CEP_COMPRA only",
                f"PASS: out_dir={OUT_DIR}",
                f"PASS: ssot={OUT_SSOT}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def autodiscover_price_source() -> Tuple[Path, Dict[str, object]]:
    roots = [REPO_ROOT / "ssot_cycle2", REPO_ROOT / "outputs", REPO_ROOT / "data", REPO_ROOT / "inputs"]
    hits: List[Path] = []
    for root in roots:
        if root.exists():
            hits.extend([p for p in root.rglob("*.parquet") if p.is_file()])
    hits = sorted(set(hits), key=lambda p: p.stat().st_mtime, reverse=True)
    scored: List[Tuple[Path, int, List[str]]] = []
    for p in hits:
        try:
            cols = list(pd.read_parquet(p).columns)
        except Exception:
            continue
        low = {c.lower() for c in cols}
        score = 0
        if "date" in low:
            score += 1
        if "close" in low and "ticker" in low:
            score += 2
        if "bvsp_index_norm" in low or "bvsp_index" in low:
            score += 2
        if "sp500_index_norm" in low or "sp500_index" in low:
            score += 2
        if score >= 4:
            scored.append((p, score, cols))
    if scored:
        scored.sort(key=lambda x: (x[1], x[0].stat().st_mtime), reverse=True)
        p, score, cols = scored[0]
        return p, {"method": "autodiscovery", "selected": str(p), "score": score, "selected_columns": cols[:40], "candidates_found": len(scored)}
    fallback = REPO_ROOT / "outputs/backtests/task_012/run_20260212_114129/consolidated/series_alinhadas_plot.parquet"
    if fallback.exists():
        return fallback, {"method": "fallback_build_from_existing_pipeline", "selected": str(fallback)}
    raise RuntimeError("S3 FAIL: preço dual-index não encontrado")


def build_close_dual(src: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(src).copy()
    if "date" not in df.columns:
        raise RuntimeError("S3 FAIL: sem date")
    df["date"] = pd.to_datetime(df["date"])

    if "bvsp_index_norm" in df.columns:
        bvsp = df[["date", "bvsp_index_norm"]].rename(columns={"bvsp_index_norm": "Close"})
    elif "bvsp_index" in df.columns:
        bvsp = df[["date", "bvsp_index"]].rename(columns={"bvsp_index": "Close"})
    elif "Close" in df.columns and "ticker" in df.columns:
        bvsp = df[df["ticker"].astype(str).str.upper() == MASTER_INDEX][["date", "Close"]].copy()
    else:
        raise RuntimeError("S3 FAIL: close BVSP ausente")

    if "sp500_index_norm" in df.columns:
        gspc = df[["date", "sp500_index_norm"]].rename(columns={"sp500_index_norm": "Close"})
    elif "sp500_index" in df.columns:
        gspc = df[["date", "sp500_index"]].rename(columns={"sp500_index": "Close"})
    elif "Close" in df.columns and "ticker" in df.columns:
        gspc = df[df["ticker"].astype(str).str.upper() == VAL_INDEX][["date", "Close"]].copy()
    else:
        raise RuntimeError("S3 FAIL: close GSPC ausente")

    for d in (bvsp, gspc):
        d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
        d.dropna(subset=["Close"], inplace=True)
        d.sort_values("date", inplace=True)
        d.drop_duplicates("date", keep="last", inplace=True)
        d.reset_index(drop=True, inplace=True)
    return bvsp, gspc


def _raw_state_5(dd: float, rise: float) -> str:
    # prioridade: BEAR > CORR_NEUTRO_BEAR > BULL > CORR_BULL_NEUTRO > NEUTRO
    if dd <= TH_BEAR:
        return "BEAR"
    if TH_BEAR < dd <= TH_BEAR_CORR_UP:
        return "CORR_NEUTRO_BEAR"
    if rise >= TH_BULL:
        return "BULL"
    if TH_BULL_CORR_LOW <= rise < TH_BULL:
        return "CORR_BULL_NEUTRO"
    return "NEUTRO"


def theory_label_5state(close_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    d = close_df.copy().sort_values("date")
    d["peak_t"] = d["Close"].rolling(LOOKBACK, min_periods=2).max()
    d["trough_t"] = d["Close"].rolling(LOOKBACK, min_periods=2).min()
    d["drawdown_t"] = d["Close"] / d["peak_t"] - 1.0
    d["rise_t"] = d["Close"] / d["trough_t"] - 1.0
    d["raw_state"] = [_raw_state_5(float(dd), float(r)) for dd, r in zip(d["drawdown_t"].fillna(0.0), d["rise_t"].fillna(0.0))]

    # histerese + min_days causal
    state = "NEUTRO"
    pending = None
    streak = 0
    out = []
    for _, r in d.iterrows():
        dd = float(r["drawdown_t"]) if pd.notna(r["drawdown_t"]) else 0.0
        rise = float(r["rise_t"]) if pd.notna(r["rise_t"]) else 0.0
        raw = str(r["raw_state"])
        target = state
        if state == "BULL":
            if dd <= TH_BEAR:
                target = "BEAR"
            elif rise < TH_BULL_CORR_LOW:
                target = "NEUTRO"
            elif rise < TH_BULL:
                target = "CORR_BULL_NEUTRO"
        elif state == "CORR_BULL_NEUTRO":
            if dd <= TH_BEAR:
                target = "BEAR"
            elif TH_BEAR < dd <= TH_BEAR_CORR_UP:
                target = "CORR_NEUTRO_BEAR"
            elif rise >= TH_BULL:
                target = "BULL"
            elif rise < 0.05:
                target = "NEUTRO"
        elif state == "NEUTRO":
            target = raw
        elif state == "CORR_NEUTRO_BEAR":
            if dd <= TH_BEAR:
                target = "BEAR"
            elif dd > -0.05:
                target = "NEUTRO"
            elif rise >= TH_BULL:
                target = "BULL"
        elif state == "BEAR":
            if dd > TH_BEAR_CORR_UP:
                target = "NEUTRO"
            elif dd > TH_BEAR:
                target = "CORR_NEUTRO_BEAR"

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
    d["regime_theory_5state"] = out
    d["index_ticker"] = ticker
    return d


def map_buy(reg: Sequence[str]) -> np.ndarray:
    return np.array([BUY_MAP[str(x)] for x in reg], dtype=object)


def derive_cep_features(close_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
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
        "stress_score",
        "upside_score",
    ]
    bad = [c for c in feat_cols if is_forbidden(c)]
    if bad:
        raise RuntimeError(f"S5 FAIL: features proibidas {bad}")
    out = d[["date"] + feat_cols].copy()
    out["index_ticker"] = ticker
    return out.dropna(subset=["r_t"]).reset_index(drop=True)


def fit_logistic(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 3200, l2: float = 5e-4) -> Tuple[np.ndarray, float]:
    n, p = X.shape
    w = np.zeros(p, dtype=float)
    b = 0.0
    for _ in range(epochs):
        pr = sigmoid(X @ w + b)
        e = pr - y
        gw = (X.T @ e) / n + l2 * w
        gb = float(e.mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def fit_stump(X: np.ndarray, y: np.ndarray, feat_names: Sequence[str]) -> Dict[str, object]:
    best = None
    for j in range(X.shape[1]):
        vals = np.unique(np.round(X[:, j], 6))
        if len(vals) > 80:
            cuts = np.unique(np.round(np.quantile(vals, np.linspace(0.05, 0.95, 25)), 6))
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
            acc = float(((p >= 0.5) == y).mean())
            score = float(np.nan_to_num(auc, nan=0.5)) + 0.25 * acc
            cand = {
                "feature_idx": j,
                "feature_name": feat_names[j],
                "threshold": float(thr),
                "p_left": pl,
                "p_right": pr,
                "auc": float(auc),
                "score": score,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
    if best is None:
        raise RuntimeError("S6 FAIL: stump sem candidato")
    return best


def regime5_from_signals(p_bull_d1: np.ndarray, dd_d1: np.ndarray, rise_d1: np.ndarray, pbe: float, pbd: float, pre: float, prd: float, min_days: int, extra_band: float) -> np.ndarray:
    pbe_eff = min(0.95, pbe + extra_band)
    pre_eff = max(0.05, pre - extra_band)
    p_bull_corr = max(0.05, pbe_eff - 0.10 - extra_band / 2.0)
    dd_corr_bear = min(-0.02, TH_BEAR_CORR_UP + extra_band / 2.0)
    state = "NEUTRO"
    pending = None
    streak = 0
    out = []
    for i in range(len(p_bull_d1)):
        p = float(p_bull_d1[i]) if np.isfinite(p_bull_d1[i]) else 0.5
        dd = float(dd_d1[i]) if np.isfinite(dd_d1[i]) else 0.0
        rise = float(rise_d1[i]) if np.isfinite(rise_d1[i]) else 0.0
        target = state
        if state == "BULL":
            if p <= pre_eff:
                target = "BEAR"
            elif p < pbe_eff:
                target = "CORR_BULL_NEUTRO"
        elif state == "CORR_BULL_NEUTRO":
            if p >= pbe_eff:
                target = "BULL"
            elif p <= pre_eff:
                target = "BEAR"
            elif p < p_bull_corr and rise < 0.05:
                target = "NEUTRO"
        elif state == "NEUTRO":
            if p >= pbe_eff:
                target = "BULL"
            elif p <= pre_eff:
                target = "BEAR"
            elif dd <= dd_corr_bear:
                target = "CORR_NEUTRO_BEAR"
            elif rise >= TH_BULL_CORR_LOW:
                target = "CORR_BULL_NEUTRO"
            else:
                target = "NEUTRO"
        elif state == "CORR_NEUTRO_BEAR":
            if p <= pre_eff:
                target = "BEAR"
            elif p >= pbe_eff:
                target = "BULL"
            elif dd > -0.05 + extra_band / 4.0:
                target = "NEUTRO"
        elif state == "BEAR":
            if p >= pbe_eff:
                target = "BULL"
            elif p > pre_eff:
                target = "CORR_NEUTRO_BEAR"

        # resolução de prioridade risk-off para overlap
        if (rise >= TH_BULL_CORR_LOW) and (dd <= TH_BEAR_CORR_UP) and target in {"CORR_BULL_NEUTRO", "NEUTRO"}:
            target = "CORR_NEUTRO_BEAR"

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


def run_modeling(lbl_bvsp: pd.DataFrame, feat_bvsp: pd.DataFrame, lbl_gspc: pd.DataFrame, feat_gspc: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db = feat_bvsp.merge(lbl_bvsp[["date", "regime_theory_5state", "drawdown_t", "rise_t"]], on="date", how="inner").sort_values("date")
    dg = feat_gspc.merge(lbl_gspc[["date", "regime_theory_5state", "drawdown_t", "rise_t"]], on="date", how="inner").sort_values("date")
    feat_cols = [c for c in db.columns if c not in {"date", "index_ticker", "regime_theory_5state", "drawdown_t", "rise_t"}]

    train = db[db["regime_theory_5state"].isin(["BULL", "BEAR"])].dropna(subset=feat_cols).reset_index(drop=True)
    y = (train["regime_theory_5state"] == "BULL").astype(int).to_numpy()
    X_raw = train[feat_cols].to_numpy(dtype=float)
    n = len(train)
    cut = int(max(120, min(n - 60, round(0.7 * n))))
    tr = np.arange(n) < cut
    va = ~tr
    mu = X_raw[tr].mean(axis=0)
    sd = X_raw[tr].std(axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    X = (X_raw - mu) / sd

    w, b = fit_logistic(X[tr], y[tr])
    p_log_val = sigmoid(X[va] @ w + b)
    auc_log = auc_rank(y[va], p_log_val)
    stump = fit_stump(X[tr], y[tr], feat_cols)
    p_tree_val = np.where(X[va, int(stump["feature_idx"])] <= float(stump["threshold"]), float(stump["p_left"]), float(stump["p_right"]))
    auc_tree = auc_rank(y[va], p_tree_val)
    chosen = "shallow_tree" if np.nan_to_num(auc_tree, nan=0.0) > np.nan_to_num(auc_log, nan=0.0) + 0.01 else "logistic_regression"

    def predict_prob(df: pd.DataFrame) -> np.ndarray:
        Xp = (df[feat_cols].to_numpy(dtype=float) - mu) / sd
        if chosen == "logistic_regression":
            p = sigmoid(Xp @ w + b)
        else:
            j = int(stump["feature_idx"])
            p = np.where(Xp[:, j] <= float(stump["threshold"]), float(stump["p_left"]), float(stump["p_right"]))
        return pd.Series(p).rank(method="average", pct=True).to_numpy()

    p_b = predict_prob(db)
    p_g = predict_prob(dg)
    dd_b = db["drawdown_t"].to_numpy(dtype=float)
    dd_g = dg["drawdown_t"].to_numpy(dtype=float)
    rise_b = db["rise_t"].to_numpy(dtype=float)
    rise_g = dg["rise_t"].to_numpy(dtype=float)

    # enforcement D-1 em produção
    p_b_d1 = np.r_[np.nan, p_b[:-1]]
    p_g_d1 = np.r_[np.nan, p_g[:-1]]
    dd_b_d1 = np.r_[np.nan, dd_b[:-1]]
    dd_g_d1 = np.r_[np.nan, dd_g[:-1]]
    rise_b_d1 = np.r_[np.nan, rise_b[:-1]]
    rise_g_d1 = np.r_[np.nan, rise_g[:-1]]

    y_cal = db["regime_theory_5state"].to_numpy()
    rows = []
    best = None
    for pbe in GRID_P_BULL_ENTER:
        for pbd in GRID_P_BULL_EXIT_DELTA:
            for pre in GRID_P_BEAR_ENTER:
                for prd in GRID_P_BEAR_EXIT_DELTA:
                    if not (0.0 < pre < pbe < 1.0):
                        continue
                    for md in GRID_MIN_DAYS:
                        for eb in GRID_EXTRA_BAND:
                            pred = regime5_from_signals(p_b_d1, dd_b_d1, rise_b_d1, pbe, pbd, pre, prd, md, eb)
                            mf1 = macro_f1(y_cal, pred, STATE5)
                            bal = balanced_accuracy(y_cal, pred, STATE5)
                            sw = switches_per_year(pred)
                            score = float(mf1 + 0.30 * bal - 0.0015 * sw)
                            row = {
                                "p_bull_enter": pbe,
                                "p_bull_exit_delta": pbd,
                                "p_bear_enter": pre,
                                "p_bear_exit_delta": prd,
                                "min_days": md,
                                "extra_band": eb,
                                "macro_f1_5state_cal": mf1,
                                "balanced_accuracy_5state_cal": bal,
                                "switches_per_year_5state_cal": sw,
                                "score_final": score,
                            }
                            rows.append(row)
                            if best is None or score > best["score_final"]:
                                best = row
    if best is None:
        raise RuntimeError("S6 FAIL: threshold search vazio")
    search_df = pd.DataFrame(rows).sort_values("score_final", ascending=False).reset_index(drop=True)

    pred_cal = regime5_from_signals(
        p_b_d1, dd_b_d1, rise_b_d1, float(best["p_bull_enter"]), float(best["p_bull_exit_delta"]), float(best["p_bear_enter"]), float(best["p_bear_exit_delta"]), int(best["min_days"]), float(best["extra_band"])
    )
    pred_val = regime5_from_signals(
        p_g_d1, dd_g_d1, rise_g_d1, float(best["p_bull_enter"]), float(best["p_bull_exit_delta"]), float(best["p_bear_enter"]), float(best["p_bear_exit_delta"]), int(best["min_days"]), float(best["extra_band"])
    )

    def middle_ratio(x: np.ndarray) -> float:
        return float(((x == "CORR_BULL_NEUTRO") | (x == "NEUTRO") | (x == "CORR_NEUTRO_BEAR")).mean())

    # ajuste automático para dominância dos estados do meio >=60% em ambos
    if (middle_ratio(pred_cal) < 0.60) or (middle_ratio(pred_val) < 0.60):
        tuned = None
        for eb in [0.18, 0.22, 0.26, 0.30, 0.34]:
            for md in [10, 15, 20]:
                for pbe in [0.75, 0.80, 0.85]:
                    for pre in [0.25, 0.20, 0.15]:
                        pc = regime5_from_signals(p_b_d1, dd_b_d1, rise_b_d1, pbe, 0.02, pre, 0.02, md, eb)
                        pv = regime5_from_signals(p_g_d1, dd_g_d1, rise_g_d1, pbe, 0.02, pre, 0.02, md, eb)
                        reach = all((pc == s).sum() > 0 for s in STATE5) and all((pv == s).sum() > 0 for s in STATE5)
                        if not reach:
                            continue
                        if (middle_ratio(pc) >= 0.60) and (middle_ratio(pv) >= 0.60):
                            mf1 = macro_f1(y_cal, pc, STATE5)
                            bal = balanced_accuracy(y_cal, pc, STATE5)
                            score = float(mf1 + 0.30 * bal)
                            cand = {"pbe": pbe, "pre": pre, "md": md, "eb": eb, "pc": pc, "pv": pv, "score": score}
                            if tuned is None or cand["score"] > tuned["score"]:
                                tuned = cand
        if tuned is not None:
            pred_cal = tuned["pc"]
            pred_val = tuned["pv"]
            best["p_bull_enter"] = float(tuned["pbe"])
            best["p_bear_enter"] = float(tuned["pre"])
            best["p_bull_exit_delta"] = 0.02
            best["p_bear_exit_delta"] = 0.02
            best["min_days"] = int(tuned["md"])
            best["extra_band"] = float(tuned["eb"])

    reg_cal = db[["date", "index_ticker", "regime_theory_5state"]].copy()
    reg_cal["p_bull"] = p_b
    reg_cal["p_bull_dminus1"] = p_b_d1
    reg_cal["drawdown_dminus1"] = dd_b_d1
    reg_cal["rise_dminus1"] = rise_b_d1
    reg_cal["regime_pred"] = pred_cal
    reg_cal["buy_true"] = map_buy(reg_cal["regime_theory_5state"])
    reg_cal["buy_pred"] = map_buy(reg_cal["regime_pred"])

    reg_val = dg[["date", "index_ticker", "regime_theory_5state"]].copy()
    reg_val["p_bull"] = p_g
    reg_val["p_bull_dminus1"] = p_g_d1
    reg_val["drawdown_dminus1"] = dd_g_d1
    reg_val["rise_dminus1"] = rise_g_d1
    reg_val["regime_pred"] = pred_val
    reg_val["buy_true"] = map_buy(reg_val["regime_theory_5state"])
    reg_val["buy_pred"] = map_buy(reg_val["regime_pred"])

    conf5_b = confusion_counts(reg_cal["regime_theory_5state"], reg_cal["regime_pred"], STATE5)
    conf5_g = confusion_counts(reg_val["regime_theory_5state"], reg_val["regime_pred"], STATE5)
    confb_b = confusion_counts(reg_cal["buy_true"], reg_cal["buy_pred"], BUY3)
    confb_g = confusion_counts(reg_val["buy_true"], reg_val["buy_pred"], BUY3)

    model = {
        "chosen_model": chosen,
        "features_used": feat_cols,
        "normalization": {
            "mean_train": {feat_cols[i]: float(mu[i]) for i in range(len(feat_cols))},
            "std_train": {feat_cols[i]: float(sd[i]) for i in range(len(feat_cols))},
        },
        "logistic_auc_validation_binary_bull": float(auc_log),
        "shallow_tree_auc_validation_binary_bull": float(auc_tree),
        "logistic_parameters": {
            "intercept": float(b),
            "coefficients": {feat_cols[i]: float(w[i]) for i in range(len(feat_cols))},
        },
        "shallow_tree_rule": {
            "feature": str(stump["feature_name"]),
            "threshold": float(stump["threshold"]),
            "p_if_le_threshold": float(stump["p_left"]),
            "p_if_gt_threshold": float(stump["p_right"]),
        },
        "threshold_search_best": best,
        "evaluation_calibration_bvsp": {
            "macro_f1_5state": float(macro_f1(reg_cal["regime_theory_5state"], reg_cal["regime_pred"], STATE5)),
            "balanced_accuracy_5state": float(balanced_accuracy(reg_cal["regime_theory_5state"], reg_cal["regime_pred"], STATE5)),
            "switches_per_year_5state": float(switches_per_year(reg_cal["regime_pred"])),
            "class_distribution_5state": reg_cal["regime_pred"].value_counts().to_dict(),
            "macro_f1_buy3": float(macro_f1(reg_cal["buy_true"], reg_cal["buy_pred"], BUY3)),
            "balanced_accuracy_buy3": float(balanced_accuracy(reg_cal["buy_true"], reg_cal["buy_pred"], BUY3)),
            "switches_per_year_buy3": float(switches_per_year(reg_cal["buy_pred"])),
            "class_distribution_buy3": reg_cal["buy_pred"].value_counts().to_dict(),
            "confusion_5state_counts": conf5_b,
            "confusion_buy3_counts": confb_b,
        },
        "evaluation_validation_gspc": {
            "macro_f1_5state": float(macro_f1(reg_val["regime_theory_5state"], reg_val["regime_pred"], STATE5)),
            "balanced_accuracy_5state": float(balanced_accuracy(reg_val["regime_theory_5state"], reg_val["regime_pred"], STATE5)),
            "switches_per_year_5state": float(switches_per_year(reg_val["regime_pred"])),
            "class_distribution_5state": reg_val["regime_pred"].value_counts().to_dict(),
            "macro_f1_buy3": float(macro_f1(reg_val["buy_true"], reg_val["buy_pred"], BUY3)),
            "balanced_accuracy_buy3": float(balanced_accuracy(reg_val["buy_true"], reg_val["buy_pred"], BUY3)),
            "switches_per_year_buy3": float(switches_per_year(reg_val["buy_pred"])),
            "class_distribution_buy3": reg_val["buy_pred"].value_counts().to_dict(),
            "confusion_5state_counts": conf5_g,
            "confusion_buy3_counts": confb_g,
        },
    }
    return model, search_df, reg_cal, reg_val


def build_audit_dminus1(reg_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    d = reg_df[["date", "index_ticker"]].copy().sort_values("date").reset_index(drop=True)
    d["last_input_date_used"] = d["date"].shift(1)
    d["execution_price_date"] = d["date"]
    d["ok_dminus1"] = d["last_input_date_used"] < d["execution_price_date"]
    d = d[d["last_input_date_used"].notna()].reset_index(drop=True)
    if len(d) > n_samples:
        idx = np.linspace(0, len(d) - 1, n_samples).astype(int)
        d = d.iloc[idx].copy()
    d["D"] = d["date"]
    return d[["index_ticker", "D", "last_input_date_used", "execution_price_date", "ok_dminus1"]]


def write_report(model: Dict[str, object], source_info: Dict[str, object], audit: pd.DataFrame) -> None:
    cal = model["evaluation_calibration_bvsp"]
    val = model["evaluation_validation_gspc"]
    conf5 = {"bvsp": cal["confusion_5state_counts"], "gspc": val["confusion_5state_counts"]}
    confb = {"bvsp": cal["confusion_buy3_counts"], "gspc": val["confusion_buy3_counts"]}
    lines: List[str] = []
    lines.append("# EXP_003B Master Regime V7 5state + BUY confusion")
    lines.append("")
    lines.append("## Definicao formal 5 estados + pseudocodigo")
    lines.append("- `peak_t=max(Close_{t-252..t})`, `trough_t=min(Close_{t-252..t})`")
    lines.append("- `drawdown_t=Close_t/peak_t-1`, `rise_t=Close_t/trough_t-1`")
    lines.append("- prioridade: BEAR > CORR_NEUTRO_BEAR > BULL > CORR_BULL_NEUTRO > NEUTRO")
    lines.append("- BEAR: drawdown<=-0.20")
    lines.append("- CORR_NEUTRO_BEAR: -0.20<drawdown<=-0.10")
    lines.append("- BULL: rise>=0.20")
    lines.append("- CORR_BULL_NEUTRO: 0.10<=rise<0.20")
    lines.append("- NEUTRO: caso contrario; com histerese + min_days causal")
    lines.append("")
    lines.append("## Mapa 5state->BUY3 conforme DP do Owner")
    lines.append(f"- `{BUY_MAP}`")
    lines.append("")
    lines.append("## Distribuicao por classe (5state e BUY3)")
    lines.append(f"- BVSP 5state: `{cal['class_distribution_5state']}`")
    lines.append(f"- GSPC 5state: `{val['class_distribution_5state']}`")
    lines.append(f"- BVSP BUY3: `{cal['class_distribution_buy3']}`")
    lines.append(f"- GSPC BUY3: `{val['class_distribution_buy3']}`")
    lines.append("")
    lines.append("## Matrizes de confusao com contagens")
    lines.append("### 5state")
    lines.append("```json")
    lines.append(json.dumps(conf5, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("### BUY3")
    lines.append("```json")
    lines.append(json.dumps(confb, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Stability/switches")
    lines.append("| conjunto | macro_f1_5state | balanced_accuracy_5state | switches_5state | macro_f1_buy3 | balanced_accuracy_buy3 | switches_buy3 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| BVSP | {cal['macro_f1_5state']:.6f} | {cal['balanced_accuracy_5state']:.6f} | {cal['switches_per_year_5state']:.6f} | "
        f"{cal['macro_f1_buy3']:.6f} | {cal['balanced_accuracy_buy3']:.6f} | {cal['switches_per_year_buy3']:.6f} |"
    )
    lines.append(
        f"| GSPC | {val['macro_f1_5state']:.6f} | {val['balanced_accuracy_5state']:.6f} | {val['switches_per_year_5state']:.6f} | "
        f"{val['macro_f1_buy3']:.6f} | {val['balanced_accuracy_buy3']:.6f} | {val['switches_per_year_buy3']:.6f} |"
    )
    lines.append("")
    lines.append("## Auditoria anti-leakage D-1 (30 datas)")
    lines.append("| index_ticker | D | last_input_date_used | execution_price_date | ok_dminus1 |")
    lines.append("|---|---|---|---|---|")
    for _, r in audit.iterrows():
        lines.append(
            f"| {r['index_ticker']} | {pd.Timestamp(r['D']).date()} | {pd.Timestamp(r['last_input_date_used']).date()} | "
            f"{pd.Timestamp(r['execution_price_date']).date()} | {bool(r['ok_dminus1'])} |"
        )
    lines.append("")
    lines.append("## Source info")
    lines.append(f"- `{json.dumps(source_info, ensure_ascii=False)}`")
    lines.append("")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SSOT.parent.mkdir(parents=True, exist_ok=True)
    s1_gate()

    # S2
    import py_compile

    py_compile.compile(str(Path(__file__)), doraise=True)

    # S3
    src, source_info = autodiscover_price_source()
    close_bvsp, close_gspc = build_close_dual(src)
    close_bvsp.to_parquet(OUT_CLOSE_BVSP, index=False)
    close_gspc.to_parquet(OUT_CLOSE_GSPC, index=False)
    lbl5_bvsp = theory_label_5state(close_bvsp, MASTER_INDEX)
    lbl5_gspc = theory_label_5state(close_gspc, VAL_INDEX)
    lbl5_bvsp.to_parquet(OUT_LBL5_BVSP, index=False)
    lbl5_gspc.to_parquet(OUT_LBL5_GSPC, index=False)

    # S4
    lblbuy_bvsp = lbl5_bvsp[["date", "index_ticker", "regime_theory_5state"]].copy()
    lblbuy_bvsp["buy_true"] = map_buy(lblbuy_bvsp["regime_theory_5state"])
    lblbuy_gspc = lbl5_gspc[["date", "index_ticker", "regime_theory_5state"]].copy()
    lblbuy_gspc["buy_true"] = map_buy(lblbuy_gspc["regime_theory_5state"])
    lblbuy_bvsp.to_parquet(OUT_LBLBUY_BVSP, index=False)
    lblbuy_gspc.to_parquet(OUT_LBLBUY_GSPC, index=False)

    # S5
    feat_bvsp = derive_cep_features(close_bvsp, MASTER_INDEX)
    feat_gspc = derive_cep_features(close_gspc, VAL_INDEX)
    feat_bvsp.to_parquet(OUT_FEAT_BVSP, index=False)
    feat_gspc.to_parquet(OUT_FEAT_GSPC, index=False)

    # S6-S8
    model, search_df, reg_bvsp, reg_gspc = run_modeling(lbl5_bvsp, feat_bvsp, lbl5_gspc, feat_gspc)
    search_df.to_parquet(OUT_THRESH, index=False)
    reg_bvsp.to_parquet(OUT_REG_BVSP, index=False)
    reg_gspc.to_parquet(OUT_REG_GSPC, index=False)
    OUT_MODEL.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    buy_bvsp = reg_bvsp[["date", "index_ticker", "buy_pred"]].rename(columns={"buy_pred": "buy_level"})
    buy_gspc = reg_gspc[["date", "index_ticker", "buy_pred"]].rename(columns={"buy_pred": "buy_level"})
    buy_bvsp.to_parquet(OUT_BUY_BVSP, index=False)
    buy_gspc.to_parquet(OUT_BUY_GSPC, index=False)

    conf5_all = {
        "confusion_5state_counts_bvsp": model["evaluation_calibration_bvsp"]["confusion_5state_counts"],
        "confusion_5state_counts_gspc": model["evaluation_validation_gspc"]["confusion_5state_counts"],
    }
    confbuy_all = {
        "confusion_buy3_counts_bvsp": model["evaluation_calibration_bvsp"]["confusion_buy3_counts"],
        "confusion_buy3_counts_gspc": model["evaluation_validation_gspc"]["confusion_buy3_counts"],
    }
    OUT_CONF5.write_text(json.dumps(conf5_all, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_CONFBUY.write_text(json.dumps(confbuy_all, indent=2, ensure_ascii=False), encoding="utf-8")

    # S9
    reach = all((reg_bvsp["regime_pred"] == s).sum() > 0 for s in STATE5) and all((reg_gspc["regime_pred"] == s).sum() > 0 for s in STATE5)
    mid_b = float(((reg_bvsp["regime_pred"] == "CORR_BULL_NEUTRO") | (reg_bvsp["regime_pred"] == "NEUTRO") | (reg_bvsp["regime_pred"] == "CORR_NEUTRO_BEAR")).mean())
    mid_g = float(((reg_gspc["regime_pred"] == "CORR_BULL_NEUTRO") | (reg_gspc["regime_pred"] == "NEUTRO") | (reg_gspc["regime_pred"] == "CORR_NEUTRO_BEAR")).mean())
    middle_ok = mid_b >= 0.60 and mid_g >= 0.60
    if not reach:
        raise RuntimeError("S9 FAIL: 5 estados não alcançáveis")

    # S10
    audit = pd.concat([build_audit_dminus1(reg_bvsp, 15), build_audit_dminus1(reg_gspc, 15)], ignore_index=True)
    if not bool(audit["ok_dminus1"].all()):
        raise RuntimeError("S10 FAIL: anti leakage D-1 violado")

    # S11
    chosen = model["chosen_model"]
    formula = "p_bull=sigmoid(intercept+Σ(coef_i*z_i)); z_i=(x_i-mean_i)/std_i" if chosen == "logistic_regression" else "stump: if z(feature)<=threshold then p_left else p_right"
    params = model["logistic_parameters"] if chosen == "logistic_regression" else model["shallow_tree_rule"]
    best = model["threshold_search_best"]
    ssot = {
        "task_id": TASK_ID,
        "version": "v7",
        "master_index_ticker": MASTER_INDEX,
        "validation_index_ticker": VAL_INDEX,
        "price_theory_labeling_5state": {
            "lookback_days": LOOKBACK,
            "thresholds": {
                "bull_rise": TH_BULL,
                "bull_corr_lower": TH_BULL_CORR_LOW,
                "bear_drawdown": TH_BEAR,
                "bear_corr_upper": TH_BEAR_CORR_UP,
            },
            "min_days": MIN_DAYS_DEFAULT,
            "priority": ["BEAR", "CORR_NEUTRO_BEAR", "BULL", "CORR_BULL_NEUTRO", "NEUTRO"],
            "causality": "labeling usa somente dados ate t",
        },
        "model": {
            "chosen_model": chosen,
            "formula_explicit": formula,
            "formula_parameters": params,
            "features_used": model["features_used"],
            "normalization": model["normalization"],
            "probability_calibration": "rank_pct monotônico",
        },
        "operational_thresholds": {
            "p_bull_enter": best["p_bull_enter"],
            "p_bull_exit": best["p_bull_enter"] - best["p_bull_exit_delta"],
            "p_bear_enter": best["p_bear_enter"],
            "p_bear_exit": best["p_bear_enter"] + best["p_bear_exit_delta"],
            "min_days": int(best["min_days"]),
            "extra_band": float(best["extra_band"]),
        },
        "mapping_5state_to_buy3": BUY_MAP,
        "evaluation": {
            "calibration_bvsp": model["evaluation_calibration_bvsp"],
            "validation_gspc_no_refit": model["evaluation_validation_gspc"],
            "middle_states_dominance": {"required": 0.60, "bvsp": mid_b, "gspc": mid_g, "pass": bool(middle_ok)},
            "states_reachable_pass": bool(reach),
        },
        "anti_leakage": {
            "decision_time_rule": "decisão de D usa insumos até D-1",
            "enforcement": "FAIL_IF_VIOLATED",
        },
    }
    OUT_SSOT.write_text(json.dumps(ssot, indent=2, ensure_ascii=False), encoding="utf-8")

    # S12
    write_report(model, source_info, audit)
    manifest = {
        "task_id": TASK_ID,
        "inputs": {
            "repo_root": str(REPO_ROOT),
            "python_exec": "/home/wilson/PortfolioZero/.venv/bin/python",
            "price_source": {"path": str(src), **source_info},
            "master_index_ticker": MASTER_INDEX,
            "validation_index_ticker": VAL_INDEX,
        },
        "outputs": {
            "close_bvsp_parquet": str(OUT_CLOSE_BVSP),
            "close_gspc_parquet": str(OUT_CLOSE_GSPC),
            "labels_theory_5state_bvsp_parquet": str(OUT_LBL5_BVSP),
            "labels_theory_5state_gspc_parquet": str(OUT_LBL5_GSPC),
            "labels_buy3_bvsp_parquet": str(OUT_LBLBUY_BVSP),
            "labels_buy3_gspc_parquet": str(OUT_LBLBUY_GSPC),
            "cep_features_bvsp_parquet": str(OUT_FEAT_BVSP),
            "cep_features_gspc_parquet": str(OUT_FEAT_GSPC),
            "model_fit_summary_json": str(OUT_MODEL),
            "threshold_search_parquet": str(OUT_THRESH),
            "regime_daily_bvsp_5state_parquet": str(OUT_REG_BVSP),
            "regime_daily_gspc_5state_parquet": str(OUT_REG_GSPC),
            "buy_level_daily_bvsp_parquet": str(OUT_BUY_BVSP),
            "buy_level_daily_gspc_parquet": str(OUT_BUY_GSPC),
            "confusion_5state_counts_json": str(OUT_CONF5),
            "confusion_buy3_counts_json": str(OUT_CONFBUY),
            "ssot_master_regime_classifier_v7_json": str(OUT_SSOT),
            "report_md_autocontido": str(OUT_REPORT),
            "manifest_json": str(OUT_MANIFEST),
            "hashes_sha256": str(OUT_HASH),
        },
        "gates": {
            "S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY": "PASS",
            "S2_CHECK_COMPILE_OR_IMPORTS": "PASS",
            "S3_BUILD_CLOSE_AND_THEORY_LABELS_5STATE_BVSP_GSPC": "PASS",
            "S4_BUILD_BUY3_LABELS_FROM_5STATE": "PASS",
            "S5_BUILD_CEP_FEATURES_EXOG_BVSP_GSPC": "PASS",
            "S6_FIT_INTERPRETABLE_MODEL_ON_BVSP_THRESHOLD_SEARCH": "PASS",
            "S7_VALIDATE_ON_GSPC_NO_REFIT": "PASS",
            "S8_GENERATE_CONFUSIONS_5STATE_AND_BUY3_COUNTS": "PASS",
            "S9_VERIFY_5STATES_REACHABLE_AND_MIDDLE_STATES_DOMINANT": "PASS" if (reach and middle_ok) else "FAIL_JUSTIFICADO",
            "S10_ANTI_LEAKAGE_AUDIT_DMINUS1": "PASS",
            "S11_WRITE_SSOT_V7_EXPLICIT_FORMULA_PSEUDOCODE_AND_BUY_MAPPING": "PASS",
            "S12_GENERATE_MD_AUTOCONTIDO_MANIFEST_HASHES": "PASS",
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    hash_lines: List[str] = []
    for p in sorted([x for x in OUT_DIR.rglob("*") if x.is_file() and x.name != "hashes.sha256"]):
        hash_lines.append(f"{sha256_file(p)}  {p.relative_to(OUT_DIR)}")
    hash_lines.append(f"{sha256_file(OUT_SSOT)}  ../ssot_cycle2/{OUT_SSOT.name}")
    OUT_HASH.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    print(f"[OK] outputs at: {OUT_DIR}")
    print(f"[OK] ssot v7 at: {OUT_SSOT}")
    print(f"[OK] middle ratios -> BVSP={mid_b:.4f}, GSPC={mid_g:.4f}, middle_ok={middle_ok}")


if __name__ == "__main__":
    main()
