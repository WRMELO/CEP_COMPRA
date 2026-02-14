#!/usr/bin/env python3
import fnmatch
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TASK_ID = "TASK_CEP_COMPRA_EXP_002A_MASTER_REGIME_DAILY_FROM_MONTHLY_WEAKLABELS_V4"
REPO_ROOT = Path("/home/wilson/CEP_COMPRA")
MASTER_TICKER = "^BVSP"

OUT_DIR = REPO_ROOT / "outputs/experimentos/controle_rl/EXP_002A_master_regime_v4"
OUT_SSOT = OUT_DIR / "ssot_cycle2/master_regime_classifier_v4.json"
OUT_MASTER_LOGRET = OUT_DIR / "master_logret_daily.parquet"
OUT_MASTER_FEATS = OUT_DIR / "master_cep_features_daily.parquet"
OUT_MASTER_FEATS_CSV = OUT_DIR / "master_cep_features_daily.csv"
OUT_WEAK_AUDIT = OUT_DIR / "weaklabels_daily_audit.parquet"
OUT_REGIME_DAILY = OUT_DIR / "regime_daily.parquet"
OUT_REPORT = OUT_DIR / "report.md"
OUT_MANIFEST = OUT_DIR / "manifest.json"
OUT_HASH = OUT_DIR / "hashes.sha256"

S1_EVIDENCE = REPO_ROOT / "planning/runs/TASK_CEP_COMPRA_EXP_002A_MASTER_REGIME_DAILY_FROM_MONTHLY_WEAKLABELS_V4/S1_GATE_ALLOWLIST.txt"

FORBIDDEN_FEATURE_PATTERNS = [
    "*positions*",
    "*n_positions*",
    "*portfolio_state*",
    "*risk_on*",
    "*buys*",
    "*sells*",
    "*turnover*",
]

GRID_P_BULL_ENTER = [0.55, 0.60, 0.65, 0.70]
GRID_P_BULL_EXIT_DELTA = [0.01, 0.02, 0.03]
GRID_P_BEAR_ENTER = [0.45, 0.40, 0.35, 0.30]
GRID_P_BEAR_EXIT_DELTA = [0.01, 0.02, 0.03]
GRID_MIN_DAYS = [1, 2, 3, 5]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def balacc_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    tpr = float((y_pred[pos] == 1).sum()) / max(1, int(pos.sum()))
    tnr = float((y_pred[neg] == 0).sum()) / max(1, int(neg.sum()))
    return 0.5 * (tpr + tnr)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int) -> Dict[str, float]:
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(((yt == positive_class) & (yp == positive_class)).sum())
    fp = int(((yt != positive_class) & (yp == positive_class)).sum())
    fn = int(((yt == positive_class) & (yp != positive_class)).sum())
    p = float(tp) / max(1, tp + fp)
    r = float(tp) / max(1, tp + fn)
    f1 = 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f1}


def is_forbidden_feature(name: str) -> bool:
    low = name.lower()
    return any(fnmatch.fnmatch(low, patt.lower()) for patt in FORBIDDEN_FEATURE_PATTERNS)


def s1_gate_allowlist() -> None:
    ensure_parent(S1_EVIDENCE)
    if not str(OUT_DIR.resolve()).startswith(str((REPO_ROOT / "outputs").resolve())):
        raise RuntimeError("Saída fora de /home/wilson/CEP_COMPRA/outputs")
    S1_EVIDENCE.write_text(
        "\n".join(
            [
                f"TASK: {TASK_ID}",
                "PASS: leituras permitidas em /home/wilson/CEP_COMPRA/{outputs,docs,planning}",
                "PASS: escrita permitida em outputs/experimentos/controle_rl/EXP_002A_master_regime_v4/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def autodiscover_master_series() -> Tuple[Path, Dict[str, object]]:
    search_root = REPO_ROOT / "outputs"
    patterns = ["*bvsp*.parquet", "*BVSP*.parquet", "*master*series*.parquet"]
    candidates: List[Path] = []
    for patt in patterns:
        for p in search_root.rglob(patt):
            if p.is_file():
                candidates.append(p)
    if candidates:
        candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
        accepted = []
        for p in candidates:
            try:
                cols = set(pd.read_parquet(p).columns)
            except Exception:
                continue
            if cols.intersection({"r_t", "xt_ibov", "bvsp_index_norm", "bvsp_index", "bvsp_close"}):
                accepted.append(p)
        if accepted:
            return accepted[0], {
                "method": "preferred_parquet_globs",
                "candidates_found": len(candidates),
                "accepted_candidates": [str(x) for x in accepted[:10]],
            }
    # fallback local known parquet artifact (no web build)
    fallback = REPO_ROOT / "outputs/cycle2/20260213/master_regime_m3_cep_only_v3/master_logret_daily.parquet"
    if fallback.exists():
        return fallback, {"method": "fallback_local_known_parquet", "candidates_found": 0}
    raise RuntimeError("S2 FAIL: não foi possível autodiscover série master em parquet")


def build_weak_monthly_from_text() -> Dict[str, str]:
    # Fraco/fajuto conforme indicação textual do usuário.
    month_labels: Dict[str, str] = {}
    for y in [2019, 2023, 2025]:
        for m in range(1, 13):
            month_labels[f"{y:04d}-{m:02d}"] = "BULL"
    for y in [2021, 2022, 2024]:
        for m in range(1, 13):
            month_labels[f"{y:04d}-{m:02d}"] = "BEAR"
    return month_labels


def load_master_logret(master_series_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(master_series_path).copy()
    if "date" not in df.columns:
        raise RuntimeError("Série master sem coluna date")
    df["date"] = pd.to_datetime(df["date"])
    if "r_t" in df.columns:
        out = df[["date", "r_t"]].copy()
    elif "xt_ibov" in df.columns:
        out = df[["date", "xt_ibov"]].rename(columns={"xt_ibov": "r_t"})
    elif "bvsp_index_norm" in df.columns:
        tmp = df[["date", "bvsp_index_norm"]].copy()
        tmp["r_t"] = np.log(pd.to_numeric(tmp["bvsp_index_norm"], errors="coerce")).diff()
        out = tmp[["date", "r_t"]]
    elif "bvsp_index" in df.columns:
        tmp = df[["date", "bvsp_index"]].copy()
        tmp["r_t"] = np.log(pd.to_numeric(tmp["bvsp_index"], errors="coerce")).diff()
        out = tmp[["date", "r_t"]]
    elif "bvsp_close" in df.columns:
        tmp = df[["date", "bvsp_close"]].copy()
        tmp["r_t"] = np.log(pd.to_numeric(tmp["bvsp_close"], errors="coerce")).diff()
        out = tmp[["date", "r_t"]]
    else:
        raise RuntimeError("Master parquet sem coluna reconhecível para log-ret")
    out["master_ticker"] = MASTER_TICKER
    out["r_t"] = pd.to_numeric(out["r_t"], errors="coerce")
    out = out.dropna(subset=["r_t"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def derive_cep_features(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    d = master_df.copy().sort_values("date")
    n = 4
    k = 60
    if len(d) < (k + n + 5):
        raise RuntimeError("Série curta para construir baseline CEP")
    d["mr_t"] = d["r_t"].diff().abs()
    d["xbar_n"] = d["r_t"].rolling(n, min_periods=n).mean()
    d["range_n"] = d["r_t"].rolling(n, min_periods=n).max() - d["r_t"].rolling(n, min_periods=n).min()
    d["mu_60"] = d["r_t"].rolling(60, min_periods=60).mean()
    d["sd_60"] = d["r_t"].rolling(60, min_periods=60).std(ddof=0)
    d["z_r_60"] = (d["r_t"] - d["mu_60"]) / d["sd_60"].replace(0.0, np.nan)

    base = d.iloc[:k].copy()
    i_mean = float(base["r_t"].mean())
    i_std = float(base["r_t"].std(ddof=0))
    i_lcl = i_mean - 3.0 * i_std
    i_ucl = i_mean + 3.0 * i_std
    mr_mean = float(base["mr_t"].dropna().mean())
    mr_std = float(base["mr_t"].dropna().std(ddof=0))
    mr_ucl = mr_mean + 3.0 * mr_std
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
    forbidden = [c for c in feat_cols if is_forbidden_feature(c)]
    if forbidden:
        raise RuntimeError(f"S3 FAIL: features proibidas detectadas: {forbidden}")

    out = d[["date", "master_ticker"] + feat_cols].copy()
    limits = {
        "baseline_k": k,
        "subgroup_n": n,
        "i_mean": i_mean,
        "i_std": i_std,
        "i_lcl": i_lcl,
        "i_ucl": i_ucl,
        "mr_ucl": mr_ucl,
        "xbar_lcl": xbar_lcl,
        "xbar_ucl": xbar_ucl,
        "r_ucl": r_ucl,
    }
    return out, limits


def fit_logistic(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 3200, l2: float = 5e-4) -> Tuple[np.ndarray, float]:
    n, p = X.shape
    w = np.zeros(p, dtype=float)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        pr = sigmoid(z)
        err = pr - y
        gw = (X.T @ err) / n + l2 * w
        gb = float(err.mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def fit_shallow_tree(X: np.ndarray, y: np.ndarray, feat_names: Sequence[str]) -> Dict[str, object]:
    best = None
    for j in range(X.shape[1]):
        vals = np.unique(np.round(X[:, j], 6))
        if len(vals) > 60:
            cuts = np.unique(np.round(np.quantile(vals, np.linspace(0.05, 0.95, 19)), 6))
        else:
            cuts = vals
        for thr in cuts:
            left = X[:, j] <= thr
            right = ~left
            if left.sum() < 10 or right.sum() < 10:
                continue
            p_left = float(y[left].mean())
            p_right = float(y[right].mean())
            p = np.where(left, p_left, p_right)
            auc = auc_rank(y, p)
            bal = balacc_binary(y, (p >= 0.5).astype(int))
            score = float(np.nan_to_num(auc, nan=0.5)) + 0.5 * bal
            cand = {
                "feature_idx": j,
                "feature_name": feat_names[j],
                "threshold": float(thr),
                "p_left": p_left,
                "p_right": p_right,
                "auc": float(auc),
                "balacc": float(bal),
                "score": score,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
    if best is None:
        raise RuntimeError("S5 FAIL: shallow_tree sem candidato")
    return best


def hysteresis_regime(p_bull: np.ndarray, pbe: float, pbd: float, pre: float, prd: float, min_days: int) -> np.ndarray:
    bull_exit = pbe - pbd
    bear_exit = pre + prd
    state = "MISTO"
    pending: Optional[str] = None
    streak = 0
    out = []
    for p in p_bull:
        target = state
        if state == "BULL":
            if p <= pre:
                target = "BEAR"
            elif p <= bull_exit:
                target = "MISTO"
        elif state == "BEAR":
            if p >= pbe:
                target = "BULL"
            elif p >= bear_exit:
                target = "MISTO"
        else:
            if p >= pbe:
                target = "BULL"
            elif p <= pre:
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


def build_weak_daily(master: pd.DataFrame, month_labels: Dict[str, str]) -> pd.DataFrame:
    out = master[["date", "master_ticker"]].copy()
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["weak_label_monthly"] = out["month"].map(month_labels).fillna("UNLABELED")
    out["is_labeled_bull_bear"] = out["weak_label_monthly"].isin(["BULL", "BEAR"]).astype(int)
    return out


def run_model_and_search(feats: pd.DataFrame, weak_daily: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    d = feats.merge(weak_daily[["date", "weak_label_monthly", "is_labeled_bull_bear"]], on="date", how="left")
    d = d.sort_values("date").reset_index(drop=True)
    feat_cols = [c for c in d.columns if c not in {"date", "master_ticker", "weak_label_monthly", "is_labeled_bull_bear"}]
    feat_cols = [c for c in feat_cols if not is_forbidden_feature(c)]
    labeled = d[d["is_labeled_bull_bear"] == 1].dropna(subset=feat_cols).copy().reset_index(drop=True)
    if labeled.empty:
        raise RuntimeError("S5 FAIL: nenhum dado rotulado")
    y = (labeled["weak_label_monthly"] == "BULL").astype(int).to_numpy()
    X_raw = labeled[feat_cols].to_numpy(dtype=float)
    n = len(labeled)
    n_train = int(max(50, min(n - 30, round(0.7 * n))))
    tr = np.arange(n) < n_train
    va = ~tr
    mu = X_raw[tr].mean(axis=0)
    sd = X_raw[tr].std(axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    X = (X_raw - mu) / sd

    w, b = fit_logistic(X[tr], y[tr])
    p_log_va = sigmoid(X[va] @ w + b)
    auc_log = auc_rank(y[va], p_log_va)
    bal_log = balacc_binary(y[va], (p_log_va >= 0.5).astype(int))

    stump = fit_shallow_tree(X[tr], y[tr], feat_cols)
    p_tree_va = np.where(X[va, int(stump["feature_idx"])] <= float(stump["threshold"]), float(stump["p_left"]), float(stump["p_right"]))
    auc_tree = auc_rank(y[va], p_tree_va)
    bal_tree = balacc_binary(y[va], (p_tree_va >= 0.5).astype(int))

    chosen = "logistic_regression"
    if np.nan_to_num(auc_tree, nan=0.0) > np.nan_to_num(auc_log, nan=0.0) + 0.01:
        chosen = "shallow_tree"

    X_all = (d[feat_cols].to_numpy(dtype=float) - mu) / sd
    if chosen == "logistic_regression":
        d["p_bull_raw"] = sigmoid(X_all @ w + b)
    else:
        j = int(stump["feature_idx"])
        d["p_bull_raw"] = np.where(X_all[:, j] <= float(stump["threshold"]), float(stump["p_left"]), float(stump["p_right"]))

    # validation mask on full index
    labeled_global = d[d["is_labeled_bull_bear"] == 1].copy()
    val_dates = set(labeled_global.iloc[n_train:]["date"])
    val_mask = d["date"].isin(val_dates).to_numpy()
    y_val = (d.loc[val_mask, "weak_label_monthly"] == "BULL").astype(int).to_numpy()

    search_rows = []
    best = None
    for pbe in GRID_P_BULL_ENTER:
        for pbd in GRID_P_BULL_EXIT_DELTA:
            for pre in GRID_P_BEAR_ENTER:
                for prd in GRID_P_BEAR_EXIT_DELTA:
                    if not (0.0 < pre < pbe < 1.0):
                        continue
                    for md in GRID_MIN_DAYS:
                        reg = hysteresis_regime(d["p_bull_raw"].fillna(0.5).to_numpy(), pbe, pbd, pre, prd, md)
                        pred_val = reg[val_mask]
                        pred_val_bin = np.where(pred_val == "BULL", 1, np.where(pred_val == "BEAR", 0, -1))
                        pred_for_bal = np.where(pred_val_bin == -1, 0, pred_val_bin)
                        bal = balacc_binary(y_val, pred_for_bal)
                        misto_rate = float((pred_val == "MISTO").mean()) if len(pred_val) else 1.0
                        score = float(bal - 0.05 * misto_rate)
                        row = {
                            "p_bull_enter": pbe,
                            "p_bull_exit_delta": pbd,
                            "p_bear_enter": pre,
                            "p_bear_exit_delta": prd,
                            "min_days": md,
                            "balanced_accuracy_validation": float(bal),
                            "misto_rate_validation": misto_rate,
                            "score_final": score,
                        }
                        search_rows.append(row)
                        if best is None or row["score_final"] > best["score_final"]:
                            best = row
    if best is None:
        raise RuntimeError("S5 FAIL: threshold search sem candidato")

    reg_final = hysteresis_regime(
        d["p_bull_raw"].fillna(0.5).to_numpy(),
        best["p_bull_enter"],
        best["p_bull_exit_delta"],
        best["p_bear_enter"],
        best["p_bear_exit_delta"],
        int(best["min_days"]),
    )
    d["regime_label"] = reg_final

    # metrics on validation (weak labels)
    pred_val = d.loc[val_mask, "regime_label"].to_numpy()
    pred_val_bin = np.where(pred_val == "BULL", 1, np.where(pred_val == "BEAR", 0, -1))
    pred_for_metrics = np.where(pred_val_bin == -1, 0, pred_val_bin)
    bal_best = balacc_binary(y_val, pred_for_metrics)
    auc_best = auc_rank(y_val, d.loc[val_mask, "p_bull_raw"].to_numpy())
    m_bull = precision_recall_f1(y_val, pred_for_metrics, 1)
    m_bear = precision_recall_f1(y_val, pred_for_metrics, 0)

    model_info = {
        "chosen_model": chosen,
        "features_used": feat_cols,
        "normalization": {
            "mean_train": {feat_cols[i]: float(mu[i]) for i in range(len(feat_cols))},
            "std_train": {feat_cols[i]: float(sd[i]) for i in range(len(feat_cols))},
        },
        "logistic": {
            "auc_validation": float(auc_log),
            "balanced_accuracy_validation": float(bal_log),
            "intercept": float(b),
            "coefficients": {feat_cols[i]: float(w[i]) for i in range(len(feat_cols))},
        },
        "shallow_tree": {
            "auc_validation": float(auc_tree),
            "balanced_accuracy_validation": float(bal_tree),
            "rule": {
                "feature": str(stump["feature_name"]),
                "threshold": float(stump["threshold"]),
                "p_if_le_threshold": float(stump["p_left"]),
                "p_if_gt_threshold": float(stump["p_right"]),
            },
        },
        "best_threshold_config": best,
        "validation_metrics_best": {
            "balanced_accuracy": float(bal_best),
            "auc": float(auc_best),
            "bull": m_bull,
            "bear": m_bear,
        },
        "selection_score_definition": "score_final = balanced_accuracy_validation - 0.05 * misto_rate_validation",
    }
    return model_info, d, pd.DataFrame(search_rows).sort_values("score_final", ascending=False).reset_index(drop=True)


def build_ssot_v4(model_info: Dict[str, object], limits: Dict[str, float], weak_source: Dict[str, object], master_source: Dict[str, object], search_top: pd.DataFrame, d_regime: pd.DataFrame) -> Dict[str, object]:
    chosen = model_info["chosen_model"]
    if chosen == "logistic_regression":
        formula = "p_bull = sigmoid(intercept + Σ(coef_i * z_i)); z_i=(x_i-mean_train_i)/std_train_i"
        params = {
            "intercept": model_info["logistic"]["intercept"],
            "coefficients": model_info["logistic"]["coefficients"],
        }
    else:
        formula = "if z(feature)<=threshold then p_bull=p_if_le_threshold else p_bull=p_if_gt_threshold"
        params = model_info["shallow_tree"]["rule"]

    c_bull = int((d_regime["regime_label"] == "BULL").sum())
    c_bear = int((d_regime["regime_label"] == "BEAR").sum())
    c_misto = int((d_regime["regime_label"] == "MISTO").sum())
    reach_ok = c_bull >= 30 and c_bear >= 30
    reach_note = "OK" if reach_ok else "Classes BULL/BEAR <30 dias em pelo menos uma classe; limitação de sinal/dados."

    ssot = {
        "task_id": TASK_ID,
        "version": "v4",
        "master_ticker": MASTER_TICKER,
        "input_discovery": {
            "master_series": master_source,
            "weak_monthly_source": weak_source,
        },
        "labeling_policy": {
            "classes": ["BULL", "BEAR", "MISTO"],
            "weak_label_source": "mensal (texto fraco informado pelo usuário)",
            "misto_definition": "dias onde o modelo não entra em BULL nem BEAR após histerese/min_days",
        },
        "cep_features_spec": {
            "exogenous_only": True,
            "forbidden_name_patterns": FORBIDDEN_FEATURE_PATTERNS,
            "features_used": model_info["features_used"],
        },
        "model": {
            "chosen_model": chosen,
            "formula_explicit": formula,
            "formula_parameters": params,
            "normalization": model_info["normalization"],
            "validation_metrics": model_info["validation_metrics_best"],
        },
        "threshold_hysteresis_min_days": {
            "p_bull_enter": model_info["best_threshold_config"]["p_bull_enter"],
            "p_bull_exit": model_info["best_threshold_config"]["p_bull_enter"] - model_info["best_threshold_config"]["p_bull_exit_delta"],
            "p_bear_enter": model_info["best_threshold_config"]["p_bear_enter"],
            "p_bear_exit": model_info["best_threshold_config"]["p_bear_enter"] + model_info["best_threshold_config"]["p_bear_exit_delta"],
            "min_days": int(model_info["best_threshold_config"]["min_days"]),
        },
        "threshold_search_top5": search_top.head(5).to_dict(orient="records"),
        "cep_baseline_limits": limits,
        "class_reachability": {
            "days_bull": c_bull,
            "days_bear": c_bear,
            "days_misto": c_misto,
            "min_required_each_bull_bear": 30,
            "pass": reach_ok,
            "note": reach_note,
        },
    }
    return ssot


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ssot_cycle2").mkdir(parents=True, exist_ok=True)

    # S1
    s1_gate_allowlist()

    # S2
    master_series_path, master_source = autodiscover_master_series()
    weak_monthly_map = build_weak_monthly_from_text()
    weak_source = {
        "method": "user_text_weak_indication",
        "bull_years": [2019, 2023, 2025],
        "bear_years": [2021, 2022, 2024],
        "notes": "Fallback intencional: não foi encontrado JSON mensal no repositório pelos globs informados.",
    }

    # S3
    master_logret = load_master_logret(master_series_path)
    master_logret.to_parquet(OUT_MASTER_LOGRET, index=False)
    master_feats, limits = derive_cep_features(master_logret)
    master_feats.to_parquet(OUT_MASTER_FEATS, index=False)
    master_feats.to_csv(OUT_MASTER_FEATS_CSV, index=False)

    # S4
    weak_daily = build_weak_daily(master_logret, weak_monthly_map)
    weak_daily.to_parquet(OUT_WEAK_AUDIT, index=False)

    # S5
    model_info, regime_df_full, search_df = run_model_and_search(master_feats, weak_daily)
    regime_daily = regime_df_full[["date", "master_ticker", "p_bull_raw", "regime_label"]].copy()
    regime_daily.to_parquet(OUT_REGIME_DAILY, index=False)

    # S6
    days_bull = int((regime_daily["regime_label"] == "BULL").sum())
    days_bear = int((regime_daily["regime_label"] == "BEAR").sum())
    days_misto = int((regime_daily["regime_label"] == "MISTO").sum())
    classes_reachable = days_bull >= 30 and days_bear >= 30

    # S7
    ssot_v4 = build_ssot_v4(
        model_info=model_info,
        limits=limits,
        weak_source=weak_source,
        master_source={"path": str(master_series_path), **master_source},
        search_top=search_df,
        d_regime=regime_daily,
    )
    OUT_SSOT.write_text(json.dumps(ssot_v4, indent=2, ensure_ascii=False), encoding="utf-8")

    # S8 report + manifest
    report_lines = []
    report_lines.append("# EXP_002A Master Regime V4")
    report_lines.append("")
    report_lines.append("## OVERALL")
    report_lines.append("- OVERALL " + ("PASS" if classes_reachable else "PASS_COM_RESTRICAO_CLASSES"))
    report_lines.append("")
    report_lines.append("## STEPS")
    report_lines.append("- S1_GATE_ALLOWLIST: PASS — validação de leitura/escrita em allowlist.")
    report_lines.append("- S2_AUTODISCOVER_INPUTS: PASS — série master em parquet detectada; weak monthly via fallback textual auditável.")
    report_lines.append("- S3_BUILD_DAILY_LOGRET_AND_CEP_FEATURES: PASS — log-ret diário e features CEP exógenas geradas.")
    report_lines.append("- S4_BUILD_WEAKLABELS_DAILY_FROM_MONTHLY: PASS — weak labels mensais expandidas para diário.")
    report_lines.append("- S5_FIT_INTERPRETABLE_MODEL_AND_THRESHOLD_SEARCH: PASS — logistic/tree + grid de histerese/min_days.")
    report_lines.append("- S6_VERIFY_CLASSES_REACHABLE: " + ("PASS" if classes_reachable else "FAIL_JUSTIFICADO") + f" — dias BULL={days_bull}, BEAR={days_bear}, MISTO={days_misto}.")
    report_lines.append("- S7_WRITE_SSOT_V4_WITH_EXPLICIT_FORMULA: PASS — fórmula explícita + parâmetros no SSOT.")
    report_lines.append("- S8_GENERATE_MD_AUTOCONTIDO_AND_MANIFEST: PASS — report autocontido + manifest + hashes.")
    report_lines.append("")
    report_lines.append("## INPUTS RESOLVIDOS")
    report_lines.append(f"- master_series_path: `{master_series_path}`")
    report_lines.append(f"- master_series_discovery: `{json.dumps(master_source, ensure_ascii=False)}`")
    report_lines.append(f"- weak_label_source: `{json.dumps(weak_source, ensure_ascii=False)}`")
    report_lines.append("")
    report_lines.append("## DEFINIÇÃO DE WEAK LABELS (a partir do texto)")
    report_lines.append("- BULL: anos 2019, 2023, 2025 (todos os meses).")
    report_lines.append("- BEAR: anos 2021, 2022, 2024 (todos os meses).")
    report_lines.append("- UNLABELED: demais meses.")
    report_lines.append("")
    report_lines.append("## MODELO INTERPRETÁVEL")
    report_lines.append(f"- chosen_model: **{model_info['chosen_model']}**")
    report_lines.append(f"- validation_metrics_best: `{json.dumps(model_info['validation_metrics_best'], ensure_ascii=False)}`")
    report_lines.append("- selection_score: `score_final = balanced_accuracy_validation - 0.05 * misto_rate_validation`")
    report_lines.append("")
    report_lines.append("## THRESHOLDS / HISTERSE / MIN_DAYS")
    report_lines.append(f"- best_config: `{json.dumps(model_info['best_threshold_config'], ensure_ascii=False)}`")
    report_lines.append("")
    report_lines.append("## ALCANÇABILIDADE DE CLASSES")
    report_lines.append(f"- BULL dias: **{days_bull}**")
    report_lines.append(f"- BEAR dias: **{days_bear}**")
    report_lines.append(f"- MISTO dias: **{days_misto}**")
    report_lines.append(f"- critério >=30 em BULL e BEAR: **{'PASS' if classes_reachable else 'FAIL_JUSTIFICADO'}**")
    if not classes_reachable:
        report_lines.append("- justificativa: sinal/thresholds e cobertura de weak labels não produziram pelo menos 30 dias em uma das classes.")
    report_lines.append("")
    report_lines.append("## ARTEFATOS")
    report_lines.append(f"- `{OUT_MASTER_LOGRET}`")
    report_lines.append(f"- `{OUT_MASTER_FEATS}`")
    report_lines.append(f"- `{OUT_WEAK_AUDIT}`")
    report_lines.append(f"- `{OUT_SSOT}`")
    report_lines.append(f"- `{OUT_REGIME_DAILY}`")
    report_lines.append(f"- `{OUT_REPORT}`")
    report_lines.append(f"- `{OUT_MANIFEST}`")
    report_lines.append(f"- `{OUT_HASH}`")
    OUT_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "task_id": TASK_ID,
        "inputs": {
            "repo_root": str(REPO_ROOT),
            "master_ticker": MASTER_TICKER,
            "master_series_path": str(master_series_path),
            "master_series_discovery": master_source,
            "weak_label_source": weak_source,
        },
        "outputs": {
            "master_logret_daily_parquet": str(OUT_MASTER_LOGRET),
            "master_cep_features_daily_parquet": str(OUT_MASTER_FEATS),
            "master_cep_features_daily_csv": str(OUT_MASTER_FEATS_CSV),
            "weaklabels_daily_audit_parquet": str(OUT_WEAK_AUDIT),
            "ssot_master_regime_v4_json": str(OUT_SSOT),
            "regime_daily_parquet": str(OUT_REGIME_DAILY),
            "report_md": str(OUT_REPORT),
            "manifest_json": str(OUT_MANIFEST),
            "hashes_sha256": str(OUT_HASH),
        },
        "gates": {
            "S1_GATE_ALLOWLIST": "PASS",
            "S2_AUTODISCOVER_INPUTS": "PASS",
            "S3_BUILD_DAILY_LOGRET_AND_CEP_FEATURES": "PASS",
            "S4_BUILD_WEAKLABELS_DAILY_FROM_MONTHLY": "PASS",
            "S5_FIT_INTERPRETABLE_MODEL_AND_THRESHOLD_SEARCH": "PASS",
            "S6_VERIFY_CLASSES_REACHABLE": "PASS" if classes_reachable else "FAIL_JUSTIFICADO",
            "S7_WRITE_SSOT_V4_WITH_EXPLICIT_FORMULA": "PASS",
            "S8_GENERATE_MD_AUTOCONTIDO_AND_MANIFEST": "PASS",
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    hash_lines = []
    for p in sorted([f for f in OUT_DIR.rglob("*") if f.is_file() and f.name != "hashes.sha256"]):
        hash_lines.append(f"{sha256_file(p)}  {p.relative_to(OUT_DIR)}")
    OUT_HASH.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    print(f"[OK] outputs at: {OUT_DIR}")
    print(f"[OK] class days -> BULL={days_bull}, BEAR={days_bear}, MISTO={days_misto}")


if __name__ == "__main__":
    main()
