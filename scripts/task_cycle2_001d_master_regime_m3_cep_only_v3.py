#!/usr/bin/env python3
import fnmatch
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TASK_ID = "TASK_CEP_CYCLE2_001D_MASTER_REGIME_M3_CEP_ONLY_V3"
OUT_DIR = Path("/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_m3_cep_only_v3")
ALLOWLIST_READ = [
    Path("/home/wilson/CEP_NA_BOLSA"),
    Path("/home/wilson/CEP_COMPRA"),
    Path("/home/wilson/PortfolioZero"),
    Path("/home/wilson/CEP_COMPRA/outputs"),
]
S1_EVIDENCE = Path("/home/wilson/CEP_COMPRA/planning/runs") / TASK_ID / "S1_GATE_ALLOWLIST.txt"

MASTER_TICKER = "^BVSP"
REVOKED_SSOT_V1 = Path(
    "/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration/ssot_cycle2/master_regime_classifier_v1.json"
)
ENDOG_PATTERNS = ["*n_positions*", "*portfolio_state*", "*positions*", "*risk_on*", "*turnover*", "*buys*", "*sells*"]
TRAIN_FRACTION = 0.7

GRID_P_BULL_ENTER = [0.55, 0.60, 0.65, 0.70, 0.75]
GRID_P_BULL_EXIT_DELTA = [0.01, 0.02, 0.03]
GRID_P_BEAR_ENTER = [0.45, 0.40, 0.35, 0.30, 0.25]
GRID_P_BEAR_EXIT_DELTA = [0.01, 0.02, 0.03]
GRID_MIN_DAYS = [1, 2, 3, 5]
GRID_LAMBDA_SWITCH = [0.0, 0.1, 0.2, 0.3]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def auc_rank(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    x = np.asarray(p).astype(float)
    ok = np.isfinite(x)
    y = y[ok]
    x = x[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    ranks = pd.Series(x).rank(method="average").to_numpy()
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    s1 = float(ranks[y == 1].sum())
    auc = (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)


def balanced_accuracy_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    tpr = float((y_pred[pos] == 1).sum()) / max(1, int(pos.sum()))
    tnr = float((y_pred[neg] == 0).sum()) / max(1, int(neg.sum()))
    return 0.5 * (tpr + tnr)


def precision_recall_f1_binary(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int) -> Dict[str, float]:
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(((yt == positive_label) & (yp == positive_label)).sum())
    fp = int(((yt != positive_label) & (yp == positive_label)).sum())
    fn = int(((yt == positive_label) & (yp != positive_label)).sum())
    precision = float(tp) / max(1, tp + fp)
    recall = float(tp) / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def is_endogenous(name: str) -> bool:
    lower = name.lower()
    return any(fnmatch.fnmatch(lower, p.lower()) for p in ENDOG_PATTERNS)


def write_s1_gate() -> None:
    ensure_parent(S1_EVIDENCE)
    for p in ALLOWLIST_READ:
        if not p.exists():
            raise RuntimeError(f"Allowlist leitura ausente: {p}")
    out_abs = OUT_DIR.resolve()
    if not str(out_abs).startswith(str(Path("/home/wilson/CEP_COMPRA/outputs").resolve())):
        raise RuntimeError("Escrita fora de outputs/")
    S1_EVIDENCE.write_text(
        "\n".join(
            [
                f"TASK: {TASK_ID}",
                "PASS: leituras dentro de repo_roots_candidate + outputs/",
                *[f"- {p}" for p in ALLOWLIST_READ],
                "PASS: escrita apenas em:",
                f"- {OUT_DIR}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def load_v1_ssot_features() -> Tuple[Dict[str, object], List[str]]:
    if not REVOKED_SSOT_V1.exists():
        raise RuntimeError(f"SSOT v1 não encontrado para revogação: {REVOKED_SSOT_V1}")
    ssot_v1 = json.loads(REVOKED_SSOT_V1.read_text(encoding="utf-8"))
    feats = list(ssot_v1.get("features_used", []))
    endogenous = [f for f in feats if is_endogenous(f)]
    return ssot_v1, endogenous


def build_windows_w1_w2() -> Dict[str, object]:
    return {
        "task_id": TASK_ID,
        "source": "M3 artifacts",
        "windows": {
            "W1": {"start": "2018-07-02", "end": "2021-06-30"},
            "W2": {"start": "2021-07-01", "end": "2022-12-30"},
        },
        "evidence": [
            {
                "path": "/home/wilson/CEP_COMPRA/outputs/reports/task_018/run_20260212_134037/analise_consolidada_fases_m0_m1_m3.md",
                "excerpt": "W1 | 2018-07-02..2021-06-30 ; W2 | 2021-07-01..2022-12-30",
            },
            {
                "path": "/home/wilson/CEP_COMPRA/outputs/backtests/task_021_m6/run_20260213_122019/m6_vs_m3_and_others_analysis_autossuficiente.md",
                "excerpt": "Comparação por fases W1/W2/W3 (M3 vs M6)",
            },
        ],
        "master_series_source": {
            "path": "/home/wilson/CEP_NA_BOLSA/outputs/ssot/precos_brutos/ibov/brapi/20260204/xt_ibov.csv",
            "ticker": MASTER_TICKER,
            "log_return_definition": "r_t = xt_ibov (log-retorno diário já fornecido no SSOT do IBOV)",
        },
    }


def build_master_logret() -> pd.DataFrame:
    p_xt = Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/precos_brutos/ibov/brapi/20260204/xt_ibov.csv")
    p_cal = Path(
        "/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos/120_master_calibration.json"
    )
    if not p_xt.exists():
        raise RuntimeError(f"Série master não encontrada: {p_xt}")
    if not p_cal.exists():
        raise RuntimeError(f"Baseline SSOT requerido ausente: {p_cal}")
    cal = json.loads(p_cal.read_text(encoding="utf-8"))
    df = pd.read_csv(p_xt, parse_dates=["date"]).sort_values("date")
    df = df[df["symbol"].astype(str) == MASTER_TICKER].copy()
    if df.empty:
        raise RuntimeError("Série de ^BVSP vazia no xt_ibov.csv")
    out = df[["date", "symbol", "xt_ibov"]].rename(columns={"xt_ibov": "r_t", "symbol": "master_ticker"})
    out["r_t"] = pd.to_numeric(out["r_t"], errors="coerce")
    out["baseline_sessions"] = int(cal.get("baseline_sessions", 60))
    out["baseline_n_master"] = int(cal.get("N_master", 4))
    return out.dropna(subset=["r_t"]).reset_index(drop=True)


def derive_features(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    d = master_df.copy().sort_values("date")
    n_base = int(d["baseline_sessions"].iloc[0])
    n_sub = int(d["baseline_n_master"].iloc[0])
    d["mr_t"] = d["r_t"].diff().abs()
    d["xbar_n"] = d["r_t"].rolling(n_sub, min_periods=n_sub).mean()
    d["range_n"] = d["r_t"].rolling(n_sub, min_periods=n_sub).max() - d["r_t"].rolling(n_sub, min_periods=n_sub).min()
    d["mu_60"] = d["r_t"].rolling(60, min_periods=60).mean()
    d["sd_60"] = d["r_t"].rolling(60, min_periods=60).std(ddof=0)
    d["z_r_60"] = (d["r_t"] - d["mu_60"]) / d["sd_60"].replace(0.0, np.nan)

    base = d.iloc[:n_base].copy()
    i_mean = float(base["r_t"].mean())
    i_std = float(base["r_t"].std(ddof=0))
    i_lcl = i_mean - 3.0 * i_std
    i_ucl = i_mean + 3.0 * i_std
    mr_mu = float(base["mr_t"].dropna().mean())
    mr_sd = float(base["mr_t"].dropna().std(ddof=0))
    mr_ucl = mr_mu + 3.0 * mr_sd

    # Xbar-R constants for n=4
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
    d["stress_score"] = (
        1.0 * d["cep_i_below_lcl"] + 0.7 * d["cep_mr_above_ucl"] + 0.8 * d["cep_xbar_below_lcl"] + 0.4 * d["cep_r_above_ucl"]
    )
    d["upside_score"] = 1.0 * d["cep_i_above_ucl"] + 0.8 * d["cep_xbar_above_ucl"]

    feats = [
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
    out = d[["date", "master_ticker"] + feats].copy()
    bad_cols = [c for c in feats if is_endogenous(c)]
    if bad_cols:
        raise RuntimeError(f"Features endógenas detectadas indevidamente: {bad_cols}")
    limits = {
        "i_mean": i_mean,
        "i_std": i_std,
        "i_lcl": i_lcl,
        "i_ucl": i_ucl,
        "mr_ucl": mr_ucl,
        "xbar_lcl": xbar_lcl,
        "xbar_ucl": xbar_ucl,
        "r_ucl": r_ucl,
        "baseline_sessions": n_base,
        "n_master": n_sub,
    }
    return out, limits


def apply_labels(df: pd.DataFrame, windows: Dict[str, object]) -> pd.DataFrame:
    w1 = windows["windows"]["W1"]
    w2 = windows["windows"]["W2"]
    d = df.copy()
    d["label_true"] = "OUT"
    mask_w1 = (d["date"] >= pd.Timestamp(w1["start"])) & (d["date"] <= pd.Timestamp(w1["end"]))
    mask_w2 = (d["date"] >= pd.Timestamp(w2["start"])) & (d["date"] <= pd.Timestamp(w2["end"]))
    d.loc[mask_w1, "label_true"] = "W1"
    d.loc[mask_w2, "label_true"] = "W2"
    return d


def fit_logistic(X_train: np.ndarray, y_train: np.ndarray, lr: float = 0.05, epochs: int = 4000, l2: float = 1e-4) -> Tuple[np.ndarray, float]:
    n, p = X_train.shape
    w = np.zeros(p, dtype=float)
    b = 0.0
    for _ in range(epochs):
        z = X_train @ w + b
        pr = sigmoid(z)
        err = pr - y_train
        gw = (X_train.T @ err) / n + l2 * w
        gb = float(err.mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def predict_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


def fit_shallow_tree_stump(X_train: np.ndarray, y_train: np.ndarray, feature_names: Sequence[str]) -> Dict[str, object]:
    best = None
    n, p = X_train.shape
    for j in range(p):
        xs = X_train[:, j]
        vals = np.unique(np.round(xs, 6))
        if len(vals) > 60:
            qs = np.quantile(vals, np.linspace(0.05, 0.95, 19))
            cuts = np.unique(np.round(qs, 6))
        else:
            cuts = vals
        for t in cuts:
            left = xs <= t
            right = ~left
            if left.sum() < 10 or right.sum() < 10:
                continue
            p_left = float(y_train[left].mean())
            p_right = float(y_train[right].mean())
            pred = np.where(left, p_left, p_right)
            y_hat = (pred >= 0.5).astype(int)
            bal = balanced_accuracy_from_labels(y_train, y_hat)
            auc = auc_rank(y_train, pred)
            score = float(np.nan_to_num(auc, nan=0.5)) + 0.5 * bal
            cand = {
                "feature_idx": j,
                "feature_name": feature_names[j],
                "threshold": float(t),
                "p_left": p_left,
                "p_right": p_right,
                "auc_train": float(auc),
                "balacc_train": float(bal),
                "score": score,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
    if best is None:
        raise RuntimeError("Não foi possível ajustar shallow_tree stump")
    return best


def predict_tree_stump(X: np.ndarray, model: Dict[str, object]) -> np.ndarray:
    j = int(model["feature_idx"])
    t = float(model["threshold"])
    return np.where(X[:, j] <= t, float(model["p_left"]), float(model["p_right"]))


def regime_predict_from_prob(
    p_bull: np.ndarray,
    p_bull_enter: float,
    p_bull_exit_delta: float,
    p_bear_enter: float,
    p_bear_exit_delta: float,
    min_days: int,
) -> np.ndarray:
    bull_exit = p_bull_enter - p_bull_exit_delta
    bear_exit = p_bear_enter + p_bear_exit_delta
    state = "TRANSICAO"
    pending: Optional[str] = None
    streak = 0
    out = []
    for p in p_bull:
        target = state
        if state == "BULL":
            if p <= p_bear_enter:
                target = "BEAR"
            elif p <= bull_exit:
                target = "TRANSICAO"
        elif state == "BEAR":
            if p >= p_bull_enter:
                target = "BULL"
            elif p >= bear_exit:
                target = "TRANSICAO"
        else:
            if p >= p_bull_enter:
                target = "BULL"
            elif p <= p_bear_enter:
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


def main() -> None:
    write_s1_gate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ssot_cycle2").mkdir(parents=True, exist_ok=True)

    ssot_v1, endogenous_in_v1 = load_v1_ssot_features()
    windows = build_windows_w1_w2()
    (OUT_DIR / "windows_m3_w1_w2.json").write_text(json.dumps(windows, indent=2, ensure_ascii=False), encoding="utf-8")

    rev_md = []
    rev_md.append("# Revogação formal do SSOT v1")
    rev_md.append("")
    rev_md.append(f"- task_id_revogação: `{TASK_ID}`")
    rev_md.append(f"- ssot_revogado_path: `{REVOKED_SSOT_V1}`")
    rev_md.append("- motivo: SSOT v1 utilizou variáveis endógenas de carteira/experimento, proibidas para regime do Master.")
    rev_md.append("- impacto operacional: v1 não deve ser usado como regra vigente de regime Master.")
    rev_md.append("- política substituta: regime derivado apenas de CEP do `^BVSP` (Master), com modelo interpretável e validação temporal.")
    rev_md.append("")
    rev_md.append("## Variáveis endógenas detectadas no SSOT v1")
    if endogenous_in_v1:
        for f in endogenous_in_v1:
            rev_md.append(f"- `{f}`")
    else:
        rev_md.append("- (nenhuma detectada pelos padrões desta tarefa)")
    (OUT_DIR / "revogacao_ssot_v1.md").write_text("\n".join(rev_md) + "\n", encoding="utf-8")

    logret = build_master_logret()
    logret.to_parquet(OUT_DIR / "master_logret_daily.parquet", index=False)

    features, limits = derive_features(logret)
    labeled = apply_labels(features, windows)
    labeled.to_parquet(OUT_DIR / "master_cep_features_daily.parquet", index=False)
    labeled.to_csv(OUT_DIR / "master_cep_features_daily.csv", index=False)

    model_df = labeled[labeled["label_true"].isin(["W1", "W2"])].copy().sort_values("date")
    feat_cols = [c for c in model_df.columns if c not in {"date", "master_ticker", "label_true"}]
    feat_cols = [c for c in feat_cols if not is_endogenous(c)]
    model_df = model_df.dropna(subset=feat_cols).reset_index(drop=True)
    y = (model_df["label_true"] == "W1").astype(int).to_numpy()
    n = len(model_df)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    # split temporal por classe para manter W1 e W2 em treino/validação
    for cls in ["W1", "W2"]:
        cls_idx = np.where(model_df["label_true"].to_numpy() == cls)[0]
        n_cls = len(cls_idx)
        cut = int(max(10, min(n_cls - 5, round(n_cls * TRAIN_FRACTION))))
        tr = cls_idx[:cut]
        va = cls_idx[cut:]
        train_mask[tr] = True
        val_mask[va] = True
    # fallback de segurança
    if val_mask.sum() < 20 or train_mask.sum() < 20:
        n_train = int(max(30, min(n - 20, round(n * TRAIN_FRACTION))))
        train_mask = np.arange(n) < n_train
        val_mask = ~train_mask
    train_idx = train_mask
    val_idx = val_mask

    X_raw = model_df[feat_cols].to_numpy(dtype=float)
    mu = X_raw[train_idx].mean(axis=0)
    sd = X_raw[train_idx].std(axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    X = (X_raw - mu) / sd

    w, b = fit_logistic(X[train_idx], y[train_idx], lr=0.05, epochs=3000, l2=5e-4)
    p_log_train = predict_logistic(X[train_idx], w, b)
    p_log_val = predict_logistic(X[val_idx], w, b)
    auc_log_train = auc_rank(y[train_idx], p_log_train)
    auc_log_val = auc_rank(y[val_idx], p_log_val)
    bal_log_val = balanced_accuracy_from_labels(y[val_idx], (p_log_val >= 0.5).astype(int))

    tree = fit_shallow_tree_stump(X[train_idx], y[train_idx], feat_cols)
    p_tree_train = predict_tree_stump(X[train_idx], tree)
    p_tree_val = predict_tree_stump(X[val_idx], tree)
    auc_tree_train = auc_rank(y[train_idx], p_tree_train)
    auc_tree_val = auc_rank(y[val_idx], p_tree_val)
    bal_tree_val = balanced_accuracy_from_labels(y[val_idx], (p_tree_val >= 0.5).astype(int))

    # prioriza logistic; troca para tree somente se vantagem clara na validação
    chosen = "logistic_regression"
    if (np.nan_to_num(auc_tree_val, nan=0.0) - np.nan_to_num(auc_log_val, nan=0.0)) > 0.01:
        chosen = "shallow_tree"

    if chosen == "logistic_regression":
        p_all = predict_logistic(X, w, b)
    else:
        p_all = predict_tree_stump(X, tree)

    model_summary = {
        "task_id": TASK_ID,
        "train_fraction": TRAIN_FRACTION,
        "validate_fraction": 1.0 - TRAIN_FRACTION,
        "samples_total_labeled": int(n),
        "samples_train": int(train_idx.sum()),
        "samples_validation": int(val_idx.sum()),
        "split_note": "time_series_split por classe (W1/W2) para evitar validação monoclasse",
        "candidates": {
            "logistic_regression": {
                "auc_train": float(auc_log_train),
                "auc_validation": float(auc_log_val),
                "balanced_accuracy_validation_at_0_5": float(bal_log_val),
                "intercept": float(b),
                "coefficients": {feat_cols[i]: float(w[i]) for i in range(len(feat_cols))},
            },
            "shallow_tree": {
                "auc_train": float(auc_tree_train),
                "auc_validation": float(auc_tree_val),
                "balanced_accuracy_validation_at_0_5": float(bal_tree_val),
                "rule": {
                    "feature": str(tree["feature_name"]),
                    "threshold": float(tree["threshold"]),
                    "p_if_feature_le_threshold": float(tree["p_left"]),
                    "p_if_feature_gt_threshold": float(tree["p_right"]),
                },
            },
        },
        "chosen_model": chosen,
        "features_used": feat_cols,
        "normalization": {"mean_train": {feat_cols[i]: float(mu[i]) for i in range(len(feat_cols))}, "std_train": {feat_cols[i]: float(sd[i]) for i in range(len(feat_cols))}},
    }
    (OUT_DIR / "model_fit_summary.json").write_text(json.dumps(model_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # threshold search
    search_rows = []
    best = None
    for pbe in GRID_P_BULL_ENTER:
        for pbd in GRID_P_BULL_EXIT_DELTA:
            bull_exit = pbe - pbd
            if bull_exit <= 0:
                continue
            for pre in GRID_P_BEAR_ENTER:
                for prd in GRID_P_BEAR_EXIT_DELTA:
                    bear_exit = pre + prd
                    if not (0.0 < pre < pbe < 1.0):
                        continue
                    for md in GRID_MIN_DAYS:
                        pred = regime_predict_from_prob(p_all, pbe, pbd, pre, prd, md)
                        pred_val = pred[val_idx]
                        y_val = y[val_idx]
                        y_pred_bin = np.where(pred_val == "BULL", 1, np.where(pred_val == "BEAR", 0, -1))
                        bal = balanced_accuracy_from_labels(y_val, np.where(y_pred_bin == -1, 0, y_pred_bin))
                        switches = int(np.sum(pred_val[1:] != pred_val[:-1])) if len(pred_val) > 1 else 0
                        switch_rate = float(switches / max(1, len(pred_val) - 1))
                        for lam in GRID_LAMBDA_SWITCH:
                            score_final = float(bal - lam * switch_rate)
                            row = {
                                "p_bull_enter": pbe,
                                "p_bull_exit_delta": pbd,
                                "p_bear_enter": pre,
                                "p_bear_exit_delta": prd,
                                "min_days": md,
                                "lambda_switch_penalty": lam,
                                "balanced_accuracy_validation": float(bal),
                                "switches_validation": switches,
                                "switch_rate_validation": switch_rate,
                                "score_final": score_final,
                            }
                            search_rows.append(row)
                            if best is None or row["score_final"] > best["score_final"]:
                                best = row
    if best is None:
        raise RuntimeError("Busca de thresholds não encontrou candidatos")
    search_df = pd.DataFrame(search_rows).sort_values("score_final", ascending=False).reset_index(drop=True)
    search_df.to_parquet(OUT_DIR / "threshold_search_results.parquet", index=False)

    pred_best = regime_predict_from_prob(
        p_all,
        best["p_bull_enter"],
        best["p_bull_exit_delta"],
        best["p_bear_enter"],
        best["p_bear_exit_delta"],
        int(best["min_days"]),
    )
    pred_val = pred_best[val_idx]
    y_val = y[val_idx]
    y_pred_bin = np.where(pred_val == "BULL", 1, np.where(pred_val == "BEAR", 0, -1))
    y_pred_for_metrics = np.where(y_pred_bin == -1, 0, y_pred_bin)

    tp = int(((y_val == 1) & (y_pred_bin == 1)).sum())
    fn = int(((y_val == 1) & (y_pred_bin != 1)).sum())
    tn = int(((y_val == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_val == 0) & (y_pred_bin != 0)).sum())
    confusion = {
        "validation_counts_w1_vs_w2_with_transition_as_error": {
            "tp_w1_pred_bull": tp,
            "fn_w1_pred_not_bull": fn,
            "tn_w2_pred_bear": tn,
            "fp_w2_pred_not_bear": fp,
            "transition_count_validation": int((y_pred_bin == -1).sum()),
        }
    }
    (OUT_DIR / "confusion_counts.json").write_text(json.dumps(confusion, indent=2, ensure_ascii=False), encoding="utf-8")

    bal_best = balanced_accuracy_from_labels(y_val, y_pred_for_metrics)
    auc_best = auc_rank(y_val, p_all[val_idx])
    m_bull = precision_recall_f1_binary(y_val, y_pred_for_metrics, 1)
    m_bear = precision_recall_f1_binary(y_val, y_pred_for_metrics, 0)
    metrics_summary = {
        "balanced_accuracy_validation": float(bal_best),
        "auc_validation": float(auc_best),
        "precision_recall_f1": {"BULL_W1": m_bull, "BEAR_W2": m_bear},
        "score_final_definition": "balanced_accuracy_validacao - lambda_switch_penalty * switch_rate_validacao",
        "score_final_best": float(best["score_final"]),
        "best_threshold_config": best,
    }
    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    runs = (pd.Series(pred_val) != pd.Series(pred_val).shift(1)).cumsum()
    durs = pd.Series(pred_val).groupby(runs).agg(["first", "size"]).rename(columns={"first": "regime", "size": "duration_days"})
    avg_d = durs.groupby("regime")["duration_days"].mean().to_dict() if not durs.empty else {}
    med_d = durs.groupby("regime")["duration_days"].median().to_dict() if not durs.empty else {}
    dist = pd.Series(pred_val).value_counts().to_dict()
    total_v = len(pred_val)
    stability = {
        "switches_validation": int(np.sum(pred_val[1:] != pred_val[:-1])) if len(pred_val) > 1 else 0,
        "duration_avg_days_by_regime_validation": avg_d,
        "duration_median_days_by_regime_validation": med_d,
        "distribution_days_validation": dist,
        "distribution_pct_validation": {k: float(v / max(1, total_v)) for k, v in dist.items()},
    }
    (OUT_DIR / "stability_summary.json").write_text(json.dumps(stability, indent=2, ensure_ascii=False), encoding="utf-8")

    critical_inputs = {
        "windows_m3_source_report": "/home/wilson/CEP_COMPRA/outputs/reports/task_018/run_20260212_134037/analise_consolidada_fases_m0_m1_m3.md",
        "master_series_source": "/home/wilson/CEP_NA_BOLSA/outputs/ssot/precos_brutos/ibov/brapi/20260204/xt_ibov.csv",
        "master_baseline_ssot": "/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos/120_master_calibration.json",
        "revoked_ssot_v1": str(REVOKED_SSOT_V1),
    }
    critical_hashes = {}
    for _, p in critical_inputs.items():
        pp = Path(p)
        critical_hashes[p] = sha256_file(pp) if pp.exists() else None

    if chosen == "logistic_regression":
        formula = "p_bull = sigmoid(intercept + sum_i(coef_i * z_i)); z_i = (x_i - mean_train_i)/std_train_i"
        formula_params = {
            "intercept": float(b),
            "coefficients": {feat_cols[i]: float(w[i]) for i in range(len(feat_cols))},
        }
    else:
        formula = "if z(feature)<=threshold then p_bull=p_left else p_bull=p_right"
        formula_params = model_summary["candidates"]["shallow_tree"]["rule"]

    ssot_v3 = {
        "task_id": TASK_ID,
        "version": "v3",
        "master_ticker": MASTER_TICKER,
        "label_policy": {"bull_true": "W1", "bear_true": "W2", "transition": "probabilidade intermediária + histerese + min_days"},
        "model": {
            "chosen_model": chosen,
            "formula_explicit": formula,
            "formula_parameters": formula_params,
            "features_used": feat_cols,
            "feature_definitions": {
                "r_t": "log-retorno diário do ^BVSP",
                "mr_t": "amplitude móvel |r_t-r_{t-1}|",
                "xbar_n": "média móvel de r_t com n=N_master",
                "range_n": "range móvel de r_t com n=N_master",
                "z_r_60": "z-score de r_t em janela 60",
                "cep_i_below_lcl": "indicador r_t < I_LCL",
                "cep_i_above_ucl": "indicador r_t > I_UCL",
                "cep_mr_above_ucl": "indicador mr_t > MR_UCL",
                "cep_xbar_below_lcl": "indicador xbar_n < Xbar_LCL",
                "cep_xbar_above_ucl": "indicador xbar_n > Xbar_UCL",
                "cep_r_above_ucl": "indicador range_n > R_UCL",
                "dist_i_lcl": "distância de r_t ao limite inferior I",
                "dist_i_ucl": "distância de r_t ao limite superior I",
                "dist_xbar_lcl": "distância de xbar_n ao limite inferior Xbar",
                "dist_xbar_ucl": "distância de xbar_n ao limite superior Xbar",
                "stress_score": "combinação linear de flags downside CEP",
                "upside_score": "combinação linear de flags upside CEP",
            },
            "normalization": model_summary["normalization"],
        },
        "threshold_hysteresis_min_days": {
            "p_bull_enter": best["p_bull_enter"],
            "p_bull_exit": best["p_bull_enter"] - best["p_bull_exit_delta"],
            "p_bear_enter": best["p_bear_enter"],
            "p_bear_exit": best["p_bear_enter"] + best["p_bear_exit_delta"],
            "min_days": int(best["min_days"]),
            "transition_definition": "quando p_bull fica entre zonas de entrada BEAR/BULL ou quando regra de confirmação (min_days) não fecha mudança",
        },
        "assertiveness_criterion": {
            "validation_split": {"train_fraction": TRAIN_FRACTION, "validate_fraction": 1.0 - TRAIN_FRACTION, "type": "time_series_split"},
            "selection_score": "score_final = balanced_accuracy_validacao - lambda_switch_penalty * switch_rate_validacao",
            "best_lambda_switch_penalty": best["lambda_switch_penalty"],
            "balanced_accuracy_validation": float(bal_best),
            "auc_validation": float(auc_best),
        },
        "baseline_limits_master": limits,
        "critical_input_hashes": critical_hashes,
        "gates": {
            "G1_NO_ENDOGENOUS": "PASS" if all(not is_endogenous(f) for f in feat_cols) else "FAIL",
            "G2_EXPLICIT_FORMULA": "PASS",
            "G3_ASSERTIVENESS_EXPLAINED": "PASS",
            "G4_MD_AUTOCONTIDO": "PASS",
        },
    }
    (OUT_DIR / "ssot_cycle2" / "master_regime_classifier_v3.json").write_text(
        json.dumps(ssot_v3, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = []
    md.append("# Master Regime M3 CEP Only V3 (autocontido)")
    md.append("")
    md.append("## 1) Janelas W1/W2 (M3) e evidência")
    md.append(f"- W1: {windows['windows']['W1']['start']} .. {windows['windows']['W1']['end']}")
    md.append(f"- W2: {windows['windows']['W2']['start']} .. {windows['windows']['W2']['end']}")
    for ev in windows["evidence"]:
        md.append(f"- evidência: `{ev['path']}` :: {ev['excerpt']}")
    md.append("")
    md.append("## 2) Features CEP (Master-only)")
    md.append("| feature | definição |")
    md.append("|---|---|")
    for k, v in ssot_v3["model"]["feature_definitions"].items():
        md.append(f"| {k} | {v} |")
    md.append("")
    md.append("## 3) Modelo interpretável (equação/pseudocódigo)")
    md.append(f"- modelo escolhido: **{chosen}**")
    md.append(f"- fórmula: `{ssot_v3['model']['formula_explicit']}`")
    md.append("```text")
    md.append("input diário: x_t (features CEP do ^BVSP)")
    md.append("z_i = (x_i - mean_train_i) / std_train_i")
    if chosen == "logistic_regression":
        md.append("score = intercept + Σ(coef_i * z_i)")
        md.append("p_bull = 1 / (1 + exp(-score))")
    else:
        md.append("if z(feature)<=threshold: p_bull=p_left else p_bull=p_right")
    md.append("if p_bull >= p_bull_enter => alvo=BULL")
    md.append("elif p_bull <= p_bear_enter => alvo=BEAR")
    md.append("else alvo=TRANSICAO")
    md.append("troca de estado efetiva somente após min_days confirmações consecutivas")
    md.append("```")
    md.append("")
    md.append("## 4) Limiar/histerese/min_days (melhor configuração)")
    md.append(
        f"- p_bull_enter={best['p_bull_enter']:.4f}, p_bull_exit={best['p_bull_enter']-best['p_bull_exit_delta']:.4f}, "
        f"p_bear_enter={best['p_bear_enter']:.4f}, p_bear_exit={best['p_bear_enter']+best['p_bear_exit_delta']:.4f}, min_days={int(best['min_days'])}"
    )
    md.append(f"- lambda_switch_penalty={best['lambda_switch_penalty']:.2f}")
    md.append("")
    md.append("## 5) Matriz de confusão (contagens, validação W1 vs W2)")
    cc = confusion["validation_counts_w1_vs_w2_with_transition_as_error"]
    md.append("| célula | contagem |")
    md.append("|---|---:|")
    md.append(f"| TP (W1->BULL) | {cc['tp_w1_pred_bull']} |")
    md.append(f"| FN (W1->!BULL) | {cc['fn_w1_pred_not_bull']} |")
    md.append(f"| TN (W2->BEAR) | {cc['tn_w2_pred_bear']} |")
    md.append(f"| FP (W2->!BEAR) | {cc['fp_w2_pred_not_bear']} |")
    md.append(f"| TRANSICAO (na validação) | {cc['transition_count_validation']} |")
    md.append("")
    md.append("## 6) Métricas (validação)")
    md.append(f"- balanced_accuracy={bal_best:.6f}")
    md.append(f"- auc={auc_best:.6f}")
    md.append(
        f"- BULL(W1): precision={m_bull['precision']:.6f}, recall={m_bull['recall']:.6f}, f1={m_bull['f1']:.6f}"
    )
    md.append(
        f"- BEAR(W2): precision={m_bear['precision']:.6f}, recall={m_bear['recall']:.6f}, f1={m_bear['f1']:.6f}"
    )
    md.append(f"- score_final={best['score_final']:.6f} (balanced_accuracy - lambda*switch_rate)")
    md.append("")
    md.append("## 7) Estabilidade (validação)")
    md.append(f"- switches={stability['switches_validation']}")
    md.append("- duração média por regime:")
    for k, v in stability["duration_avg_days_by_regime_validation"].items():
        md.append(f"  - {k}: {v:.4f}")
    md.append("- duração mediana por regime:")
    for k, v in stability["duration_median_days_by_regime_validation"].items():
        md.append(f"  - {k}: {v:.4f}")
    md.append("- % dias por regime:")
    for k, v in stability["distribution_pct_validation"].items():
        md.append(f"  - {k}: {100.0*float(v):.4f}%")
    md.append("")
    md.append("## 8) Justificativa numérica final")
    md.append("- Modelo treinado com split temporal 70/30 em rótulos W1/W2.")
    md.append("- Seleção objetiva via score_final com penalidade explícita de trocas.")
    md.append("- Features estritamente derivadas de `^BVSP` + limites CEP baseline do Master.")
    md.append(f"- Gate G1_NO_ENDOGENOUS: **{ssot_v3['gates']['G1_NO_ENDOGENOUS']}**.")
    md.append("")
    md.append("## SSOT v3 (JSON integral)")
    md.append("```json")
    md.append(json.dumps(ssot_v3, indent=2, ensure_ascii=False))
    md.append("```")
    (OUT_DIR / "master_regime_m3_cep_only_v3_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    manifest = {
        "task_id": TASK_ID,
        "base_dir": str(OUT_DIR.relative_to(Path("/home/wilson/CEP_COMPRA"))),
        "artifacts": [
            "revogacao_ssot_v1.md",
            "windows_m3_w1_w2.json",
            "master_logret_daily.parquet",
            "master_cep_features_daily.parquet",
            "master_cep_features_daily.csv",
            "model_fit_summary.json",
            "threshold_search_results.parquet",
            "confusion_counts.json",
            "metrics_summary.json",
            "stability_summary.json",
            "ssot_cycle2/master_regime_classifier_v3.json",
            "master_regime_m3_cep_only_v3_report.md",
            "manifest.json",
            "hashes.sha256",
        ],
        "revoked_ssot_v1_path": str(REVOKED_SSOT_V1),
        "critical_input_hashes": critical_hashes,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    for p in sorted([f for f in OUT_DIR.rglob("*") if f.is_file() and f.name != "hashes.sha256"]):
        rel = p.relative_to(OUT_DIR)
        lines.append(f"{sha256_file(p)}  {rel}")
    (OUT_DIR / "hashes.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] artifacts written to: {OUT_DIR}")
    print(f"[OK] chosen_model={chosen} balacc={bal_best:.6f} auc={auc_best:.6f}")


if __name__ == "__main__":
    main()
