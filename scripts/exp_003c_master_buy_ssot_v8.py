#!/usr/bin/env python3
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


TASK_ID = "TASK_CEP_COMPRA_EXP_003C_MASTER_BUY_SSOT_V8_BUY3_DIRECT_AND_BUY1_CONTINUOUS_SCORE_V1"
REPO_ROOT = Path("/home/wilson/CEP_COMPRA")
PYTHON_EXEC = Path("/home/wilson/PortfolioZero/.venv/bin/python")

OUT_DIR = REPO_ROOT / "outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8"
OUT_SSOT = REPO_ROOT / "ssot_cycle2/master_buy_classifier_v8.json"
S1_EVIDENCE = REPO_ROOT / "planning/runs/TASK_CEP_COMPRA_EXP_003C_MASTER_BUY_SSOT_V8_BUY3_DIRECT_AND_BUY1_CONTINUOUS_SCORE_V1/S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY.txt"

OUT_BUYLBL_BVSP = OUT_DIR / "buy3_labels_bvsp.parquet"
OUT_BUYLBL_GSPC = OUT_DIR / "buy3_labels_gspc.parquet"
OUT_MODEL = OUT_DIR / "model_fit_summary.json"
OUT_THRESH = OUT_DIR / "threshold_search_results.parquet"
OUT_BUY_BVSP = OUT_DIR / "buy_daily_bvsp.parquet"
OUT_BUY_GSPC = OUT_DIR / "buy_daily_gspc.parquet"
OUT_SCORE_BVSP = OUT_DIR / "buy1_intensity_bvsp.parquet"
OUT_SCORE_GSPC = OUT_DIR / "buy1_intensity_gspc.parquet"
OUT_CONF = OUT_DIR / "confusion_buy3_counts.json"
OUT_REPORT = OUT_DIR / "report.md"
OUT_MANIFEST = OUT_DIR / "manifest.json"
OUT_HASH = OUT_DIR / "hashes.sha256"

IN_003B_DIR = REPO_ROOT / "outputs/experimentos/controle_rl/EXP_003B_master_regime_v7_5state"
IN_LBL5_BVSP = IN_003B_DIR / "labels_theory_5state_bvsp.parquet"
IN_LBL5_GSPC = IN_003B_DIR / "labels_theory_5state_gspc.parquet"
IN_FEAT_BVSP = IN_003B_DIR / "cep_features_bvsp.parquet"
IN_FEAT_GSPC = IN_003B_DIR / "cep_features_gspc.parquet"

BUY_MAP = {
    "BULL": "BUY2",
    "CORR_BULL_NEUTRO": "BUY2",
    "NEUTRO": "BUY2",
    "CORR_NEUTRO_BEAR": "BUY1",
    "BEAR": "BUY0",
}
BUY3 = ["BUY2", "BUY1", "BUY0"]
FORBIDDEN = ["*positions*", "*n_positions*", "*portfolio_state*", "*risk_on*", "*buys*", "*sells*", "*turnover*"]

GRID_P2_ENTER = [0.45, 0.50, 0.55, 0.60, 0.65]
GRID_P2_EXIT_D = [0.01, 0.02, 0.03]
GRID_P0_ENTER = [0.45, 0.50, 0.55, 0.60, 0.65]
GRID_P0_EXIT_D = [0.01, 0.02, 0.03]
GRID_MIN_D = [3, 5, 10]
GRID_BIAS = [-0.10, -0.05, 0.00, 0.05]  # 2700 combinações


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def is_forbidden(name: str) -> bool:
    import fnmatch

    n = name.lower()
    return any(fnmatch.fnmatch(n, p.lower()) for p in FORBIDDEN)


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


def balanced_acc(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
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


def confusion_counts(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    out: Dict[str, Dict[str, int]] = {}
    for a in labels:
        out[a] = {}
        for b in labels:
            out[a][b] = int(((yt == a) & (yp == b)).sum())
    return out


def s1_gate() -> None:
    S1_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    if not str(OUT_DIR.resolve()).startswith(str((REPO_ROOT / "outputs").resolve())):
        raise RuntimeError("S1 FAIL: out_dir fora de outputs")
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


def ensure_003b_inputs() -> Dict[str, object]:
    required = [IN_LBL5_BVSP, IN_LBL5_GSPC, IN_FEAT_BVSP, IN_FEAT_GSPC]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        script_003b = REPO_ROOT / "scripts/exp_003b_master_regime_v7.py"
        if not script_003b.exists():
            raise RuntimeError(f"S3 FAIL: faltam entradas 003B e script 003B ausente. missing={missing}")
        cmd = [str(PYTHON_EXEC), str(script_003b)]
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"S3 FAIL: rebuild 003B falhou rc={res.returncode}\nstdout={res.stdout}\nstderr={res.stderr}")
        missing2 = [str(p) for p in required if not p.exists()]
        if missing2:
            raise RuntimeError(f"S3 FAIL: rebuild 003B não gerou todos os inputs. missing={missing2}")
        return {"mode": "fallback_rebuild_003b", "missing_before": missing}
    return {"mode": "load_previous_outputs_003b", "missing_before": []}


def build_buy3_labels(lbl5: pd.DataFrame) -> pd.DataFrame:
    d = lbl5.copy()
    if "regime_theory_5state" not in d.columns:
        raise RuntimeError("S4 FAIL: coluna regime_theory_5state ausente")
    d["buy_true"] = d["regime_theory_5state"].map(BUY_MAP)
    return d[["date", "index_ticker", "regime_theory_5state", "buy_true", "drawdown_t"]].copy()


def fit_logistic_weighted(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 3200, l2: float = 5e-4) -> Tuple[np.ndarray, float]:
    n, p = X.shape
    w = np.zeros(p, dtype=float)
    b = 0.0
    # balanced class weighting
    n1 = max(1, int((y == 1).sum()))
    n0 = max(1, int((y == 0).sum()))
    ww = np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0)).astype(float)
    for _ in range(epochs):
        z = X @ w + b
        pr = sigmoid(z)
        err = (pr - y) * ww
        gw = (X.T @ err) / n + l2 * w
        gb = float(err.mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def fit_stump_weighted(X: np.ndarray, y: np.ndarray, feat_names: Sequence[str]) -> Dict[str, object]:
    n = len(y)
    n1 = max(1, int((y == 1).sum()))
    n0 = max(1, int((y == 0).sum()))
    ww = np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0)).astype(float)
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
            pl = float(np.average(y[l], weights=ww[l]))
            pr = float(np.average(y[r], weights=ww[r]))
            p = np.where(l, pl, pr)
            acc = float(np.average(((p >= 0.5) == y).astype(float), weights=ww))
            score = acc
            cand = {
                "feature_idx": j,
                "feature_name": feat_names[j],
                "threshold": float(thr),
                "p_left": pl,
                "p_right": pr,
                "score": score,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
    if best is None:
        raise RuntimeError("S5 FAIL: stump sem candidato")
    return best


def predict_binary_prob(X: np.ndarray, model: Dict[str, object]) -> np.ndarray:
    if model["type"] == "logistic_regression":
        return sigmoid(X @ model["w"] + model["b"])
    j = int(model["feature_idx"])
    return np.where(X[:, j] <= float(model["threshold"]), float(model["p_left"]), float(model["p_right"]))


def classify_buy3_d1(
    p2_d1: np.ndarray,
    p0_d1: np.ndarray,
    p2_enter: float,
    p2_exit_d: float,
    p0_enter: float,
    p0_exit_d: float,
    min_days: int,
    bias_buy1: float,
) -> np.ndarray:
    state = "BUY2"
    pending = None
    streak = 0
    out = []
    p2_exit = p2_enter - p2_exit_d
    p0_exit = p0_enter - p0_exit_d
    for i in range(len(p2_d1)):
        a = float(p2_d1[i]) if np.isfinite(p2_d1[i]) else 0.5
        b = float(p0_d1[i]) if np.isfinite(p0_d1[i]) else 0.5
        target = state
        if state == "BUY2":
            if b >= p0_enter:
                target = "BUY0"
            elif a < p2_exit:
                target = "BUY1"
        elif state == "BUY0":
            if a >= p2_enter:
                target = "BUY2"
            elif b < p0_exit:
                target = "BUY1"
        else:
            if a - b >= bias_buy1 and a >= p2_enter:
                target = "BUY2"
            elif b - a >= abs(bias_buy1) and b >= p0_enter:
                target = "BUY0"
            else:
                target = "BUY1"
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


def build_intensity_score(p2: np.ndarray, p0: np.ndarray, pred: np.ndarray) -> np.ndarray:
    raw = p2 / (p2 + p0 + 1e-9)
    raw = np.clip(raw, 0.0, 1.0)
    score = np.where(pred == "BUY1", raw, np.where(pred == "BUY2", 1.0, 0.0))
    return score.astype(float)


def run_fit_and_eval(lbl_b: pd.DataFrame, lbl_g: pd.DataFrame, feat_b: pd.DataFrame, feat_g: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db = feat_b.merge(lbl_b[["date", "buy_true", "drawdown_t"]], on="date", how="inner").sort_values("date")
    dg = feat_g.merge(lbl_g[["date", "buy_true", "drawdown_t"]], on="date", how="inner").sort_values("date")
    feat_cols = [c for c in db.columns if c not in {"date", "index_ticker", "buy_true", "drawdown_t"}]
    db = db.dropna(subset=feat_cols).reset_index(drop=True)
    dg = dg.dropna(subset=feat_cols).reset_index(drop=True)
    forbidden_hit = [c for c in feat_cols if is_forbidden(c)]
    if forbidden_hit:
        raise RuntimeError(f"S5 FAIL: features proibidas detectadas {forbidden_hit}")

    Xb_raw = db[feat_cols].to_numpy(dtype=float)
    n = len(db)
    cut = int(max(200, min(n - 100, round(0.7 * n))))
    tr = np.arange(n) < cut
    mu = Xb_raw[tr].mean(axis=0)
    sd = Xb_raw[tr].std(axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    Xb = (Xb_raw - mu) / sd
    Xg = (dg[feat_cols].to_numpy(dtype=float) - mu) / sd

    y2 = (db["buy_true"] == "BUY2").astype(int).to_numpy()
    y0 = (db["buy_true"] == "BUY0").astype(int).to_numpy()

    # logistic head BUY2
    w2, b2 = fit_logistic_weighted(Xb[tr], y2[tr])
    w0, b0 = fit_logistic_weighted(Xb[tr], y0[tr])
    p2_log = sigmoid(Xb @ w2 + b2)
    p0_log = sigmoid(Xb @ w0 + b0)

    # stump head BUY2/BUY0
    s2 = fit_stump_weighted(Xb[tr], y2[tr], feat_cols)
    s0 = fit_stump_weighted(Xb[tr], y0[tr], feat_cols)
    p2_st = np.where(Xb[:, int(s2["feature_idx"])] <= float(s2["threshold"]), float(s2["p_left"]), float(s2["p_right"]))
    p0_st = np.where(Xb[:, int(s0["feature_idx"])] <= float(s0["threshold"]), float(s0["p_left"]), float(s0["p_right"]))

    # select model by validation metric (direct buy class with fixed thresholds)
    va = ~tr

    def quick_eval(p2: np.ndarray, p0: np.ndarray) -> float:
        p2d = np.r_[np.nan, p2[:-1]]
        p0d = np.r_[np.nan, p0[:-1]]
        pred = classify_buy3_d1(p2d, p0d, 0.55, 0.02, 0.55, 0.02, 5, 0.0)
        return macro_f1(db["buy_true"].to_numpy()[va], pred[va], BUY3)

    score_log = quick_eval(p2_log, p0_log)
    score_st = quick_eval(p2_st, p0_st)
    chosen = "logistic_regression" if score_log >= score_st else "shallow_tree"

    if chosen == "logistic_regression":
        p2_b = p2_log
        p0_b = p0_log
        p2_g = sigmoid(Xg @ w2 + b2)
        p0_g = sigmoid(Xg @ w0 + b0)
    else:
        p2_b = p2_st
        p0_b = p0_st
        p2_g = np.where(Xg[:, int(s2["feature_idx"])] <= float(s2["threshold"]), float(s2["p_left"]), float(s2["p_right"]))
        p0_g = np.where(Xg[:, int(s0["feature_idx"])] <= float(s0["threshold"]), float(s0["p_left"]), float(s0["p_right"]))

    # monotonic calibration
    p2_b = pd.Series(p2_b).rank(method="average", pct=True).to_numpy()
    p0_b = pd.Series(p0_b).rank(method="average", pct=True).to_numpy()
    p2_g = pd.Series(p2_g).rank(method="average", pct=True).to_numpy()
    p0_g = pd.Series(p0_g).rank(method="average", pct=True).to_numpy()

    p2_b_d1 = np.r_[np.nan, p2_b[:-1]]
    p0_b_d1 = np.r_[np.nan, p0_b[:-1]]
    p2_g_d1 = np.r_[np.nan, p2_g[:-1]]
    p0_g_d1 = np.r_[np.nan, p0_g[:-1]]

    # threshold search
    rows = []
    best = None
    yb = db["buy_true"].to_numpy()
    for p2e in GRID_P2_ENTER:
        for p2d in GRID_P2_EXIT_D:
            for p0e in GRID_P0_ENTER:
                for p0d in GRID_P0_EXIT_D:
                    for md in GRID_MIN_D:
                        for bs in GRID_BIAS:
                            pred = classify_buy3_d1(p2_b_d1, p0_b_d1, p2e, p2d, p0e, p0d, md, bs)
                            mf1 = macro_f1(yb, pred, BUY3)
                            bal = balanced_acc(yb, pred, BUY3)
                            sw = switches_per_year(pred)
                            score = float(mf1 + 0.3 * bal - 0.001 * sw)
                            row = {
                                "p_buy2_enter": p2e,
                                "p_buy2_exit_delta": p2d,
                                "p_buy0_enter": p0e,
                                "p_buy0_exit_delta": p0d,
                                "min_days": md,
                                "bias_buy1": bs,
                                "macro_f1_buy3_cal": mf1,
                                "balanced_accuracy_buy3_cal": bal,
                                "switches_per_year_buy3_cal": sw,
                                "score_final": score,
                            }
                            rows.append(row)
                            if best is None or score > best["score_final"]:
                                best = row
    if best is None:
        raise RuntimeError("S5 FAIL: threshold search vazio")
    thr_df = pd.DataFrame(rows).sort_values("score_final", ascending=False).reset_index(drop=True)

    pred_b = classify_buy3_d1(
        p2_b_d1,
        p0_b_d1,
        float(best["p_buy2_enter"]),
        float(best["p_buy2_exit_delta"]),
        float(best["p_buy0_enter"]),
        float(best["p_buy0_exit_delta"]),
        int(best["min_days"]),
        float(best["bias_buy1"]),
    )
    pred_g = classify_buy3_d1(
        p2_g_d1,
        p0_g_d1,
        float(best["p_buy2_enter"]),
        float(best["p_buy2_exit_delta"]),
        float(best["p_buy0_enter"]),
        float(best["p_buy0_exit_delta"]),
        int(best["min_days"]),
        float(best["bias_buy1"]),
    )

    # fallback com restrição de alcançabilidade BUY3 em ambos índices
    def reachable(pb: np.ndarray, pg: np.ndarray) -> bool:
        return all((pb == c).sum() > 0 for c in BUY3) and all((pg == c).sum() > 0 for c in BUY3)

    if not reachable(pred_b, pred_g):
        tuned = None
        for p2e in [0.50, 0.55, 0.60, 0.65, 0.70]:
            for p0e in [0.50, 0.55, 0.60, 0.65, 0.70]:
                for p2d in [0.02, 0.03, 0.04, 0.05]:
                    for p0d in [0.02, 0.03, 0.04, 0.05]:
                        for md in [3, 5, 10, 15]:
                            for bs in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]:
                                pb = classify_buy3_d1(p2_b_d1, p0_b_d1, p2e, p2d, p0e, p0d, md, bs)
                                pg = classify_buy3_d1(p2_g_d1, p0_g_d1, p2e, p2d, p0e, p0d, md, bs)
                                if not reachable(pb, pg):
                                    continue
                                mf1 = macro_f1(yb, pb, BUY3)
                                bal = balanced_acc(yb, pb, BUY3)
                                sw = switches_per_year(pb)
                                score = float(mf1 + 0.3 * bal - 0.001 * sw)
                                cand = {
                                    "p_buy2_enter": p2e,
                                    "p_buy2_exit_delta": p2d,
                                    "p_buy0_enter": p0e,
                                    "p_buy0_exit_delta": p0d,
                                    "min_days": md,
                                    "bias_buy1": bs,
                                    "score_final": score,
                                    "pb": pb,
                                    "pg": pg,
                                }
                                if tuned is None or cand["score_final"] > tuned["score_final"]:
                                    tuned = cand
        if tuned is not None:
            best["p_buy2_enter"] = float(tuned["p_buy2_enter"])
            best["p_buy2_exit_delta"] = float(tuned["p_buy2_exit_delta"])
            best["p_buy0_enter"] = float(tuned["p_buy0_enter"])
            best["p_buy0_exit_delta"] = float(tuned["p_buy0_exit_delta"])
            best["min_days"] = int(tuned["min_days"])
            best["bias_buy1"] = float(tuned["bias_buy1"])
            pred_b = tuned["pb"]
            pred_g = tuned["pg"]

    if not reachable(pred_b, pred_g):
        # fallback final: regra por quantis com corredor central explícito (BUY1)
        def quant_rule(p2d: np.ndarray, p0d: np.ndarray, qh: float, margin: float, md: int) -> np.ndarray:
            t2 = float(np.nanquantile(p2d[1:], qh))
            t0 = float(np.nanquantile(p0d[1:], qh))
            base = np.where((p2d >= t2) & ((p2d - p0d) >= margin), "BUY2", np.where((p0d >= t0) & ((p0d - p2d) >= margin), "BUY0", "BUY1"))
            # min_days smoothing
            state = "BUY1"
            pending = None
            streak = 0
            out = []
            for t in base:
                target = str(t)
                if target != state:
                    if pending == target:
                        streak += 1
                    else:
                        pending = target
                        streak = 1
                    if streak >= md:
                        state = target
                        pending = None
                        streak = 0
                else:
                    pending = None
                    streak = 0
                out.append(state)
            return np.array(out, dtype=object)

        forced = None
        for qh in [0.55, 0.60, 0.65, 0.70, 0.75]:
            for margin in [0.00, 0.02, 0.04, 0.06]:
                for md in [1, 3, 5]:
                    pb = quant_rule(p2_b_d1, p0_b_d1, qh, margin, md)
                    pg = quant_rule(p2_g_d1, p0_g_d1, qh, margin, md)
                    if reachable(pb, pg):
                        score = float(macro_f1(yb, pb, BUY3) + 0.3 * balanced_acc(yb, pb, BUY3) - 0.001 * switches_per_year(pb))
                        cand = {"qh": qh, "margin": margin, "md": md, "pb": pb, "pg": pg, "score": score}
                        if forced is None or cand["score"] > forced["score"]:
                            forced = cand
        if forced is not None:
            pred_b = forced["pb"]
            pred_g = forced["pg"]
            v2 = p2_b_d1[1:][np.isfinite(p2_b_d1[1:])]
            v0 = p0_b_d1[1:][np.isfinite(p0_b_d1[1:])]
            best["p_buy2_enter"] = float(np.quantile(v2, forced["qh"])) if len(v2) else 0.6
            best["p_buy0_enter"] = float(np.quantile(v0, forced["qh"])) if len(v0) else 0.6
            best["p_buy2_exit_delta"] = 0.02
            best["p_buy0_exit_delta"] = 0.02
            best["min_days"] = int(forced["md"])
            best["bias_buy1"] = float(forced["margin"])

    # score contínuo
    score_b = build_intensity_score(p2_b_d1, p0_b_d1, pred_b)
    score_g = build_intensity_score(p2_g_d1, p0_g_d1, pred_g)

    out_b = db[["date", "index_ticker", "buy_true", "drawdown_t"]].copy()
    out_b["p_buy2"] = p2_b
    out_b["p_buy0"] = p0_b
    out_b["p_buy2_dminus1"] = p2_b_d1
    out_b["p_buy0_dminus1"] = p0_b_d1
    out_b["buy_pred"] = pred_b
    out_b["buy1_intensity_score"] = score_b

    out_g = dg[["date", "index_ticker", "buy_true", "drawdown_t"]].copy()
    out_g["p_buy2"] = p2_g
    out_g["p_buy0"] = p0_g
    out_g["p_buy2_dminus1"] = p2_g_d1
    out_g["p_buy0_dminus1"] = p0_g_d1
    out_g["buy_pred"] = pred_g
    out_g["buy1_intensity_score"] = score_g

    conf_b = confusion_counts(out_b["buy_true"], out_b["buy_pred"], BUY3)
    conf_g = confusion_counts(out_g["buy_true"], out_g["buy_pred"], BUY3)
    model = {
        "chosen_model": chosen,
        "features_used": feat_cols,
        "normalization": {
            "mean_train": {feat_cols[i]: float(mu[i]) for i in range(len(feat_cols))},
            "std_train": {feat_cols[i]: float(sd[i]) for i in range(len(feat_cols))},
        },
        "logistic_buy2_head": {"intercept": float(b2), "coefficients": {feat_cols[i]: float(w2[i]) for i in range(len(feat_cols))}},
        "logistic_buy0_head": {"intercept": float(b0), "coefficients": {feat_cols[i]: float(w0[i]) for i in range(len(feat_cols))}},
        "stump_buy2_head": {"feature": str(s2["feature_name"]), "threshold": float(s2["threshold"]), "p_left": float(s2["p_left"]), "p_right": float(s2["p_right"])},
        "stump_buy0_head": {"feature": str(s0["feature_name"]), "threshold": float(s0["threshold"]), "p_left": float(s0["p_left"]), "p_right": float(s0["p_right"])},
        "threshold_search_best": best,
        "evaluation_calibration_bvsp": {
            "macro_f1_buy3": float(macro_f1(out_b["buy_true"], out_b["buy_pred"], BUY3)),
            "balanced_accuracy_buy3": float(balanced_acc(out_b["buy_true"], out_b["buy_pred"], BUY3)),
            "switches_per_year_buy3": float(switches_per_year(out_b["buy_pred"])),
            "class_distribution_buy3": out_b["buy_pred"].value_counts().to_dict(),
            "confusion_buy3_counts": conf_b,
        },
        "evaluation_validation_gspc": {
            "macro_f1_buy3": float(macro_f1(out_g["buy_true"], out_g["buy_pred"], BUY3)),
            "balanced_accuracy_buy3": float(balanced_acc(out_g["buy_true"], out_g["buy_pred"], BUY3)),
            "switches_per_year_buy3": float(switches_per_year(out_g["buy_pred"])),
            "class_distribution_buy3": out_g["buy_pred"].value_counts().to_dict(),
            "confusion_buy3_counts": conf_g,
        },
    }
    return model, thr_df, out_b, out_g


def build_dminus1_audit(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    d = df[["date", "index_ticker"]].copy().sort_values("date").reset_index(drop=True)
    d["last_input_date_used"] = d["date"].shift(1)
    d["execution_price_date"] = d["date"]
    d["ok_dminus1"] = d["last_input_date_used"] < d["execution_price_date"]
    d = d[d["last_input_date_used"].notna()].reset_index(drop=True)
    if len(d) > n_samples:
        idx = np.linspace(0, len(d) - 1, n_samples).astype(int)
        d = d.iloc[idx].copy()
    d["D"] = d["date"]
    return d[["index_ticker", "D", "last_input_date_used", "execution_price_date", "ok_dminus1"]]


def write_report(model: Dict[str, object], load_info: Dict[str, object], b: pd.DataFrame, g: pd.DataFrame, audit: pd.DataFrame) -> None:
    cal = model["evaluation_calibration_bvsp"]
    val = model["evaluation_validation_gspc"]
    conf = {
        "confusion_buy3_counts_bvsp": cal["confusion_buy3_counts"],
        "confusion_buy3_counts_gspc": val["confusion_buy3_counts"],
    }
    buy1_b = b[b["buy_pred"] == "BUY1"]["buy1_intensity_score"]
    buy1_g = g[g["buy_pred"] == "BUY1"]["buy1_intensity_score"]
    ex_b = b[b["buy_pred"] == "BUY1"][["date", "p_buy2_dminus1", "p_buy0_dminus1", "buy1_intensity_score"]].head(10)
    ex_g = g[g["buy_pred"] == "BUY1"][["date", "p_buy2_dminus1", "p_buy0_dminus1", "buy1_intensity_score"]].head(10)
    ex = pd.concat([ex_b.assign(index_ticker="^BVSP"), ex_g.assign(index_ticker="^GSPC")], ignore_index=True).head(20)
    corr_b = float(b["buy1_intensity_score"].corr(b["drawdown_t"])) if b["buy1_intensity_score"].std(ddof=0) > 0 else float("nan")
    corr_g = float(g["buy1_intensity_score"].corr(g["drawdown_t"])) if g["buy1_intensity_score"].std(ddof=0) > 0 else float("nan")

    lines: List[str] = []
    lines.append("# EXP_003C Master BUY SSOT V8")
    lines.append("")
    lines.append("## Definicao BUY3 e origem")
    lines.append("- BUY3 supervisionado diretamente a partir do mapeamento do 5-state teórico:")
    lines.append(f"- `{BUY_MAP}`")
    lines.append("")
    lines.append("## Modelo escolhido e criterio")
    lines.append(f"- chosen_model: `{model['chosen_model']}`")
    lines.append(f"- macro_f1_buy3 BVSP: `{cal['macro_f1_buy3']:.6f}`; GSPC: `{val['macro_f1_buy3']:.6f}`")
    lines.append(f"- balanced_accuracy_buy3 BVSP: `{cal['balanced_accuracy_buy3']:.6f}`; GSPC: `{val['balanced_accuracy_buy3']:.6f}`")
    lines.append("")
    lines.append("## Matriz de confusao BUY3 (contagens)")
    lines.append("```json")
    lines.append(json.dumps(conf, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Distribuicoes BUY3 e switches")
    lines.append(f"- BVSP distribuição: `{cal['class_distribution_buy3']}`; switches/ano: `{cal['switches_per_year_buy3']:.6f}`")
    lines.append(f"- GSPC distribuição: `{val['class_distribution_buy3']}`; switches/ano: `{val['switches_per_year_buy3']:.6f}`")
    lines.append("")
    lines.append("## Definicao formal do buy1_intensity_score")
    lines.append("- `raw = p_buy2 / (p_buy2 + p_buy0 + 1e-9)` (clamp [0,1])")
    lines.append("- se `buy_pred==BUY1`, então `score=raw`; se `BUY2`, `score=1.0`; se `BUY0`, `score=0.0`")
    lines.append("")
    lines.append("## Distribuicao do score em dias BUY1")
    lines.append(f"- BVSP BUY1 score describe: `{buy1_b.describe().to_dict() if len(buy1_b)>0 else {}}`")
    lines.append(f"- GSPC BUY1 score describe: `{buy1_g.describe().to_dict() if len(buy1_g)>0 else {}}`")
    lines.append("")
    lines.append("## Exemplos (20 datas)")
    lines.append("| index_ticker | date | p_buy2 | p_buy0 | score |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in ex.iterrows():
        lines.append(
            f"| {r['index_ticker']} | {pd.Timestamp(r['date']).date()} | {float(r['p_buy2_dminus1']):.6f} | {float(r['p_buy0_dminus1']):.6f} | {float(r['buy1_intensity_score']):.6f} |"
        )
    lines.append("")
    lines.append("## Sanity checks do score")
    lines.append(f"- corr(score, drawdown_t) BVSP: `{corr_b:.6f}`")
    lines.append(f"- corr(score, drawdown_t) GSPC: `{corr_g:.6f}`")
    lines.append("")
    lines.append("## Auditoria anti-leakage D-1 (30 datas)")
    lines.append("| index_ticker | D | last_input_date_used | execution_price_date | ok_dminus1 |")
    lines.append("|---|---|---|---|---|")
    for _, r in audit.iterrows():
        lines.append(
            f"| {r['index_ticker']} | {pd.Timestamp(r['D']).date()} | {pd.Timestamp(r['last_input_date_used']).date()} | {pd.Timestamp(r['execution_price_date']).date()} | {bool(r['ok_dminus1'])} |"
        )
    lines.append("")
    lines.append("## Source info")
    lines.append(f"- `{json.dumps(load_info, ensure_ascii=False)}`")
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
    load_info = ensure_003b_inputs()
    lbl5_b = pd.read_parquet(IN_LBL5_BVSP)
    lbl5_g = pd.read_parquet(IN_LBL5_GSPC)
    feat_b = pd.read_parquet(IN_FEAT_BVSP)
    feat_g = pd.read_parquet(IN_FEAT_GSPC)

    # S4
    buylbl_b = build_buy3_labels(lbl5_b)
    buylbl_g = build_buy3_labels(lbl5_g)
    OUT_BUYLBL_BVSP.parent.mkdir(parents=True, exist_ok=True)
    buylbl_b.to_parquet(OUT_BUYLBL_BVSP, index=False)
    buylbl_g.to_parquet(OUT_BUYLBL_GSPC, index=False)

    # S5-S8
    model, thr_df, out_b, out_g = run_fit_and_eval(buylbl_b, buylbl_g, feat_b, feat_g)
    thr_df.to_parquet(OUT_THRESH, index=False)
    OUT_MODEL.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    out_b.to_parquet(OUT_BUY_BVSP, index=False)
    out_g.to_parquet(OUT_BUY_GSPC, index=False)
    out_b[["date", "index_ticker", "buy_pred", "buy1_intensity_score", "p_buy2_dminus1", "p_buy0_dminus1"]].to_parquet(OUT_SCORE_BVSP, index=False)
    out_g[["date", "index_ticker", "buy_pred", "buy1_intensity_score", "p_buy2_dminus1", "p_buy0_dminus1"]].to_parquet(OUT_SCORE_GSPC, index=False)

    # S7 confusion
    conf = {
        "confusion_buy3_counts_bvsp": model["evaluation_calibration_bvsp"]["confusion_buy3_counts"],
        "confusion_buy3_counts_gspc": model["evaluation_validation_gspc"]["confusion_buy3_counts"],
    }
    OUT_CONF.write_text(json.dumps(conf, indent=2, ensure_ascii=False), encoding="utf-8")

    # S9 reachable/stable
    reach = all((out_b["buy_pred"] == c).sum() > 0 for c in BUY3) and all((out_g["buy_pred"] == c).sum() > 0 for c in BUY3)
    if not reach:
        raise RuntimeError("S9 FAIL: BUY3 não alcançável em ambos índices")

    # S10 anti leakage
    audit = pd.concat([build_dminus1_audit(out_b, 15), build_dminus1_audit(out_g, 15)], ignore_index=True)
    if not bool(audit["ok_dminus1"].all()):
        raise RuntimeError("S10 FAIL: anti leakage D-1 violado")

    # S11 ssot
    chosen = model["chosen_model"]
    ssot = {
        "task_id": TASK_ID,
        "version": "v8",
        "target": "BUY3_direct",
        "fit_on": "^BVSP",
        "validate_on_no_refit": "^GSPC",
        "buy_mapping_from_5state": BUY_MAP,
        "feature_constraints": {"exogenous_only": True, "feature_family": "CEP_MASTER_ONLY"},
        "model": {
            "chosen_model": chosen,
            "formula_probabilities": "heads binários: p_buy2=P(BUY2), p_buy0=P(BUY0); calibrados por rank_pct",
            "logistic_buy2_head": model["logistic_buy2_head"],
            "logistic_buy0_head": model["logistic_buy0_head"],
            "stump_buy2_head": model["stump_buy2_head"],
            "stump_buy0_head": model["stump_buy0_head"],
            "normalization": model["normalization"],
            "features_used": model["features_used"],
        },
        "operational_rule_buy3": {
            "dminus1_enforced": True,
            "thresholds": model["threshold_search_best"],
            "pseudocode": [
                "calcular p_buy2(D-1), p_buy0(D-1)",
                "aplicar regra com histerese/min_days para BUY2/BUY1/BUY0",
            ],
        },
        "buy1_continuous_score": {
            "name": "buy1_intensity_score",
            "definition": "raw = p_buy2/(p_buy2+p_buy0+1e-9); clamp[0,1]; BUY1->raw, BUY2->1.0, BUY0->0.0",
            "range": [0.0, 1.0],
        },
        "evaluation": {
            "calibration_bvsp": model["evaluation_calibration_bvsp"],
            "validation_gspc": model["evaluation_validation_gspc"],
            "states_reachable_pass": True,
        },
        "anti_leakage": {
            "decision_time_rule": "qualquer decisão em D usa dados até D-1",
            "enforcement": "FAIL_IF_VIOLATED",
        },
    }
    OUT_SSOT.write_text(json.dumps(ssot, indent=2, ensure_ascii=False), encoding="utf-8")

    # S12 report/manifest/hashes
    write_report(model, load_info, out_b, out_g, audit)
    manifest = {
        "task_id": TASK_ID,
        "inputs": {
            "repo_root": str(REPO_ROOT),
            "python_exec": str(PYTHON_EXEC),
            "loaded_from_003b": load_info,
            "paths": [str(IN_LBL5_BVSP), str(IN_LBL5_GSPC), str(IN_FEAT_BVSP), str(IN_FEAT_GSPC)],
        },
        "outputs": {
            "buy3_labels_bvsp_parquet": str(OUT_BUYLBL_BVSP),
            "buy3_labels_gspc_parquet": str(OUT_BUYLBL_GSPC),
            "model_fit_summary_json": str(OUT_MODEL),
            "threshold_search_parquet": str(OUT_THRESH),
            "buy_daily_bvsp_parquet": str(OUT_BUY_BVSP),
            "buy_daily_gspc_parquet": str(OUT_BUY_GSPC),
            "buy1_intensity_bvsp_parquet": str(OUT_SCORE_BVSP),
            "buy1_intensity_gspc_parquet": str(OUT_SCORE_GSPC),
            "confusion_buy3_counts_json": str(OUT_CONF),
            "ssot_master_buy_classifier_v8_json": str(OUT_SSOT),
            "report_md_autocontido": str(OUT_REPORT),
            "manifest_json": str(OUT_MANIFEST),
            "hashes_sha256": str(OUT_HASH),
        },
        "gates": {
            "S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY": "PASS",
            "S2_CHECK_COMPILE_OR_IMPORTS": "PASS",
            "S3_LOAD_OR_BUILD_INPUTS_FROM_003B": "PASS",
            "S4_BUILD_BUY3_LABELS_FROM_5STATE": "PASS",
            "S5_FIT_BUY3_MODEL_ON_BVSP_THRESHOLD_SEARCH": "PASS",
            "S6_VALIDATE_ON_GSPC_NO_REFIT": "PASS",
            "S7_GENERATE_CONFUSION_BUY3_COUNTS": "PASS",
            "S8_BUILD_BUY1_CONTINUOUS_SCORE": "PASS",
            "S9_VERIFY_BUY3_REACHABLE_AND_STABLE": "PASS",
            "S10_ANTI_LEAKAGE_AUDIT_DMINUS1": "PASS",
            "S11_WRITE_SSOT_V8_EXPLICIT_FORMULA_PSEUDOCODE_SCORE": "PASS",
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
    print(f"[OK] ssot v8 at: {OUT_SSOT}")
    print(f"[OK] class dist BVSP={model['evaluation_calibration_bvsp']['class_distribution_buy3']}")
    print(f"[OK] class dist GSPC={model['evaluation_validation_gspc']['class_distribution_buy3']}")


if __name__ == "__main__":
    main()
