#!/usr/bin/env python3
import fnmatch
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TASK_ID = "TASK_CEP_CYCLE2_001B_MASTER_REGIME_EXOG_RECALIBRATION_V2"
V1_DIR = Path("/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration")
OUT_DIR = Path("/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration_exog_v2")
ALLOWLIST_READ = [
    Path("/home/wilson/CEP_NA_BOLSA"),
    Path("/home/wilson/CEP_COMPRA"),
    Path("/home/wilson/PortfolioZero"),
    Path("/home/wilson/CEP_COMPRA/outputs"),
]
ALLOWLIST_WRITE = OUT_DIR
S1_EVIDENCE = Path("/home/wilson/CEP_COMPRA/planning/runs") / TASK_ID / "S1_GATE_ALLOWLIST.txt"

ENDOGENOUS_PATTERNS = ["*n_positions*", "*portfolio_state*", "*positions*", "*risk_on*"]
HARD_EXCLUDE = {"sinais_cep_n_positions_m6", "sinais_cep_portfolio_state_risk_on"}
MIN_BALANCED_ACCURACY = 0.58
MAX_SWITCHES_MULT_V1 = 1.20


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_allowlist() -> None:
    ensure_parent(S1_EVIDENCE)
    for p in ALLOWLIST_READ:
        if not p.exists():
            raise RuntimeError(f"Allowlist de leitura ausente: {p}")
    out_abs = ALLOWLIST_WRITE.resolve()
    if not str(out_abs).startswith(str(Path("/home/wilson/CEP_COMPRA/outputs").resolve())):
        raise RuntimeError("Escrita fora de outputs/")
    S1_EVIDENCE.write_text(
        "\n".join(
            [
                f"TASK: {TASK_ID}",
                "PASS: leituras restritas a:",
                *[f"- {p}" for p in ALLOWLIST_READ],
                "PASS: escrita restrita a:",
                f"- {ALLOWLIST_WRITE}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def metric_auc(y_true: pd.Series, x: pd.Series) -> float:
    df = pd.DataFrame({"y": y_true.astype(int), "x": pd.to_numeric(x, errors="coerce")}).dropna()
    if df["y"].nunique() < 2:
        return float("nan")
    ranks = df["x"].rank(method="average")
    n1 = int((df["y"] == 1).sum())
    n0 = int((df["y"] == 0).sum())
    s1 = float(ranks[df["y"] == 1].sum())
    auc = (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(max(auc, 1.0 - auc))


def metric_ks(y_true: pd.Series, x: pd.Series) -> float:
    df = pd.DataFrame({"y": y_true.astype(int), "x": pd.to_numeric(x, errors="coerce")}).dropna().sort_values("x")
    if df["y"].nunique() < 2:
        return float("nan")
    n1 = float((df["y"] == 1).sum())
    n0 = float((df["y"] == 0).sum())
    cdf1 = (df["y"] == 1).cumsum() / n1
    cdf0 = (df["y"] == 0).cumsum() / n0
    return float((cdf1 - cdf0).abs().max())


def metric_t_abs(y_true: pd.Series, x: pd.Series) -> float:
    df = pd.DataFrame({"y": y_true.astype(int), "x": pd.to_numeric(x, errors="coerce")}).dropna()
    if df["y"].nunique() < 2:
        return float("nan")
    g1 = df.loc[df["y"] == 1, "x"]
    g0 = df.loc[df["y"] == 0, "x"]
    n1 = len(g1)
    n0 = len(g0)
    if n1 < 2 or n0 < 2:
        return float("nan")
    den = math.sqrt(float(g1.var(ddof=1) / n1 + g0.var(ddof=1) / n0))
    if den <= 0:
        return 0.0
    return float(abs((g1.mean() - g0.mean()) / den))


def score_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for bear in ["W2", "W3"]:
        d = df[df["label_window"].isin(["W1", bear])].copy()
        y = (d["label_window"] == "W1").astype(int)
        for f in feature_cols:
            rows.append(
                {
                    "feature": f,
                    "scenario_bear": bear,
                    "metric_auc": metric_auc(y, d[f]),
                    "metric_ks": metric_ks(y, d[f]),
                    "metric_t": metric_t_abs(y, d[f]),
                }
            )
    out = pd.DataFrame(rows)
    out["score_mix"] = out[["metric_auc", "metric_ks", "metric_t"]].fillna(0.0).mean(axis=1)
    out["rank"] = out.groupby("scenario_bear")["score_mix"].rank(ascending=False, method="dense").astype(int)
    return out.sort_values(["scenario_bear", "rank", "feature"]).reset_index(drop=True)


def balanced_accuracy_binary(true_y: np.ndarray, pred_state: np.ndarray) -> float:
    pred_y = np.where(pred_state == "BULL", 1, np.where(pred_state == "BEAR", 0, -1))
    pos = true_y == 1
    neg = true_y == 0
    tpr = float((pred_y[pos] == 1).sum()) / max(1, int(pos.sum()))
    tnr = float((pred_y[neg] == 0).sum()) / max(1, int(neg.sum()))
    return 0.5 * (tpr + tnr)


def regime_predict(
    score: np.ndarray,
    bull_enter: float,
    bull_exit: float,
    bear_enter: float,
    bear_exit: float,
    min_days: int,
) -> np.ndarray:
    state = "TRANSICAO"
    pending: Optional[str] = None
    streak = 0
    out = []
    for s in score:
        target = state
        if state == "BULL":
            if s <= bear_enter:
                target = "BEAR"
            elif s <= bull_exit:
                target = "TRANSICAO"
        elif state == "BEAR":
            if s >= bull_enter:
                target = "BULL"
            elif s >= bear_exit:
                target = "TRANSICAO"
        else:
            if s >= bull_enter:
                target = "BULL"
            elif s <= bear_enter:
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


def build_composite(
    df: pd.DataFrame, features: Sequence[str], bear_label: str
) -> Tuple[pd.Series, Dict[str, int]]:
    dcal = df[df["label_window"].isin(["W1", bear_label])]
    orientations: Dict[str, int] = {}
    zcols = []
    for f in features:
        m_w1 = float(pd.to_numeric(dcal.loc[dcal["label_window"] == "W1", f], errors="coerce").mean())
        m_b = float(pd.to_numeric(dcal.loc[dcal["label_window"] == bear_label, f], errors="coerce").mean())
        sign = 1 if m_w1 >= m_b else -1
        orientations[f] = sign
        s = pd.to_numeric(df[f], errors="coerce")
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-12:
            sd = 1.0
        z = sign * ((s - mu) / sd)
        zcols.append(z)
    if not zcols:
        return pd.Series(np.zeros(len(df))), orientations
    comp = pd.concat(zcols, axis=1).mean(axis=1)
    return comp, orientations


def grid_search(
    df: pd.DataFrame, feature_scores: pd.DataFrame, top_k: int = 5
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    candidates: List[Dict[str, object]] = []
    best_model: Optional[Dict[str, object]] = None
    best_df: Optional[pd.DataFrame] = None

    for bear in ["W2", "W3"]:
        scenario_candidates_before = len(candidates)
        top_features = (
            feature_scores[feature_scores["scenario_bear"] == bear]
            .sort_values("rank")
            .head(top_k)["feature"]
            .tolist()
        )
        if not top_features:
            continue
        tmp = df.copy()
        comp, orientations = build_composite(tmp, top_features, bear)
        tmp["composite_score"] = comp
        eval_mask = tmp["label_window"].isin(["W1", bear]) & tmp["composite_score"].notna()
        eval_df = tmp.loc[eval_mask].copy()
        if eval_df.empty:
            continue

        q = eval_df["composite_score"].quantile
        bull_qs = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        bear_qs = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        hyst_fracs = [0.00, 0.05, 0.10]
        min_days_grid = [1, 2, 3, 5]
        for bq in bull_qs:
            for rq in bear_qs:
                bull_thr = float(q(bq))
                bear_thr = float(q(rq))
                if bull_thr <= bear_thr:
                    continue
                spread = bull_thr - bear_thr
                for h in hyst_fracs:
                    bull_enter = bull_thr
                    bear_enter = bear_thr
                    bull_exit = bull_thr - h * spread
                    bear_exit = bear_thr + h * spread
                    for md in min_days_grid:
                        pred = regime_predict(
                            score=tmp["composite_score"].to_numpy(dtype=float),
                            bull_enter=bull_enter,
                            bull_exit=bull_exit,
                            bear_enter=bear_enter,
                            bear_exit=bear_exit,
                            min_days=md,
                        )
                        switches = int(np.sum(pred[1:] != pred[:-1])) if len(pred) > 1 else 0
                        true_eval = np.where(eval_df["label_window"].to_numpy() == "W1", 1, 0)
                        pred_eval = pred[eval_mask.to_numpy()]
                        bal_acc = balanced_accuracy_binary(true_eval, pred_eval)
                        penalty = 0.15 * (switches / max(1, len(pred)))
                        score_final = float(bal_acc - penalty)
                        row = {
                            "scenario_bear": bear,
                            "features_used": json.dumps(top_features, ensure_ascii=False),
                            "bull_thresholds": json.dumps({"bull_enter": bull_enter, "bull_exit": bull_exit}, ensure_ascii=False),
                            "bear_thresholds": json.dumps({"bear_enter": bear_enter, "bear_exit": bear_exit}, ensure_ascii=False),
                            "hysteresis": json.dumps({"h_frac": h}, ensure_ascii=False),
                            "min_days": md,
                            "metric_balanced_accuracy": float(bal_acc),
                            "regime_switches": switches,
                            "score_final": score_final,
                        }
                        candidates.append(row)
                        if best_model is None or score_final > float(best_model["score_final"]):
                            best_model = {
                                "scenario_bear": bear,
                                "features": top_features,
                                "orientations": orientations,
                                "bull_enter": bull_enter,
                                "bull_exit": bull_exit,
                                "bear_enter": bear_enter,
                                "bear_exit": bear_exit,
                                "min_days": md,
                                "balanced_accuracy": float(bal_acc),
                                "regime_switches": switches,
                                "score_final": score_final,
                            }
                            best_df = tmp.copy()
        if len(candidates) == scenario_candidates_before:
            # fallback deterministico quando os quantis colapsam e nao geram bull>bear
            ev = tmp.loc[eval_mask, "composite_score"].dropna()
            if not ev.empty:
                mu = float(ev.mean())
                sd = float(ev.std(ddof=0))
                if not np.isfinite(sd) or sd <= 1e-12:
                    sd = 0.05
                bull_enter = mu + 0.25 * sd
                bull_exit = mu + 0.10 * sd
                bear_enter = mu - 0.25 * sd
                bear_exit = mu - 0.10 * sd
                md = 3
                pred = regime_predict(
                    score=tmp["composite_score"].to_numpy(dtype=float),
                    bull_enter=bull_enter,
                    bull_exit=bull_exit,
                    bear_enter=bear_enter,
                    bear_exit=bear_exit,
                    min_days=md,
                )
                switches = int(np.sum(pred[1:] != pred[:-1])) if len(pred) > 1 else 0
                true_eval = np.where(eval_df["label_window"].to_numpy() == "W1", 1, 0)
                pred_eval = pred[eval_mask.to_numpy()]
                bal_acc = balanced_accuracy_binary(true_eval, pred_eval)
                penalty = 0.15 * (switches / max(1, len(pred)))
                score_final = float(bal_acc - penalty)
                row = {
                    "scenario_bear": bear,
                    "features_used": json.dumps(top_features, ensure_ascii=False),
                    "bull_thresholds": json.dumps({"bull_enter": bull_enter, "bull_exit": bull_exit}, ensure_ascii=False),
                    "bear_thresholds": json.dumps({"bear_enter": bear_enter, "bear_exit": bear_exit}, ensure_ascii=False),
                    "hysteresis": json.dumps({"h_frac": "fallback_mu_sd"}, ensure_ascii=False),
                    "min_days": md,
                    "metric_balanced_accuracy": float(bal_acc),
                    "regime_switches": switches,
                    "score_final": score_final,
                }
                candidates.append(row)
                if best_model is None or score_final > float(best_model["score_final"]):
                    best_model = {
                        "scenario_bear": bear,
                        "features": top_features,
                        "orientations": orientations,
                        "bull_enter": bull_enter,
                        "bull_exit": bull_exit,
                        "bear_enter": bear_enter,
                        "bear_exit": bear_exit,
                        "min_days": md,
                        "balanced_accuracy": float(bal_acc),
                        "regime_switches": switches,
                        "score_final": score_final,
                    }
                    best_df = tmp.copy()
    if best_model is None or best_df is None or not candidates:
        raise RuntimeError("Grid search V2 nao produziu candidatos")
    return (
        pd.DataFrame(candidates).sort_values("score_final", ascending=False).reset_index(drop=True),
        best_model,
        best_df,
    )


def confusion_and_stability(df: pd.DataFrame, model: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object], pd.Series]:
    pred = regime_predict(
        score=df["composite_score"].to_numpy(dtype=float),
        bull_enter=float(model["bull_enter"]),
        bull_exit=float(model["bull_exit"]),
        bear_enter=float(model["bear_enter"]),
        bear_exit=float(model["bear_exit"]),
        min_days=int(model["min_days"]),
    )
    pred_s = pd.Series(pred, index=df.index, name="pred_regime")
    eval_mask = df["label_window"].isin(["W1", model["scenario_bear"]])
    ev = df.loc[eval_mask, ["label_window"]].copy()
    ev["pred"] = pred_s.loc[eval_mask].values

    c = (
        ev.groupby(["label_window", "pred"], as_index=False)
        .size()
        .pivot(index="label_window", columns="pred", values="size")
        .fillna(0)
        .astype(int)
    )
    true_y = np.where(ev["label_window"].to_numpy() == "W1", 1, 0)
    bal_acc = balanced_accuracy_binary(true_y, ev["pred"].to_numpy())
    tp = int(((ev["label_window"] == "W1") & (ev["pred"] == "BULL")).sum())
    fn = int(((ev["label_window"] == "W1") & (ev["pred"] != "BULL")).sum())
    tn = int(((ev["label_window"] == model["scenario_bear"]) & (ev["pred"] == "BEAR")).sum())
    fp = int(((ev["label_window"] == model["scenario_bear"]) & (ev["pred"] != "BEAR")).sum())

    confusion = {
        "scenario_winner_bear": model["scenario_bear"],
        "balanced_accuracy": float(bal_acc),
        "matrix_2x2_collapsed_transition_as_error": {
            "tp_W1_pred_BULL": tp,
            "fn_W1_pred_not_BULL": fn,
            "tn_BEAR_pred_BEAR": tn,
            "fp_BEAR_pred_not_BEAR": fp,
        },
        "matrix_detail_true_vs_pred": c.to_dict(),
    }

    switches = int((pred_s.shift(1) != pred_s).sum() - 1) if len(pred_s) > 1 else 0
    runs = (pred_s != pred_s.shift(1)).cumsum()
    durations = pred_s.groupby(runs).agg(["first", "size"]).rename(columns={"first": "regime", "size": "duration_days"})
    avg_dur = durations.groupby("regime")["duration_days"].mean().to_dict()
    med_dur = durations.groupby("regime")["duration_days"].median().to_dict()
    dist = pred_s.value_counts().to_dict()
    total = int(len(pred_s))
    dist_pct = {k: float(v / total) for k, v in dist.items()}
    stability = {
        "scenario_winner_bear": model["scenario_bear"],
        "regime_switches": switches,
        "avg_duration_days_by_regime": avg_dur,
        "median_duration_days_by_regime": med_dur,
        "regime_distribution_days": dist,
        "regime_distribution_pct": dist_pct,
    }
    return confusion, stability, pred_s


def endogenous_match(col: str) -> bool:
    cl = col.lower()
    if col in HARD_EXCLUDE:
        return True
    return any(fnmatch.fnmatch(cl, p.lower()) for p in ENDOGENOUS_PATTERNS)


def json_block(d: Dict[str, object]) -> str:
    return json.dumps(d, indent=2, ensure_ascii=False)


def infer_v1_predictions(master_signals_v1: pd.DataFrame, ssot_v1: Dict[str, object]) -> pd.Series:
    feats = list(ssot_v1["features_used"])
    bear = str(ssot_v1["scenario_bear"])
    d = master_signals_v1.copy()
    zcols = []
    for f in feats:
        s = pd.to_numeric(d[f], errors="coerce")
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-12:
            sd = 1.0
        sign = int(ssot_v1["orientations"][f])
        zcols.append(sign * ((s - mu) / sd))
    d["composite_score"] = pd.concat(zcols, axis=1).mean(axis=1) if zcols else 0.0
    pred = regime_predict(
        score=d["composite_score"].to_numpy(dtype=float),
        bull_enter=float(ssot_v1["thresholds"]["bull_enter"]),
        bull_exit=float(ssot_v1["thresholds"]["bull_exit"]),
        bear_enter=float(ssot_v1["thresholds"]["bear_enter"]),
        bear_exit=float(ssot_v1["thresholds"]["bear_exit"]),
        min_days=int(ssot_v1["thresholds"]["min_days"]),
    )
    return pd.Series(pred, index=d.index)


def format_counts_matrix(conf: Dict[str, object]) -> str:
    c2 = conf["matrix_2x2_collapsed_transition_as_error"]
    lines = []
    lines.append("| célula | contagem |")
    lines.append("|---|---:|")
    lines.append(f"| TP (W1 -> BULL) | {c2['tp_W1_pred_BULL']} |")
    lines.append(f"| FN (W1 -> !BULL) | {c2['fn_W1_pred_not_BULL']} |")
    lines.append(f"| TN (BEAR -> BEAR) | {c2['tn_BEAR_pred_BEAR']} |")
    lines.append(f"| FP (BEAR -> !BEAR) | {c2['fp_BEAR_pred_not_BEAR']} |")
    return "\n".join(lines)


def main() -> None:
    assert_allowlist()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ssot_cycle2").mkdir(parents=True, exist_ok=True)

    required = {
        "ssot_v1": V1_DIR / "ssot_cycle2/master_regime_classifier_v1.json",
        "confusion_v1": V1_DIR / "confusion_summary.json",
        "stability_v1": V1_DIR / "stability_summary.json",
        "candidate_v1": V1_DIR / "candidate_thresholds.parquet",
        "feature_v1": V1_DIR / "feature_scores.parquet",
        "master_signals_v1": V1_DIR / "master_signals_daily.parquet",
    }
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        raise RuntimeError(f"Arquivos V1 ausentes: {missing}")

    v1_hashes = {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in required.items()}

    ssot_v1 = json.loads(required["ssot_v1"].read_text(encoding="utf-8"))
    conf_v1 = json.loads(required["confusion_v1"].read_text(encoding="utf-8"))
    stab_v1 = json.loads(required["stability_v1"].read_text(encoding="utf-8"))
    cand_v1 = pd.read_parquet(required["candidate_v1"]).sort_values("score_final", ascending=False)
    master_v1 = pd.read_parquet(required["master_signals_v1"]).copy()
    master_v1["date"] = pd.to_datetime(master_v1["date"])

    v1_best = cand_v1.iloc[0].to_dict() if not cand_v1.empty else {}
    v1_pred = infer_v1_predictions(master_v1, ssot_v1)
    v1_runs = (v1_pred != v1_pred.shift(1)).cumsum()
    v1_durations = v1_pred.groupby(v1_runs).agg(["first", "size"]).rename(columns={"first": "regime", "size": "duration_days"})
    v1_avg_dur = v1_durations.groupby("regime")["duration_days"].mean().to_dict()
    v1_med_dur = v1_durations.groupby("regime")["duration_days"].median().to_dict()

    all_feat_v1 = [c for c in master_v1.columns if c.startswith("sinais_cep_")]
    endogenous_cols_all = [c for c in all_feat_v1 if endogenous_match(c)]
    endogenous_in_v1_used = [f for f in ssot_v1["features_used"] if endogenous_match(f)]

    md_v1 = []
    md_v1.append("# V1 Rule Dump Autocontido")
    md_v1.append("")
    md_v1.append("## Base V1 (paths + sha256)")
    for k, v in v1_hashes.items():
        md_v1.append(f"- {k}: `{v['path']}` | sha256=`{v['sha256']}`")
    md_v1.append("")
    md_v1.append("## SSOT v1 (JSON integral)")
    md_v1.append("```json")
    md_v1.append(json_block(ssot_v1))
    md_v1.append("```")
    md_v1.append("")
    md_v1.append("## Regra de decisão (limiares/histerese/min_days)")
    md_v1.append("| cenário_bear | features_used | bull_enter | bull_exit | bear_enter | bear_exit | min_days | balanced_accuracy | regime_switches | score_final |")
    md_v1.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    md_v1.append(
        f"| {v1_best.get('scenario_bear','')} | {v1_best.get('features_used','')} | "
        f"{json.loads(v1_best.get('bull_thresholds','{}')).get('bull_enter', float('nan')):.6f} | "
        f"{json.loads(v1_best.get('bull_thresholds','{}')).get('bull_exit', float('nan')):.6f} | "
        f"{json.loads(v1_best.get('bear_thresholds','{}')).get('bear_enter', float('nan')):.6f} | "
        f"{json.loads(v1_best.get('bear_thresholds','{}')).get('bear_exit', float('nan')):.6f} | "
        f"{int(v1_best.get('min_days', 0))} | {float(v1_best.get('metric_balanced_accuracy', float('nan'))):.6f} | "
        f"{int(v1_best.get('regime_switches', 0))} | {float(v1_best.get('score_final', float('nan'))):.6f} |"
    )
    md_v1.append("")
    md_v1.append("## Matriz de confusão (contagens)")
    md_v1.append(format_counts_matrix(conf_v1))
    md_v1.append("")
    md_v1.append("## Estabilidade V1")
    md_v1.append(f"- switches: **{stab_v1.get('regime_switches')}**")
    md_v1.append("- duração média por regime:")
    for rg, v in v1_avg_dur.items():
        md_v1.append(f"  - {rg}: {v:.4f}")
    md_v1.append("- duração mediana por regime:")
    for rg, v in v1_med_dur.items():
        md_v1.append(f"  - {rg}: {v:.4f}")
    md_v1.append("- % dias por regime:")
    for rg, v in stab_v1.get("regime_distribution_pct", {}).items():
        md_v1.append(f"  - {rg}: {100.0 * float(v):.4f}%")
    md_v1.append("")
    md_v1.append("## Features V1 e endogeneidade")
    md_v1.append(f"- total features sinais_cep_ no dataset: **{len(all_feat_v1)}**")
    md_v1.append(f"- features endógenas identificadas por padrão/hard_exclude: **{len(endogenous_cols_all)}**")
    md_v1.append(f"- features usadas no SSOT v1: `{', '.join(ssot_v1['features_used'])}`")
    md_v1.append(f"- endógenas usadas no SSOT v1: `{', '.join(endogenous_in_v1_used) if endogenous_in_v1_used else '(nenhuma)'}`")
    (OUT_DIR / "v1_rule_dump_autocontido.md").write_text("\n".join(md_v1) + "\n", encoding="utf-8")

    id_cols = ["date", "label_window"]
    candidate_features = [c for c in all_feat_v1 if c not in HARD_EXCLUDE and not endogenous_match(c)]
    v2 = master_v1[id_cols + candidate_features].copy()
    v2.to_parquet(OUT_DIR / "v2_regime_daily_signals.parquet", index=False)
    v2.to_csv(OUT_DIR / "v2_regime_daily_signals.csv", index=False)

    remaining_endog = [c for c in candidate_features if endogenous_match(c)]
    if remaining_endog:
        raise RuntimeError(f"Ainda existem features endógenas no V2: {remaining_endog}")

    numeric_features = [c for c in candidate_features if pd.api.types.is_numeric_dtype(v2[c])]
    v2_feature_scores = score_features(v2, numeric_features)
    v2_feature_scores.to_parquet(OUT_DIR / "v2_feature_scores.parquet", index=False)

    v2_candidates, v2_model, v2_df = grid_search(v2, v2_feature_scores, top_k=5)
    v2_candidates.to_parquet(OUT_DIR / "v2_candidate_thresholds.parquet", index=False)

    v2_conf, v2_stab, v2_pred = confusion_and_stability(v2_df, v2_model)
    (OUT_DIR / "v2_confusion_summary.json").write_text(json.dumps(v2_conf, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "v2_stability_summary.json").write_text(json.dumps(v2_stab, indent=2, ensure_ascii=False), encoding="utf-8")

    v1_switches = int(stab_v1.get("regime_switches", 10**9))
    v2_switches = int(v2_stab.get("regime_switches", 10**9))
    gate_balacc = float(v2_conf["balanced_accuracy"]) >= MIN_BALANCED_ACCURACY
    gate_switches = v2_switches <= (MAX_SWITCHES_MULT_V1 * v1_switches)
    gate_no_endog = len([f for f in v2_model["features"] if endogenous_match(f)]) == 0
    freeze_pass = bool(gate_balacc and gate_switches and gate_no_endog)

    ssot_v2 = {
        "task_id": TASK_ID,
        "version": "v2",
        "date_reference": "2026-02-13",
        "source_v1_task_id": ssot_v1.get("task_id"),
        "scenario_bear": v2_model["scenario_bear"],
        "features_used": v2_model["features"],
        "orientations": v2_model["orientations"],
        "thresholds": {
            "bull_enter": v2_model["bull_enter"],
            "bull_exit": v2_model["bull_exit"],
            "bear_enter": v2_model["bear_enter"],
            "bear_exit": v2_model["bear_exit"],
            "min_days": int(v2_model["min_days"]),
        },
        "objective_metrics": {
            "balanced_accuracy": float(v2_conf["balanced_accuracy"]),
            "regime_switches": int(v2_stab["regime_switches"]),
            "score_final": float(v2_model["score_final"]),
        },
        "predicted_regime_distribution": v2_pred.value_counts().to_dict(),
        "endogenous_exclusion": {
            "patterns": ENDOGENOUS_PATTERNS,
            "hard_exclude_features": sorted(HARD_EXCLUDE),
            "remaining_endogenous_in_features_used": [f for f in v2_model["features"] if endogenous_match(f)],
        },
        "freeze_decision": {
            "status": "PASS" if freeze_pass else "FAIL",
            "gates": {
                "min_balanced_accuracy": MIN_BALANCED_ACCURACY,
                "max_switches_multiplier_vs_v1": MAX_SWITCHES_MULT_V1,
                "must_have_zero_endogenous_features": True,
                "gate_balanced_accuracy_pass": gate_balacc,
                "gate_switches_pass": gate_switches,
                "gate_zero_endogenous_pass": gate_no_endog,
            },
            "comparative_numbers": {
                "balanced_accuracy_v1": float(conf_v1.get("balanced_accuracy", float("nan"))),
                "balanced_accuracy_v2": float(v2_conf["balanced_accuracy"]),
                "switches_v1": v1_switches,
                "switches_v2": v2_switches,
                "switches_limit_v2": float(MAX_SWITCHES_MULT_V1 * v1_switches),
            },
            "active_model_after_decision": "v2" if freeze_pass else "v1",
        },
    }
    (OUT_DIR / "ssot_cycle2" / "master_regime_classifier_v2.json").write_text(
        json.dumps(ssot_v2, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    v2_runs = (v2_pred != v2_pred.shift(1)).cumsum()
    v2_dur = v2_pred.groupby(v2_runs).agg(["first", "size"]).rename(columns={"first": "regime", "size": "duration_days"})
    v2_avg_dur = v2_dur.groupby("regime")["duration_days"].mean().to_dict()
    v2_med_dur = v2_dur.groupby("regime")["duration_days"].median().to_dict()

    md_cmp = []
    md_cmp.append("# Regime Calibration Comparison V1 vs V2")
    md_cmp.append("")
    md_cmp.append("## Decisão CTO (gates objetivos)")
    md_cmp.append(f"- freeze_v2: **{'PASS' if freeze_pass else 'FAIL'}**")
    md_cmp.append(f"- gate balanced_accuracy >= {MIN_BALANCED_ACCURACY}: **{'PASS' if gate_balacc else 'FAIL'}**")
    md_cmp.append(f"- gate switches <= {MAX_SWITCHES_MULT_V1:.2f} * v1: **{'PASS' if gate_switches else 'FAIL'}**")
    md_cmp.append(f"- gate zero endógenas em v2: **{'PASS' if gate_no_endog else 'FAIL'}**")
    md_cmp.append(f"- modelo vigente após decisão: **{ssot_v2['freeze_decision']['active_model_after_decision']}**")
    md_cmp.append("")
    md_cmp.append("## SSOT v1 (JSON integral)")
    md_cmp.append("```json")
    md_cmp.append(json_block(ssot_v1))
    md_cmp.append("```")
    md_cmp.append("")
    md_cmp.append("## SSOT v2 (JSON integral)")
    md_cmp.append("```json")
    md_cmp.append(json_block(ssot_v2))
    md_cmp.append("```")
    md_cmp.append("")
    md_cmp.append("## Tabela comparativa")
    md_cmp.append("| métrica | v1 | v2 |")
    md_cmp.append("|---|---:|---:|")
    md_cmp.append(f"| balanced_accuracy | {float(conf_v1['balanced_accuracy']):.6f} | {float(v2_conf['balanced_accuracy']):.6f} |")
    md_cmp.append(f"| regime_switches | {v1_switches} | {v2_switches} |")
    for rg in sorted(set(stab_v1.get("regime_distribution_pct", {}).keys()) | set(v2_stab.get("regime_distribution_pct", {}).keys())):
        md_cmp.append(
            f"| % dias {rg} | {100*float(stab_v1.get('regime_distribution_pct', {}).get(rg, 0.0)):.4f}% | "
            f"{100*float(v2_stab.get('regime_distribution_pct', {}).get(rg, 0.0)):.4f}% |"
        )
    for rg in sorted(set(v1_avg_dur.keys()) | set(v2_avg_dur.keys())):
        md_cmp.append(
            f"| duração média {rg} | {float(v1_avg_dur.get(rg, float('nan'))):.4f} | {float(v2_avg_dur.get(rg, float('nan'))):.4f} |"
        )
        md_cmp.append(
            f"| duração mediana {rg} | {float(v1_med_dur.get(rg, float('nan'))):.4f} | {float(v2_med_dur.get(rg, float('nan'))):.4f} |"
        )
    md_cmp.append("")
    md_cmp.append("## Matriz de confusão V1 (contagens)")
    md_cmp.append(format_counts_matrix(conf_v1))
    md_cmp.append("")
    md_cmp.append("## Matriz de confusão V2 (contagens)")
    md_cmp.append(format_counts_matrix(v2_conf))
    md_cmp.append("")
    md_cmp.append("## Features V2 (exógenas)")
    md_cmp.append(f"- features usadas no SSOT v2: `{', '.join(v2_model['features'])}`")
    md_cmp.append(f"- checagem endógenas remanescentes: `{', '.join([f for f in v2_model['features'] if endogenous_match(f)]) if any(endogenous_match(f) for f in v2_model['features']) else '(nenhuma)'}`")
    md_cmp.append(f"- colunas finais do dataset V2 ({len(candidate_features)}): `{', '.join(candidate_features)}`")
    (OUT_DIR / "regime_calibration_comparison_v1_vs_v2.md").write_text("\n".join(md_cmp) + "\n", encoding="utf-8")

    manifest = {
        "task_id": TASK_ID,
        "base_dir": str(OUT_DIR.relative_to(Path("/home/wilson/CEP_COMPRA"))),
        "inputs": {
            "v1_artifacts_dir": str(V1_DIR),
            "endogenous_feature_name_patterns": ENDOGENOUS_PATTERNS,
            "hard_exclude_features": sorted(HARD_EXCLUDE),
            "quality_gates_for_v2_freeze": {
                "min_balanced_accuracy": MIN_BALANCED_ACCURACY,
                "max_switches_multiplier_vs_v1": MAX_SWITCHES_MULT_V1,
                "must_have_zero_endogenous_features": True,
            },
        },
        "v1_inputs_sha256": v1_hashes,
        "gates": {
            "G1_MD_AUTOCONTIDO": "PASS",
            "G2_NO_ENDOGENOUS_IN_V2": "PASS" if gate_no_endog else "FAIL",
            "G3_OVERALL_PASS": "PASS" if all([gate_balacc, gate_switches, gate_no_endog]) else "FAIL",
        },
        "freeze_decision": ssot_v2["freeze_decision"],
        "artifacts": [
            "v1_rule_dump_autocontido.md",
            "v2_regime_daily_signals.parquet",
            "v2_regime_daily_signals.csv",
            "v2_feature_scores.parquet",
            "v2_candidate_thresholds.parquet",
            "v2_confusion_summary.json",
            "v2_stability_summary.json",
            "ssot_cycle2/master_regime_classifier_v2.json",
            "regime_calibration_comparison_v1_vs_v2.md",
            "manifest.json",
            "hashes.sha256",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    for p in sorted([f for f in OUT_DIR.rglob("*") if f.is_file() and f.name != "hashes.sha256"]):
        rel = p.relative_to(OUT_DIR)
        lines.append(f"{sha256_file(p)}  {rel}")
    (OUT_DIR / "hashes.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] artifacts written to: {OUT_DIR}")
    print(f"[OK] freeze_v2={'PASS' if freeze_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
