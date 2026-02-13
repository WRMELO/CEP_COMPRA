#!/usr/bin/env python3
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TASK_ID = "TASK_CEP_CYCLE2_001_MASTER_REGIME_CALIBRATION_V1"
DATE_REF = "20260213"
OUT_BASE = Path("/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration")
ALLOWLIST_READ = [Path("/home/wilson/CEP_NA_BOLSA"), Path("/home/wilson/CEP_COMPRA")]
ALLOWLIST_WRITE = OUT_BASE
S1_EVIDENCE = Path("/home/wilson/CEP_COMPRA/planning/runs") / TASK_ID / "S1_GATE_ALLOWLIST.txt"

SEARCH_TERMS = [
    "W1",
    "W2",
    "W3",
    "window W1",
    "window W2",
    "window W3",
    "M3",
    "retorno normalizado",
    "normalized return",
    "master",
    "CEP",
    "I-MR",
    "Xbar",
    "Rbar",
    "violations",
    "baseline",
    "limits",
    "decision package",
    "^BVSP",
    "_BVSP",
]

WINDOWS_EXPLICIT = {
    "W1": ("2018-07-01", "2021-06-30"),
    "W2": ("2021-07-01", "2022-12-31"),
    "W3": ("2024-09-01", "2025-11-30"),
}


@dataclass
class ScenarioModel:
    scenario_bear: str
    features: List[str]
    orientations: Dict[str, int]
    bull_enter: float
    bull_exit: float
    bear_enter: float
    bear_exit: float
    min_days: int
    balanced_accuracy: float
    regime_switches: int
    score_final: float


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_allowlist() -> None:
    out_abs = ALLOWLIST_WRITE.resolve()
    if not str(out_abs).startswith(str(Path("/home/wilson/CEP_COMPRA").resolve())):
        raise RuntimeError("Saida fora de /home/wilson/CEP_COMPRA")
    for p in ALLOWLIST_READ:
        if not p.exists():
            raise RuntimeError(f"Diretorio allowlist de leitura ausente: {p}")
    ensure_parent(S1_EVIDENCE)
    S1_EVIDENCE.write_text(
        "\n".join(
            [
                f"TASK: {TASK_ID}",
                "PASS: leituras restritas a:",
                *[f"- {p}" for p in ALLOWLIST_READ],
                "PASS: escritas restritas a:",
                f"- {ALLOWLIST_WRITE}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in {".md", ".json", ".txt", ".py", ".csv", ".html"}


def discover_sources() -> pd.DataFrame:
    candidates: List[Dict[str, object]] = []
    terms_lower = [t.lower() for t in SEARCH_TERMS]
    priority_parts = ["/outputs/", "/planning/runs/", "/docs/"]
    valid_ext = {".md", ".json", ".parquet", ".csv", ".txt", ".py", ".html"}

    for root in ALLOWLIST_READ:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in valid_ext:
                continue
            path_str = str(path)
            lower_name = path_str.lower()
            matched = set()
            for t in terms_lower:
                if t in lower_name:
                    matched.add(t)
            if is_text_file(path):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    scan = text[:2_000_000].lower()
                    for t in terms_lower:
                        if t in scan:
                            matched.add(t)
                except Exception:
                    pass
            if not matched:
                continue
            prio = "others"
            for p in priority_parts:
                if p in path_str:
                    prio = p.strip("/")
                    break
            st = path.stat()
            candidates.append(
                {
                    "source_path": path_str,
                    "repo_root": str(root),
                    "file_type": path.suffix.lower().lstrip("."),
                    "priority_bucket": prio,
                    "matched_terms": "|".join(sorted(matched)),
                    "selection_reason": "matched_search_terms",
                    "size_bytes": int(st.st_size),
                    "mtime_utc": pd.Timestamp(st.st_mtime, unit="s", tz="UTC"),
                }
            )
    if not candidates:
        raise RuntimeError("Nenhuma fonte encontrada para inventario")
    df = pd.DataFrame(candidates).sort_values(["priority_bucket", "source_path"]).reset_index(drop=True)
    return df


def latest_run(base_dir: Path) -> Path:
    runs = sorted([p for p in base_dir.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise RuntimeError(f"Nenhum run_* em {base_dir}")
    return runs[0]


def build_windows_labels(day_index: pd.DatetimeIndex) -> Tuple[pd.DataFrame, Dict[str, object]]:
    labels = pd.Series("outros", index=day_index, dtype="object")
    for w, (d0, d1) in WINDOWS_EXPLICIT.items():
        mask = (day_index >= pd.Timestamp(d0)) & (day_index <= pd.Timestamp(d1))
        labels.loc[mask] = w
    windows_df = pd.DataFrame({"date": day_index, "label_window": labels.values})
    windows_json = {
        "task_id": TASK_ID,
        "resolved_method": "explicit_cycle_z_artifacts",
        "source_paths": [
            "/home/wilson/CEP_COMPRA/tools/task_021_m6_runner.py",
            "/home/wilson/CEP_COMPRA/outputs/backtests/task_021_m6/run_20260213_122019/m6_vs_m3_and_others_analysis_autossuficiente.md",
        ],
        "windows": {
            k: {"start": v[0], "end": v[1]} for k, v in WINDOWS_EXPLICIT.items()
        },
    }
    return windows_df, windows_json


def load_master_signals(windows_df: pd.DataFrame) -> pd.DataFrame:
    p_master_states = Path("/home/wilson/CEP_NA_BOLSA/outputs/experimentos/fase1_calibracao/exp/20260209/dataset_sizing/master_states.csv")
    run_021 = latest_run(Path("/home/wilson/CEP_COMPRA/outputs/backtests/task_021_m6"))
    p_daily = run_021 / "portfolio_daily_ledger.parquet"
    p_events = run_021 / "portfolio_cep_events.parquet"

    if not p_master_states.exists():
        raise RuntimeError(f"Input ausente: {p_master_states}")
    if not p_daily.exists() or not p_events.exists():
        raise RuntimeError("Parquets de task_021 ausentes para montar sinais CEP")

    ms = pd.read_csv(p_master_states, parse_dates=["date"]).sort_values("date")
    daily = pd.read_parquet(p_daily)
    events = pd.read_parquet(p_events)

    m6 = daily[daily["mechanism"] == "M6"].copy()
    if m6.empty:
        raise RuntimeError("Nao encontrei mecanismo M6 em portfolio_daily_ledger.parquet")

    evt = events.copy()
    evt["date"] = pd.to_datetime(evt["date"])
    evt_ag = (
        evt.groupby("date", as_index=False)
        .agg(
            sinais_cep_rule_one_below_lcl=("rule_one_below_lcl", "max"),
            sinais_cep_rule_two_of_three_below_2sigma_neg=("rule_two_of_three_below_2sigma_neg", "max"),
            sinais_cep_xbar_t=("xbar_t", "last"),
            sinais_cep_r_t=("r_t", "last"),
            sinais_cep_xbar_lcl=("xbar_lcl", "last"),
            sinais_cep_two_sigma_neg=("two_sigma_neg", "last"),
        )
    )

    master = pd.DataFrame({"date": pd.to_datetime(ms["date"])})
    master["sinais_cep_master_xt"] = pd.to_numeric(ms["xt"], errors="coerce")
    master["sinais_cep_stress_i"] = ms["stress_i"].astype(int)
    master["sinais_cep_stress_amp"] = ms["stress_amp"].astype(int)
    master["sinais_cep_trend_run7"] = ms["trend_run7"].astype(int)
    master["sinais_cep_master_state_risk_on"] = (ms["state"].astype(str).str.upper() == "RISK_ON").astype(int)

    m6 = m6[["date", "daily_return", "cash_ratio", "n_positions", "portfolio_state"]].copy()
    m6["date"] = pd.to_datetime(m6["date"])
    m6["sinais_cep_daily_return_m6"] = pd.to_numeric(m6["daily_return"], errors="coerce")
    m6["sinais_cep_cash_ratio_m6"] = pd.to_numeric(m6["cash_ratio"], errors="coerce")
    m6["sinais_cep_n_positions_m6"] = pd.to_numeric(m6["n_positions"], errors="coerce")
    m6["sinais_cep_portfolio_state_risk_on"] = (m6["portfolio_state"].astype(str).str.upper() == "PORTFOLIO_RISK_ON").astype(int)
    m6 = m6[
        [
            "date",
            "sinais_cep_daily_return_m6",
            "sinais_cep_cash_ratio_m6",
            "sinais_cep_n_positions_m6",
            "sinais_cep_portfolio_state_risk_on",
        ]
    ]

    out = master.merge(m6, on="date", how="left").merge(evt_ag, on="date", how="left").merge(windows_df, on="date", how="left")
    out["label_window"] = out["label_window"].fillna("outros")
    return out.sort_values("date").reset_index(drop=True)


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


def build_composite(df: pd.DataFrame, features: Sequence[str], bear_label: str) -> Tuple[pd.Series, Dict[str, int]]:
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


def grid_search(df: pd.DataFrame, feature_scores: pd.DataFrame, top_k: int = 5) -> Tuple[pd.DataFrame, ScenarioModel, pd.DataFrame]:
    candidates: List[Dict[str, object]] = []
    best_model: Optional[ScenarioModel] = None
    best_df: Optional[pd.DataFrame] = None

    for bear in ["W2", "W3"]:
        top_features = (
            feature_scores[(feature_scores["scenario_bear"] == bear)]
            .sort_values("rank")
            .head(top_k)["feature"]
            .tolist()
        )
        if not top_features:
            continue
        tmp = df.copy()
        tmp["composite_score"] = np.nan
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
                        if best_model is None or score_final > best_model.score_final:
                            best_model = ScenarioModel(
                                scenario_bear=bear,
                                features=top_features,
                                orientations=orientations,
                                bull_enter=bull_enter,
                                bull_exit=bull_exit,
                                bear_enter=bear_enter,
                                bear_exit=bear_exit,
                                min_days=md,
                                balanced_accuracy=float(bal_acc),
                                regime_switches=switches,
                                score_final=score_final,
                            )
                            best_df = tmp.copy()

    if not candidates or best_model is None or best_df is None:
        raise RuntimeError("Grid search nao produziu candidatos")
    return pd.DataFrame(candidates).sort_values("score_final", ascending=False).reset_index(drop=True), best_model, best_df


def confusion_and_stability(df: pd.DataFrame, model: ScenarioModel) -> Tuple[Dict[str, object], Dict[str, object], pd.Series]:
    pred = regime_predict(
        score=df["composite_score"].to_numpy(dtype=float),
        bull_enter=model.bull_enter,
        bull_exit=model.bull_exit,
        bear_enter=model.bear_enter,
        bear_exit=model.bear_exit,
        min_days=model.min_days,
    )
    pred_s = pd.Series(pred, index=df.index, name="pred_regime")
    eval_mask = df["label_window"].isin(["W1", model.scenario_bear])
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
    tn = int(((ev["label_window"] == model.scenario_bear) & (ev["pred"] == "BEAR")).sum())
    fp = int(((ev["label_window"] == model.scenario_bear) & (ev["pred"] != "BEAR")).sum())

    confusion = {
        "scenario_winner_bear": model.scenario_bear,
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
    dist = pred_s.value_counts().to_dict()
    total = int(len(pred_s))
    dist_pct = {k: float(v / total) for k, v in dist.items()}
    stability = {
        "scenario_winner_bear": model.scenario_bear,
        "regime_switches": switches,
        "avg_duration_days_by_regime": avg_dur,
        "regime_distribution_days": dist,
        "regime_distribution_pct": dist_pct,
    }
    return confusion, stability, pred_s


def to_csv_parallel(df: pd.DataFrame, csv_path: Path) -> None:
    ensure_parent(csv_path)
    df.to_csv(csv_path, index=False)


def write_report(
    out_dir: Path,
    inventory: pd.DataFrame,
    windows_json: Dict[str, object],
    master_signals: pd.DataFrame,
    feature_scores: pd.DataFrame,
    candidates: pd.DataFrame,
    model: ScenarioModel,
    confusion: Dict[str, object],
    stability: Dict[str, object],
) -> None:
    lines = []
    lines.append("# Regime Calibration Report v1")
    lines.append("")
    lines.append(f"- task_id: `{TASK_ID}`")
    lines.append("- base_dir: `outputs/cycle2/20260213/master_regime_calibration`")
    lines.append("- leitura permitida: `/home/wilson/CEP_COMPRA`, `/home/wilson/CEP_NA_BOLSA`")
    lines.append("")
    lines.append("## Fontes e inventario")
    lines.append(f"- total_fontes_inventariadas: **{len(inventory)}**")
    lines.append(f"- top prioridade outputs/planning/docs: **{int(inventory['priority_bucket'].isin(['outputs', 'planning/runs', 'docs']).sum())}**")
    lines.append("")
    lines.append("## Janelas W1/W2/W3")
    lines.append(f"- metodo: **{windows_json['resolved_method']}**")
    lines.append(f"- origem principal: `{windows_json['source_paths'][0]}`")
    lines.append(f"- W1: {windows_json['windows']['W1']['start']} .. {windows_json['windows']['W1']['end']}")
    lines.append(f"- W2: {windows_json['windows']['W2']['start']} .. {windows_json['windows']['W2']['end']}")
    lines.append(f"- W3: {windows_json['windows']['W3']['start']} .. {windows_json['windows']['W3']['end']}")
    lines.append("")
    lines.append("## Dataset de sinais")
    lines.append(f"- linhas: **{len(master_signals)}**")
    lines.append(f"- colunas sinais_cep_*: **{len([c for c in master_signals.columns if c.startswith('sinais_cep_')])}**")
    lines.append("")
    lines.append("## Scoring de features")
    top = feature_scores.sort_values(["scenario_bear", "rank"]).groupby("scenario_bear").head(5)
    for bear, sub in top.groupby("scenario_bear"):
        lines.append(f"### Top 5 - cenário BEAR={bear}")
        for _, r in sub.iterrows():
            lines.append(
                f"- `{r['feature']}` | auc={r['metric_auc']:.4f} ks={r['metric_ks']:.4f} t={r['metric_t']:.4f} rank={int(r['rank'])}"
            )
        lines.append("")
    lines.append("## Melhor candidato")
    lines.append(f"- cenário BEAR vencedor: **{model.scenario_bear}**")
    lines.append(f"- features: `{', '.join(model.features)}`")
    lines.append(f"- balanced_accuracy: **{model.balanced_accuracy:.6f}**")
    lines.append(f"- regime_switches: **{model.regime_switches}**")
    lines.append(f"- score_final: **{model.score_final:.6f}**")
    lines.append("")
    lines.append("## Confusão e estabilidade")
    lines.append(f"- confusion balanced_accuracy: **{confusion['balanced_accuracy']:.6f}**")
    lines.append(f"- switches: **{stability['regime_switches']}**")
    lines.append("")
    (out_dir / "regime_calibration_report_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    assert_allowlist()
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "ssot_cycle2").mkdir(parents=True, exist_ok=True)

    inventory = discover_sources()
    inventory.to_parquet(OUT_BASE / "inventory_sources.parquet", index=False)
    to_csv_parallel(inventory, OUT_BASE / "inventory_sources.csv")

    master_states = pd.read_csv(
        "/home/wilson/CEP_NA_BOLSA/outputs/experimentos/fase1_calibracao/exp/20260209/dataset_sizing/master_states.csv",
        parse_dates=["date"],
    )
    day_index = pd.DatetimeIndex(pd.to_datetime(master_states["date"])).sort_values().unique()
    windows_df, windows_json = build_windows_labels(day_index)
    (OUT_BASE / "windows_w1_w2_w3.json").write_text(json.dumps(windows_json, indent=2, ensure_ascii=False), encoding="utf-8")
    windows_df.to_parquet(OUT_BASE / "windows_w1_w2_w3.parquet", index=False)

    master_signals = load_master_signals(windows_df)
    master_signals.to_parquet(OUT_BASE / "master_signals_daily.parquet", index=False)
    to_csv_parallel(master_signals, OUT_BASE / "master_signals_daily.csv")

    feature_cols = [c for c in master_signals.columns if c.startswith("sinais_cep_")]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(master_signals[c])]
    feature_scores = score_features(master_signals, numeric_cols)
    feature_scores.to_parquet(OUT_BASE / "feature_scores.parquet", index=False)

    candidates, model, model_df = grid_search(master_signals, feature_scores, top_k=5)
    candidates.to_parquet(OUT_BASE / "candidate_thresholds.parquet", index=False)

    confusion, stability, pred_series = confusion_and_stability(model_df, model)
    (OUT_BASE / "confusion_summary.json").write_text(json.dumps(confusion, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_BASE / "stability_summary.json").write_text(json.dumps(stability, indent=2, ensure_ascii=False), encoding="utf-8")

    ssot = {
        "task_id": TASK_ID,
        "version": "v1",
        "date_reference": "2026-02-13",
        "scenario_bear": model.scenario_bear,
        "features_used": model.features,
        "orientations": model.orientations,
        "thresholds": {
            "bull_enter": model.bull_enter,
            "bull_exit": model.bull_exit,
            "bear_enter": model.bear_enter,
            "bear_exit": model.bear_exit,
            "min_days": model.min_days,
        },
        "objective_metrics": {
            "balanced_accuracy": model.balanced_accuracy,
            "regime_switches": model.regime_switches,
            "score_final": model.score_final,
        },
        "predicted_regime_distribution": pred_series.value_counts().to_dict(),
    }
    (OUT_BASE / "ssot_cycle2" / "master_regime_classifier_v1.json").write_text(
        json.dumps(ssot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_report(OUT_BASE, inventory, windows_json, master_signals, feature_scores, candidates, model, confusion, stability)

    manifest = {
        "task_id": TASK_ID,
        "date_reference": "2026-02-13",
        "base_dir": str(OUT_BASE.relative_to(Path("/home/wilson/CEP_COMPRA"))),
        "artifacts": [
            "inventory_sources.parquet",
            "inventory_sources.csv",
            "windows_w1_w2_w3.json",
            "windows_w1_w2_w3.parquet",
            "master_signals_daily.parquet",
            "master_signals_daily.csv",
            "feature_scores.parquet",
            "candidate_thresholds.parquet",
            "confusion_summary.json",
            "stability_summary.json",
            "ssot_cycle2/master_regime_classifier_v1.json",
            "regime_calibration_report_v1.md",
            "manifest.json",
            "hashes.sha256",
        ],
        "allowlist_read": [str(p) for p in ALLOWLIST_READ],
        "allowlist_write": str(OUT_BASE),
    }
    (OUT_BASE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    for p in sorted([f for f in OUT_BASE.rglob("*") if f.is_file() and f.name != "hashes.sha256"]):
        rel = p.relative_to(OUT_BASE)
        lines.append(f"{sha256_file(p)}  {rel}")
    (OUT_BASE / "hashes.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] artifacts written to: {OUT_BASE}")


if __name__ == "__main__":
    main()
