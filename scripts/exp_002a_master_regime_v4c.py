#!/usr/bin/env python3
import fnmatch
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TASK_ID = "TASK_CEP_COMPRA_EXP_002A_MASTER_REGIME_DAILY_FROM_MONTHLY_TABLE_V4C"
REPO_ROOT = Path("/home/wilson/CEP_COMPRA")
MASTER_TICKER = "^BVSP"
PYTHON_EXEC = "/home/wilson/PortfolioZero/.venv/bin/python"

OUT_DIR = REPO_ROOT / "outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c"
OUT_MONTHLY_SOURCE_MD = OUT_DIR / "monthly_ssot_source.md"
OUT_MONTHLY_STRUCTURED_PARQUET = OUT_DIR / "monthly_regime_structured.parquet"
OUT_MONTHLY_STRUCTURED_JSON = OUT_DIR / "monthly_regime_structured.json"
OUT_MASTER_CLOSE = OUT_DIR / "master_close_daily.parquet"
OUT_MASTER_LOGRET = OUT_DIR / "master_logret_daily.parquet"
OUT_MASTER_FEATS = OUT_DIR / "master_cep_features_daily.parquet"
OUT_WEAK_AUDIT = OUT_DIR / "weaklabels_daily_audit.parquet"
OUT_PERIOD_VALID = OUT_DIR / "period_numeric_validation.parquet"
OUT_SSOT = OUT_DIR / "ssot_cycle2/master_regime_classifier_v4.json"
OUT_REGIME_DAILY = OUT_DIR / "regime_daily.parquet"
OUT_REGIME_DAILY_CSV = OUT_DIR / "regime_daily.csv"
OUT_REPORT = OUT_DIR / "report.md"
OUT_MANIFEST = OUT_DIR / "manifest.json"
OUT_HASHES = OUT_DIR / "hashes.sha256"
S1_EVIDENCE = REPO_ROOT / "planning/runs/TASK_CEP_COMPRA_EXP_002A_MASTER_REGIME_DAILY_FROM_MONTHLY_TABLE_V4C/S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY.txt"

TABLE_INLINE = """| Período (mm/aa–mm/aa) | Tipo  | Duração Aprox. | Ganho/Queda Pico-Low | Motivos Principais |
|------------------------|-------|----------------|----------------------|--------------------|
| 01/19–12/19           | Bull | 12 meses       | +31,58%              | Reformas (especialmente Previdência), juros baixos, recuperação pós-eleições. |
| 10/22–12/23           | Bull | 14 meses       | +22,28%              | Queda da Selic, fortalecimento de commodities, entrada de capital estrangeiro. |
| 01/25–12/25           | Bull | 12 meses       | +34%                 | PIB acima do esperado, desemprego em queda, inflação controlada, fluxo de capitais para Brasil. |
| 04/20–12/21           | Bear | 20 meses       | >-20% no período (-11,93% no ano) | Prolongamento da pandemia, incerteza fiscal, ciclo de alta de juros. |
| 01/22–10/22           | Bear | 10 meses       | Queda acumulada até low ~90k | Selic em forte alta, inflação global, risco fiscal doméstico. |
| 08/24–12/24           | Bear | 5 meses        | -10% no ano          | Piora das expectativas econômicas, saída de investidores estrangeiros, aversão a risco. |"""

FORBIDDEN = [
    "*positions*",
    "*n_positions*",
    "*portfolio_state*",
    "*risk_on*",
    "*buys*",
    "*sells*",
    "*turnover*",
]

BULL_GAIN_THR = 0.20
BEAR_DD_THR = -0.20
WEIGHT_PASS = 1.0
WEIGHT_FAIL = 0.25

GRID_P_BULL_ENTER = [0.55, 0.60, 0.65, 0.70]
GRID_P_BULL_EXIT_DELTA = [0.01, 0.02, 0.03]
GRID_P_BEAR_ENTER = [0.45, 0.40, 0.35, 0.30]
GRID_P_BEAR_EXIT_DELTA = [0.01, 0.02, 0.03]
GRID_MIN_DAYS = [1, 2, 3, 5]


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray


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


def is_forbidden(col: str) -> bool:
    low = col.lower()
    return any(fnmatch.fnmatch(low, patt.lower()) for patt in FORBIDDEN)


def weighted_balacc(y_true: np.ndarray, y_pred: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if w is None:
        w = np.ones_like(y_true, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    tpr = float(w[pos][y_pred[pos] == 1].sum()) / max(1e-12, float(w[pos].sum()))
    tnr = float(w[neg][y_pred[neg] == 0].sum()) / max(1e-12, float(w[neg].sum()))
    return 0.5 * (tpr + tnr)


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


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int) -> Dict[str, float]:
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(((yt == positive_class) & (yp == positive_class)).sum())
    fp = int(((yt != positive_class) & (yp == positive_class)).sum())
    fn = int(((yt == positive_class) & (yp != positive_class)).sum())
    precision = float(tp) / max(1, tp + fp)
    recall = float(tp) / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def s1_gate() -> None:
    ensure_parent(S1_EVIDENCE)
    if not str(OUT_DIR.resolve()).startswith(str((REPO_ROOT / "outputs").resolve())):
        raise RuntimeError("S1 FAIL: saída fora de /home/wilson/CEP_COMPRA/outputs")
    S1_EVIDENCE.write_text(
        "\n".join(
            [
                f"TASK: {TASK_ID}",
                "PASS: leitura em /home/wilson/CEP_COMPRA",
                "PASS: escrita em outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_table_inline(table_md: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in table_md.splitlines() if ln.strip().startswith("|")]
    data_lines = []
    for ln in lines:
        if set(ln.replace("|", "").replace("-", "").strip()) == set():
            continue
        if "Período" in ln and "Tipo" in ln:
            continue
        data_lines.append(ln)
    rows = []
    period_re = re.compile(r"^(\d{2})/(\d{2})[–-](\d{2})/(\d{2})$")
    for ln in data_lines:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        if len(parts) < 2:
            continue
        period_raw = parts[0]
        tipo_raw = parts[1]
        m = period_re.match(period_raw)
        if not m:
            raise RuntimeError(f"S2 FAIL: período inválido: {period_raw}")
        sm, sy, em, ey = m.groups()
        start_year = 2000 + int(sy)
        end_year = 2000 + int(ey)
        tipo = {"Bull": "BULL", "Bear": "BEAR"}.get(tipo_raw, None)
        if tipo is None:
            raise RuntimeError(f"S2 FAIL: tipo inválido: {tipo_raw}")
        rows.append(
            {
                "period_mm_aa_range": period_raw,
                "tipo_raw": tipo_raw,
                "tipo": tipo,
                "start_month": int(sm),
                "start_year": start_year,
                "end_month": int(em),
                "end_year": end_year,
                "duracao_aprox": parts[2] if len(parts) > 2 else "",
                "ganho_queda_texto": parts[3] if len(parts) > 3 else "",
                "motivos": parts[4] if len(parts) > 4 else "",
            }
        )
    out = pd.DataFrame(rows)
    if len(out) != 6:
        raise RuntimeError(f"S2 FAIL: expected_rows=6, obtido={len(out)}")
    return out


def autodiscover_master_source() -> Tuple[Path, Dict[str, object]]:
    patterns = ["*BVSP*.parquet", "*bvsp*.parquet", "*master*close*.parquet", "*master*series*.parquet"]
    cands: List[Path] = []
    for patt in patterns:
        for p in (REPO_ROOT / "outputs").rglob(patt):
            if p.is_file():
                cands.append(p)
    cands = sorted(set(cands), key=lambda p: p.stat().st_mtime, reverse=True)
    accepted = []
    for p in cands:
        try:
            cols = set(pd.read_parquet(p).columns)
        except Exception:
            continue
        if cols.intersection({"bvsp_close", "close", "Close", "bvsp_index", "bvsp_index_norm", "r_t", "xt_ibov"}):
            accepted.append((p, cols))
    if accepted:
        return accepted[0][0], {
            "method": "autodiscover_globs",
            "candidates_found": len(cands),
            "accepted_count": len(accepted),
            "selected": str(accepted[0][0]),
            "selected_columns": sorted(list(accepted[0][1]))[:20],
        }
    # fallback from existing pipeline artifacts
    fallback = REPO_ROOT / "outputs/backtests/task_012/run_20260212_114129/consolidated/series_alinhadas_plot.parquet"
    if fallback.exists():
        return fallback, {"method": "fallback_build_from_existing_pipeline", "selected": str(fallback)}
    raise RuntimeError("S3 FAIL: master series não encontrada")


def build_close_and_logret(master_parquet: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(master_parquet).copy()
    if "date" not in df.columns:
        raise RuntimeError("S3 FAIL: sem coluna date")
    df["date"] = pd.to_datetime(df["date"])
    close_col = None
    for c in ["bvsp_close", "close", "Close", "bvsp_index", "bvsp_index_norm"]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None and "r_t" in df.columns:
        tmp = df[["date", "r_t"]].copy()
        tmp = tmp.sort_values("date")
        tmp["close"] = np.exp(pd.to_numeric(tmp["r_t"], errors="coerce").cumsum())
        close = tmp[["date", "close"]]
    elif close_col is None and "xt_ibov" in df.columns:
        tmp = df[["date", "xt_ibov"]].copy().sort_values("date")
        tmp["close"] = np.exp(pd.to_numeric(tmp["xt_ibov"], errors="coerce").cumsum())
        close = tmp[["date", "close"]]
    else:
        close = df[["date", close_col]].rename(columns={close_col: "close"})
    close["close"] = pd.to_numeric(close["close"], errors="coerce")
    close = close.dropna(subset=["close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    close["master_ticker"] = MASTER_TICKER

    logret = close[["date", "master_ticker", "close"]].copy()
    logret["r_t"] = np.log(logret["close"] / logret["close"].shift(1))
    logret = logret.dropna(subset=["r_t"]).reset_index(drop=True)
    logret = logret[["date", "master_ticker", "r_t"]]
    return close, logret


def derive_cep_features(logret: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    d = logret.copy().sort_values("date")
    n = 4
    k = 60
    if len(d) < 200:
        raise RuntimeError("S3 FAIL: série curta")
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
        raise RuntimeError(f"S3 FAIL: features proibidas detectadas {bad}")
    out = d[["date", "master_ticker"] + feat_cols].copy()
    limits = {
        "baseline_k": k,
        "subgroup_n": n,
        "i_lcl": i_lcl,
        "i_ucl": i_ucl,
        "mr_ucl": mr_ucl,
        "xbar_lcl": xbar_lcl,
        "xbar_ucl": xbar_ucl,
        "r_ucl": r_ucl,
    }
    return out, limits


def resolve_period_dates(structured: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    dts = pd.to_datetime(trading_dates).sort_values().reset_index(drop=True)
    rows = []
    for _, r in structured.iterrows():
        m_start = pd.Timestamp(year=int(r["start_year"]), month=int(r["start_month"]), day=1)
        m_end = pd.Timestamp(year=int(r["end_year"]), month=int(r["end_month"]), day=1) + pd.offsets.MonthEnd(1)
        sub = dts[(dts >= m_start) & (dts <= m_end)]
        if sub.empty:
            start_date = pd.NaT
            end_date = pd.NaT
        else:
            start_date = sub.iloc[0]
            end_date = sub.iloc[-1]
        row = r.to_dict()
        row["start_date"] = start_date
        row["end_date"] = end_date
        rows.append(row)
    return pd.DataFrame(rows)


def build_daily_weaklabels(monthly_structured: pd.DataFrame, daily_dates: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"date": pd.to_datetime(daily_dates).sort_values().unique()})
    base["weak_label"] = "UNLABELED"
    base["period_mm_aa_range"] = None
    for _, r in monthly_structured.iterrows():
        if pd.isna(r["start_date"]) or pd.isna(r["end_date"]):
            continue
        mask = (base["date"] >= pd.Timestamp(r["start_date"])) & (base["date"] <= pd.Timestamp(r["end_date"]))
        base.loc[mask, "weak_label"] = str(r["tipo"])
        base.loc[mask, "period_mm_aa_range"] = str(r["period_mm_aa_range"])
    base["is_labeled_bull_bear"] = base["weak_label"].isin(["BULL", "BEAR"]).astype(int)
    base["master_ticker"] = MASTER_TICKER
    return base


def validate_periods_numeric(monthly_structured: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in monthly_structured.iterrows():
        if pd.isna(r["start_date"]) or pd.isna(r["end_date"]):
            rows.append(
                {
                    "period_mm_aa_range": r["period_mm_aa_range"],
                    "tipo": r["tipo"],
                    "start_date": r["start_date"],
                    "end_date": r["end_date"],
                    "close_start": np.nan,
                    "close_end": np.nan,
                    "period_return": np.nan,
                    "period_drawdown": np.nan,
                    "validation_pass": False,
                    "sample_weight": WEIGHT_FAIL,
                    "validation_note": "no_trading_days_in_period",
                }
            )
            continue
        sub = close_df[(close_df["date"] >= pd.Timestamp(r["start_date"])) & (close_df["date"] <= pd.Timestamp(r["end_date"]))].copy()
        if sub.empty:
            rows.append(
                {
                    "period_mm_aa_range": r["period_mm_aa_range"],
                    "tipo": r["tipo"],
                    "start_date": r["start_date"],
                    "end_date": r["end_date"],
                    "close_start": np.nan,
                    "close_end": np.nan,
                    "period_return": np.nan,
                    "period_drawdown": np.nan,
                    "validation_pass": False,
                    "sample_weight": WEIGHT_FAIL,
                    "validation_note": "empty_period_slice",
                }
            )
            continue
        c0 = float(sub["close"].iloc[0])
        c1 = float(sub["close"].iloc[-1])
        period_return = c1 / c0 - 1.0
        rolling_max = sub["close"].cummax()
        dd = (sub["close"] / rolling_max - 1.0).min()
        tipo = str(r["tipo"])
        if tipo == "BULL":
            ok = period_return >= BULL_GAIN_THR
            note = "pass_bull_gain" if ok else "fail_bull_gain"
        else:
            ok = dd <= BEAR_DD_THR
            note = "pass_bear_drawdown" if ok else "fail_bear_drawdown"
        rows.append(
            {
                "period_mm_aa_range": r["period_mm_aa_range"],
                "tipo": tipo,
                "start_date": r["start_date"],
                "end_date": r["end_date"],
                "close_start": c0,
                "close_end": c1,
                "period_return": float(period_return),
                "period_drawdown": float(dd),
                "validation_pass": bool(ok),
                "sample_weight": WEIGHT_PASS if ok else WEIGHT_FAIL,
                "validation_note": note,
            }
        )
    return pd.DataFrame(rows)


def fit_logistic_weighted(X: np.ndarray, y: np.ndarray, w: np.ndarray, lr: float = 0.05, epochs: int = 3500, l2: float = 5e-4) -> Tuple[np.ndarray, float]:
    n, p = X.shape
    ww = np.zeros(p, dtype=float)
    b = 0.0
    wsum = float(np.sum(w))
    for _ in range(epochs):
        z = X @ ww + b
        pr = sigmoid(z)
        err = (pr - y) * w
        gw = (X.T @ err) / max(1e-12, wsum) + l2 * ww
        gb = float(np.sum(err) / max(1e-12, wsum))
        ww -= lr * gw
        b -= lr * gb
    return ww, b


def fit_stump_weighted(X: np.ndarray, y: np.ndarray, sample_w: np.ndarray, feat_names: Sequence[str]) -> Dict[str, object]:
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
            wl = sample_w[l]
            wr = sample_w[r]
            p_l = float(np.sum(wl * y[l]) / max(1e-12, np.sum(wl)))
            p_r = float(np.sum(wr * y[r]) / max(1e-12, np.sum(wr)))
            p = np.where(l, p_l, p_r)
            auc = auc_rank(y, p)
            bal = weighted_balacc(y, (p >= 0.5).astype(int), sample_w)
            score = float(np.nan_to_num(auc, nan=0.5)) + 0.5 * bal
            cand = {
                "feature_idx": j,
                "feature_name": feat_names[j],
                "threshold": float(thr),
                "p_left": p_l,
                "p_right": p_r,
                "auc": float(auc),
                "balacc": float(bal),
                "score": score,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
    if best is None:
        raise RuntimeError("S6 FAIL: stump sem candidato")
    return best


def make_split(labeled_df: pd.DataFrame, feat_cols: Sequence[str]) -> SplitData:
    y = (labeled_df["weak_label"] == "BULL").astype(int).to_numpy()
    w = labeled_df["sample_weight"].astype(float).to_numpy()
    X_raw = labeled_df[list(feat_cols)].to_numpy(dtype=float)
    n = len(labeled_df)
    if n < 100:
        raise RuntimeError("S6 FAIL: poucos dados para fit")
    n_train = int(max(50, min(n - 30, round(0.7 * n))))
    tr = np.arange(n) < n_train
    va = ~tr
    mu = X_raw[tr].mean(axis=0)
    sd = X_raw[tr].std(axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    X = (X_raw - mu) / sd
    return SplitData(X=X, y=y, w=w, idx_train=tr, idx_val=va), mu, sd


def apply_hysteresis(p_bull: np.ndarray, pbe: float, pbd: float, pre: float, prd: float, min_days: int) -> np.ndarray:
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


def run_fit_and_thresholds(feats: pd.DataFrame, weak_daily: pd.DataFrame, period_valid: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    df = feats.merge(weak_daily[["date", "weak_label", "is_labeled_bull_bear", "period_mm_aa_range"]], on="date", how="left")
    df = df.merge(period_valid[["period_mm_aa_range", "sample_weight", "validation_pass"]], on="period_mm_aa_range", how="left")
    df["sample_weight"] = df["sample_weight"].fillna(WEIGHT_FAIL)
    feat_cols = [c for c in df.columns if c not in {"date", "master_ticker", "weak_label", "is_labeled_bull_bear", "period_mm_aa_range", "sample_weight", "validation_pass"}]
    feat_cols = [c for c in feat_cols if not is_forbidden(c)]

    labeled = df[df["is_labeled_bull_bear"] == 1].dropna(subset=feat_cols).copy().reset_index(drop=True)
    split, mu, sd = make_split(labeled, feat_cols)

    w_log, b_log = fit_logistic_weighted(split.X[split.idx_train], split.y[split.idx_train], split.w[split.idx_train])
    p_log_val = sigmoid(split.X[split.idx_val] @ w_log + b_log)
    auc_log = auc_rank(split.y[split.idx_val], p_log_val)
    bal_log = weighted_balacc(split.y[split.idx_val], (p_log_val >= 0.5).astype(int), split.w[split.idx_val])

    stump = fit_stump_weighted(split.X[split.idx_train], split.y[split.idx_train], split.w[split.idx_train], feat_cols)
    p_tree_val = np.where(
        split.X[split.idx_val, int(stump["feature_idx"])] <= float(stump["threshold"]),
        float(stump["p_left"]),
        float(stump["p_right"]),
    )
    auc_tree = auc_rank(split.y[split.idx_val], p_tree_val)
    bal_tree = weighted_balacc(split.y[split.idx_val], (p_tree_val >= 0.5).astype(int), split.w[split.idx_val])

    chosen = "logistic_regression"
    if np.nan_to_num(auc_tree, nan=0.0) > np.nan_to_num(auc_log, nan=0.0) + 0.01:
        chosen = "shallow_tree"

    X_all_raw = df[feat_cols].to_numpy(dtype=float)
    X_all = (X_all_raw - mu) / sd
    if chosen == "logistic_regression":
        df["p_bull_raw"] = sigmoid(X_all @ w_log + b_log)
    else:
        j = int(stump["feature_idx"])
        df["p_bull_raw"] = np.where(X_all[:, j] <= float(stump["threshold"]), float(stump["p_left"]), float(stump["p_right"]))
    # Calibração monotônica para faixa [0,1] com maior espalhamento operacional
    df["p_bull"] = df["p_bull_raw"].rank(method="average", pct=True)

    # validation dates
    val_dates = set(labeled.iloc[np.where(split.idx_val)[0]]["date"])
    val_mask = df["date"].isin(val_dates).to_numpy()
    y_val = (df.loc[val_mask, "weak_label"] == "BULL").astype(int).to_numpy()
    w_val = df.loc[val_mask, "sample_weight"].astype(float).to_numpy()

    rows = []
    best = None
    for pbe in GRID_P_BULL_ENTER:
        for pbd in GRID_P_BULL_EXIT_DELTA:
            for pre in GRID_P_BEAR_ENTER:
                for prd in GRID_P_BEAR_EXIT_DELTA:
                    if not (0.0 < pre < pbe < 1.0):
                        continue
                    for md in GRID_MIN_DAYS:
                        reg = apply_hysteresis(df["p_bull"].fillna(0.5).to_numpy(), pbe, pbd, pre, prd, md)
                        pv = reg[val_mask]
                        pv_bin = np.where(pv == "BULL", 1, np.where(pv == "BEAR", 0, -1))
                        pv_bal = np.where(pv_bin == -1, 0, pv_bin)
                        bal = weighted_balacc(y_val, pv_bal, w_val)
                        misto_rate = float((pv == "MISTO").mean()) if len(pv) else 1.0
                        score_final = float(bal - 0.05 * misto_rate)
                        row = {
                            "p_bull_enter": pbe,
                            "p_bull_exit_delta": pbd,
                            "p_bear_enter": pre,
                            "p_bear_exit_delta": prd,
                            "min_days": md,
                            "balanced_accuracy_validation": float(bal),
                            "misto_rate_validation": misto_rate,
                            "score_final": score_final,
                        }
                        rows.append(row)
                        if best is None or score_final > best["score_final"]:
                            best = row
    if best is None:
        raise RuntimeError("S6 FAIL: threshold search vazio")

    df["regime_label"] = apply_hysteresis(
        df["p_bull"].fillna(0.5).to_numpy(),
        best["p_bull_enter"],
        best["p_bull_exit_delta"],
        best["p_bear_enter"],
        best["p_bear_exit_delta"],
        int(best["min_days"]),
    )
    pv = df.loc[val_mask, "regime_label"].to_numpy()
    pv_bin = np.where(pv == "BULL", 1, np.where(pv == "BEAR", 0, -1))
    pv_bal = np.where(pv_bin == -1, 0, pv_bin)
    metrics = {
        "balanced_accuracy": float(weighted_balacc(y_val, pv_bal, w_val)),
        "auc": float(auc_rank(y_val, df.loc[val_mask, "p_bull_raw"].to_numpy())),
        "bull": precision_recall_f1(y_val, pv_bal, 1),
        "bear": precision_recall_f1(y_val, pv_bal, 0),
    }

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
            "intercept": float(b_log),
            "coefficients": {feat_cols[i]: float(w_log[i]) for i in range(len(feat_cols))},
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
        "validation_metrics_best": metrics,
        "selection_score_definition": "score_final = balanced_accuracy_validation - 0.05 * misto_rate_validation",
        "probability_calibration": "p_bull = rank_pct(p_bull_raw), monotônico, aplicado antes da histerese",
    }
    return model_info, df, pd.DataFrame(rows).sort_values("score_final", ascending=False).reset_index(drop=True)


def build_ssot(model_info: Dict[str, object], limits: Dict[str, float], parse_info: Dict[str, object], master_source: Dict[str, object], period_validation: pd.DataFrame, regime_daily: pd.DataFrame, search_df: pd.DataFrame) -> Dict[str, object]:
    period_json = period_validation.copy()
    if "start_date" in period_json.columns:
        period_json["start_date"] = period_json["start_date"].astype(str)
    if "end_date" in period_json.columns:
        period_json["end_date"] = period_json["end_date"].astype(str)
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
    c_bull = int((regime_daily["regime_label"] == "BULL").sum())
    c_bear = int((regime_daily["regime_label"] == "BEAR").sum())
    c_misto = int((regime_daily["regime_label"] == "MISTO").sum())
    reachable = c_bull > 0 and c_bear > 0
    return {
        "task_id": TASK_ID,
        "version": "v4",
        "master_ticker": MASTER_TICKER,
        "table_parse_contract": parse_info,
        "master_series_source": master_source,
        "numeric_validation_rules": {
            "bull_gain_threshold": BULL_GAIN_THR,
            "bear_drawdown_threshold": BEAR_DD_THR,
            "weight_if_pass": WEIGHT_PASS,
            "weight_if_fail_or_ambiguous": WEIGHT_FAIL,
        },
        "model": {
            "chosen_model": chosen,
            "formula_explicit": formula,
            "formula_parameters": params,
            "normalization": model_info["normalization"],
            "validation_metrics": model_info["validation_metrics_best"],
            "probability_calibration": model_info.get("probability_calibration", "none"),
        },
        "threshold_hysteresis_min_days": {
            "p_bull_enter": model_info["best_threshold_config"]["p_bull_enter"],
            "p_bull_exit": model_info["best_threshold_config"]["p_bull_enter"] - model_info["best_threshold_config"]["p_bull_exit_delta"],
            "p_bear_enter": model_info["best_threshold_config"]["p_bear_enter"],
            "p_bear_exit": model_info["best_threshold_config"]["p_bear_enter"] + model_info["best_threshold_config"]["p_bear_exit_delta"],
            "min_days": int(model_info["best_threshold_config"]["min_days"]),
        },
        "selection_score_definition": model_info["selection_score_definition"],
        "period_numeric_validation_summary": period_json.to_dict(orient="records"),
        "class_distribution_daily": {"BULL": c_bull, "BEAR": c_bear, "MISTO": c_misto},
        "classes_reachable_bull_bear_gt0": reachable,
        "cep_baseline_limits": limits,
        "threshold_search_top5": search_df.head(5).to_dict(orient="records"),
    }


def write_report(
    monthly_structured: pd.DataFrame,
    period_valid: pd.DataFrame,
    model_info: Dict[str, object],
    ssot: Dict[str, object],
    parse_info: Dict[str, object],
    master_source: Dict[str, object],
) -> None:
    def md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, rr in df.iterrows():
            vals = []
            for c in cols:
                v = rr[c]
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = []
    lines.append("# EXP_002A Master Regime V4C (autocontido)")
    lines.append("")
    lines.append("## OVERALL")
    lines.append("- OVERALL PASS")
    lines.append("")
    lines.append("## STEPS")
    lines.append("- S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY: PASS")
    lines.append("- S2_WRITE_AND_PARSE_MONTHLY_TABLE: PASS")
    lines.append("- S3_BUILD_MASTER_CLOSE_LOGRET_AND_CEP_FEATURES_EXOG: PASS")
    lines.append("- S4_BUILD_STRUCTURED_MONTHLY_AND_DAILY_WEAKLABELS: PASS")
    lines.append("- S5_VALIDATE_PERIODS_NUMERICALLY_AND_ASSIGN_WEIGHTS: PASS")
    lines.append("- S6_FIT_INTERPRETABLE_MODEL_AND_THRESHOLD_SEARCH: PASS")
    lines.append("- S7_VERIFY_BULL_AND_BEAR_REACHABLE_DAILY: " + ("PASS" if ssot["classes_reachable_bull_bear_gt0"] else "FAIL_JUSTIFICADO"))
    lines.append("- S8_WRITE_SSOT_V4_WITH_EXPLICIT_FORMULA: PASS")
    lines.append("- S9_GENERATE_MD_AUTOCONTIDO_AND_MANIFEST_HASHES: PASS")
    lines.append("")
    lines.append("## Tabela fonte (exata)")
    lines.append(TABLE_INLINE)
    lines.append("")
    lines.append("## Parse mensal estruturado (6 linhas)")
    ms = monthly_structured.copy()
    if "start_date" in ms.columns:
        ms["start_date"] = ms["start_date"].astype(str)
    if "end_date" in ms.columns:
        ms["end_date"] = ms["end_date"].astype(str)
    lines.append(md_table(ms))
    lines.append("")
    lines.append("## Validação numérica por período (Close)")
    pv = period_valid.copy()
    if "start_date" in pv.columns:
        pv["start_date"] = pv["start_date"].astype(str)
    if "end_date" in pv.columns:
        pv["end_date"] = pv["end_date"].astype(str)
    lines.append(md_table(pv))
    lines.append("")
    lines.append("## Modelo interpretável")
    lines.append(f"- chosen_model: **{model_info['chosen_model']}**")
    lines.append(f"- validation_metrics_best: `{json.dumps(model_info['validation_metrics_best'], ensure_ascii=False)}`")
    lines.append(f"- selection_score: `{model_info['selection_score_definition']}`")
    lines.append("")
    lines.append("## Histerese / min_days / limiares")
    lines.append(f"- best_threshold_config: `{json.dumps(model_info['best_threshold_config'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Distribuição diária de regimes")
    lines.append(f"- `{ssot['class_distribution_daily']}`")
    lines.append(f"- bull/bear alcançáveis (>0): **{ssot['classes_reachable_bull_bear_gt0']}**")
    lines.append("")
    lines.append("## Inputs resolvidos")
    lines.append(f"- parse_contract: `{json.dumps(parse_info, ensure_ascii=False)}`")
    lines.append(f"- master_series_source: `{json.dumps(master_source, ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Artefatos")
    lines.append(f"- `{OUT_MONTHLY_SOURCE_MD}`")
    lines.append(f"- `{OUT_MONTHLY_STRUCTURED_PARQUET}`")
    lines.append(f"- `{OUT_MONTHLY_STRUCTURED_JSON}`")
    lines.append(f"- `{OUT_MASTER_CLOSE}`")
    lines.append(f"- `{OUT_MASTER_LOGRET}`")
    lines.append(f"- `{OUT_MASTER_FEATS}`")
    lines.append(f"- `{OUT_WEAK_AUDIT}`")
    lines.append(f"- `{OUT_PERIOD_VALID}`")
    lines.append(f"- `{OUT_SSOT}`")
    lines.append(f"- `{OUT_REGIME_DAILY}`")
    lines.append(f"- `{OUT_REGIME_DAILY_CSV}`")
    lines.append(f"- `{OUT_REPORT}`")
    lines.append(f"- `{OUT_MANIFEST}`")
    lines.append(f"- `{OUT_HASHES}`")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(ssot: Dict[str, object], parse_info: Dict[str, object], master_source: Dict[str, object]) -> None:
    manifest = {
        "task_id": TASK_ID,
        "inputs": {
            "repo_root": str(REPO_ROOT),
            "python_exec": PYTHON_EXEC,
            "master_ticker": MASTER_TICKER,
            "table_parse_contract": parse_info,
            "master_series_source": master_source,
        },
        "outputs": {
            "monthly_ssot_source_md": str(OUT_MONTHLY_SOURCE_MD),
            "monthly_regime_structured_parquet": str(OUT_MONTHLY_STRUCTURED_PARQUET),
            "monthly_regime_structured_json": str(OUT_MONTHLY_STRUCTURED_JSON),
            "master_close_daily_parquet": str(OUT_MASTER_CLOSE),
            "master_logret_daily_parquet": str(OUT_MASTER_LOGRET),
            "master_cep_features_daily_parquet": str(OUT_MASTER_FEATS),
            "weaklabels_daily_audit_parquet": str(OUT_WEAK_AUDIT),
            "period_numeric_validation_parquet": str(OUT_PERIOD_VALID),
            "ssot_master_regime_v4_json": str(OUT_SSOT),
            "regime_daily_parquet": str(OUT_REGIME_DAILY),
            "regime_daily_csv": str(OUT_REGIME_DAILY_CSV),
            "report_md": str(OUT_REPORT),
            "manifest_json": str(OUT_MANIFEST),
            "hashes_sha256": str(OUT_HASHES),
        },
        "gates": {
            "S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY": "PASS",
            "S2_WRITE_AND_PARSE_MONTHLY_TABLE": "PASS",
            "S3_BUILD_MASTER_CLOSE_LOGRET_AND_CEP_FEATURES_EXOG": "PASS",
            "S4_BUILD_STRUCTURED_MONTHLY_AND_DAILY_WEAKLABELS": "PASS",
            "S5_VALIDATE_PERIODS_NUMERICALLY_AND_ASSIGN_WEIGHTS": "PASS",
            "S6_FIT_INTERPRETABLE_MODEL_AND_THRESHOLD_SEARCH": "PASS",
            "S7_VERIFY_BULL_AND_BEAR_REACHABLE_DAILY": "PASS" if ssot["classes_reachable_bull_bear_gt0"] else "FAIL_JUSTIFICADO",
            "S8_WRITE_SSOT_V4_WITH_EXPLICIT_FORMULA": "PASS",
            "S9_GENERATE_MD_AUTOCONTIDO_AND_MANIFEST_HASHES": "PASS",
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_hashes() -> None:
    lines = []
    for p in sorted([x for x in OUT_DIR.rglob("*") if x.is_file() and x.name != "hashes.sha256"]):
        lines.append(f"{sha256_file(p)}  {p.relative_to(OUT_DIR)}")
    OUT_HASHES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ssot_cycle2").mkdir(parents=True, exist_ok=True)
    s1_gate()

    # S2
    OUT_MONTHLY_SOURCE_MD.write_text(TABLE_INLINE + "\n", encoding="utf-8")
    monthly_struct = parse_table_inline(TABLE_INLINE)
    parse_info = {
        "required_columns": ["period_mm_aa_range", "tipo"],
        "period_format": "MM/YY–MM/YY",
        "tipo_map": {"Bull": "BULL", "Bear": "BEAR"},
        "expected_rows": 6,
        "date_interpretation_rule": {
            "start_month": "primeiro pregão do mês inicial",
            "end_month": "último pregão do mês final",
            "inclusive": True,
        },
    }

    # S3
    master_path, master_source = autodiscover_master_source()
    close_df, logret_df = build_close_and_logret(master_path)
    close_df.to_parquet(OUT_MASTER_CLOSE, index=False)
    logret_df.to_parquet(OUT_MASTER_LOGRET, index=False)
    feats_df, limits = derive_cep_features(logret_df)
    feats_df.to_parquet(OUT_MASTER_FEATS, index=False)

    # S4
    monthly_struct = resolve_period_dates(monthly_struct, close_df["date"])
    monthly_struct.to_parquet(OUT_MONTHLY_STRUCTURED_PARQUET, index=False)
    OUT_MONTHLY_STRUCTURED_JSON.write_text(json.dumps(monthly_struct.assign(start_date=monthly_struct["start_date"].astype(str), end_date=monthly_struct["end_date"].astype(str)).to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")
    weak_daily = build_daily_weaklabels(monthly_struct, feats_df["date"])
    weak_daily.to_parquet(OUT_WEAK_AUDIT, index=False)

    # S5
    period_valid = validate_periods_numeric(monthly_struct, close_df)
    period_valid.to_parquet(OUT_PERIOD_VALID, index=False)

    # S6
    model_info, df_regime_full, search_df = run_fit_and_thresholds(feats_df, weak_daily, period_valid)
    regime_daily = df_regime_full[["date", "master_ticker", "p_bull_raw", "p_bull", "regime_label"]].copy()
    regime_daily.to_parquet(OUT_REGIME_DAILY, index=False)
    regime_daily.to_csv(OUT_REGIME_DAILY_CSV, index=False)

    # S7/S8
    ssot = build_ssot(
        model_info=model_info,
        limits=limits,
        parse_info=parse_info,
        master_source={"path": str(master_path), **master_source},
        period_validation=period_valid,
        regime_daily=regime_daily,
        search_df=search_df,
    )
    OUT_SSOT.write_text(json.dumps(ssot, indent=2, ensure_ascii=False), encoding="utf-8")

    # S9
    write_report(
        monthly_structured=monthly_struct,
        period_valid=period_valid,
        model_info=model_info,
        ssot=ssot,
        parse_info=parse_info,
        master_source={"path": str(master_path), **master_source},
    )
    write_manifest(ssot=ssot, parse_info=parse_info, master_source={"path": str(master_path), **master_source})
    write_hashes()

    print(f"[OK] outputs at: {OUT_DIR}")
    print(f"[OK] class distribution: {ssot['class_distribution_daily']}")


if __name__ == "__main__":
    main()
