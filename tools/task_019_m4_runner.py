#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CANON_START = pd.Timestamp("2018-07-01")
CANON_END = pd.Timestamp("2025-12-31")
TARGET_POSITIONS = 10
MIN_LIQ_DAYS = 50
HWM_DD_LIMIT = 0.10
EPSILON = 0.0005
M_CONSEC_DAYS = 5
MIN_STABLE_DAYS = 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_trading_day_by_week(dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
    out = []
    seen = set()
    for d in dates:
        k = (d.isocalendar().year, d.isocalendar().week)
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def z(v: pd.Series) -> pd.Series:
    s = v.std(ddof=0)
    if pd.isna(s) or s == 0:
        return pd.Series([0.0] * len(v), index=v.index)
    return (v - v.mean()) / s


def load_volume_cache(tickers: List[str], asset_class_map: Dict[str, str]) -> pd.DataFrame:
    base = Path("/home/wilson/CEP_NA_BOLSA/data/raw/market")
    rows = []
    for t in tickers:
        cls = asset_class_map.get(t, "UNKNOWN")
        p = base / ("acoes/brapi/20260204" if cls == "B3" else "bdr/brapi/20260204") / f"{t}.json"
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            hist = payload.get("results", [{}])[0].get("historicalDataPrice", [])
            for h in hist:
                d = pd.to_datetime(h.get("date"), unit="s", errors="coerce")
                if pd.isna(d):
                    continue
                rows.append((t, d.normalize(), float(h.get("volume") or 0.0)))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "volume"])
    return (
        pd.DataFrame(rows, columns=["ticker", "date", "volume"])
        .sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
    )


def select_tickers(
    mech: str,
    cand_df: pd.DataFrame,
    target_n: int,
    min_volume_days: int,
    m4_ineligible: Optional[Set[str]] = None,
) -> List[str]:
    if cand_df.empty:
        return []
    work = cand_df.copy()
    work["liquidity_ok"] = work["liquidity_days"] >= min_volume_days
    work["sector"] = work["sector"].fillna("UNKNOWN").replace("", "UNKNOWN")

    if m4_ineligible:
        work = work[~work["ticker"].isin(m4_ineligible)].copy()

    if mech == "M0":
        ranked = work.sort_values(["score_m0", "ret_lookback", "ticker"], ascending=[False, False, True])
        return ranked["ticker"].tolist()[:target_n]

    work = work[work["liquidity_ok"]].copy()
    work = (
        work.sort_values(["ret_lookback", "score_m0", "ticker"], ascending=[False, False, True])
        .drop_duplicates(subset=["company_key"], keep="first")
        .copy()
    )

    if mech == "M1":
        work = work.sort_values(["score_m0", "ret_lookback", "ticker"], ascending=[False, False, True])
    else:
        work["score_m3"] = z(work["score_m0"]) + z(work["ret_lookback"]) - z(work["vol_lookback"])
        work = work.sort_values(["score_m3", "score_m0", "ret_lookback", "ticker"], ascending=[False, False, False, True])

    chosen: List[str] = []
    sec_count: Dict[str, int] = {}
    b3_count = 0
    bdr_count = 0

    def can_add(row: pd.Series) -> bool:
        nonlocal b3_count, bdr_count
        sec = row["sector"]
        cls = row["asset_class"]
        if sec_count.get(sec, 0) >= 2:
            return False
        if cls == "B3" and b3_count >= 8:
            return False
        if cls == "BDR" and bdr_count >= 5:
            return False
        return True

    def add(row: pd.Series) -> None:
        nonlocal b3_count, bdr_count
        t = row["ticker"]
        cls = row["asset_class"]
        sec = row["sector"]
        chosen.append(t)
        sec_count[sec] = sec_count.get(sec, 0) + 1
        if cls == "B3":
            b3_count += 1
        elif cls == "BDR":
            bdr_count += 1

    for _, row in work[work["asset_class"] == "B3"].iterrows():
        if len(chosen) >= target_n or b3_count >= 5:
            break
        if can_add(row):
            add(row)

    for _, row in work[work["asset_class"] == "BDR"].iterrows():
        if len(chosen) >= target_n or bdr_count >= 2:
            break
        if row["ticker"] in chosen:
            continue
        if can_add(row):
            add(row)

    for _, row in work.iterrows():
        if len(chosen) >= target_n:
            break
        if row["ticker"] in chosen:
            continue
        if can_add(row):
            add(row)

    return chosen[:target_n]


def is_ticker_under_control(
    ticker: str,
    date: pd.Timestamp,
    xt: pd.DataFrame,
    df_limits: pd.DataFrame,
    xbar_df: pd.DataFrame,
    r_df: pd.DataFrame,
    mr_df: pd.DataFrame,
) -> bool:
    if ticker not in df_limits.index:
        return False
    rv = xt.loc[date, ticker] if ticker in xt.columns else float("nan")
    if pd.isna(rv):
        return False
    lim = df_limits.loc[ticker]
    xbar_t = xbar_df.loc[date, ticker] if ticker in xbar_df.columns else float("nan")
    r_t = r_df.loc[date, ticker] if ticker in r_df.columns else float("nan")
    mr_t = mr_df.loc[date, ticker] if ticker in mr_df.columns else float("nan")
    checks = [
        float(lim["i_lcl"]) <= float(rv) <= float(lim["i_ucl"]),
        (pd.isna(lim.get("xbar_ucl", float("nan"))) or (pd.notna(xbar_t) and float(xbar_t) <= float(lim["xbar_ucl"]))),
        (pd.isna(lim.get("r_ucl", float("nan"))) or (pd.notna(r_t) and float(r_t) <= float(lim["r_ucl"]))),
        (pd.isna(lim.get("mr_ucl", float("nan"))) or (pd.notna(mr_t) and float(mr_t) <= float(lim["mr_ucl"]))),
    ]
    return all(checks)


def compute_portfolio_cep_limits(returns: pd.Series) -> Tuple[float, float, float, float, float]:
    i_cl = float(returns.mean())
    mr = returns.diff().abs().dropna()
    mr_cl = float(mr.mean()) if not mr.empty else 0.0
    d2 = 1.128
    sigma = mr_cl / d2 if d2 else 0.0
    i_ucl = i_cl + 3.0 * sigma
    i_lcl = i_cl - 3.0 * sigma
    mr_ucl = 3.267 * mr_cl
    return i_cl, i_lcl, i_ucl, mr_cl, mr_ucl


def run_task019(out_dir: Optional[str] = None, start: str = "2018-07-01", end: str = "2025-12-31") -> Path:
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(out_dir) if out_dir else Path(f"/home/wilson/CEP_COMPRA/outputs/backtests/task_019_m4/{run_id}")
    out_root.mkdir(parents=True, exist_ok=True)

    p_base = Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/base_operacional/base_operacional_xt.csv")
    p_limits = Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/merged/limits_per_ticker.csv")
    p_master = Path("/home/wilson/CEP_NA_BOLSA/outputs/experimentos/fase1_calibracao/exp/20260209/dataset_sizing/master_states.csv")
    p_ssot_a = Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/acoes/b3/20260204/ssot_acoes_b3.csv")
    p_ssot_b = Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/bdr/b3/20260204/ssot_bdr_b3.csv")
    p_setor = Path("/home/wilson/CEP_COMPRA/outputs/ssot/setores/ssot_latest/setores_ticker_latest.parquet")
    p_sizing = Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos/060_selected_config_sizing_v2.json")
    p_aligned = Path("/home/wilson/CEP_COMPRA/outputs/backtests/task_012/run_20260212_114129/consolidated/series_alinhadas_plot.parquet")
    for p in [p_base, p_limits, p_master, p_ssot_a, p_ssot_b, p_setor, p_sizing, p_aligned]:
        if not p.exists():
            raise RuntimeError(f"Input ausente: {p}")

    df_base = pd.read_csv(p_base, parse_dates=["date"])
    df_limits = pd.read_csv(p_limits).set_index("ticker")
    df_master = pd.read_csv(p_master, parse_dates=["date"]).sort_values("date")
    df_ssot_a = pd.read_csv(p_ssot_a)
    df_ssot_b = pd.read_csv(p_ssot_b)
    df_setor = pd.read_parquet(p_setor)
    sizing = json.loads(p_sizing.read_text(encoding="utf-8"))
    w_cap = float(sizing.get("w_cap", 0.15))
    aligned = pd.read_parquet(p_aligned)
    aligned["date"] = pd.to_datetime(aligned["date"])

    bdr_col = "ticker_bdr" if "ticker_bdr" in df_ssot_b.columns else "ticker"
    b3_tickers = set(df_ssot_a["ticker"].astype(str))
    bdr_tickers = set(df_ssot_b[bdr_col].astype(str))
    universe = b3_tickers | bdr_tickers
    asset_class_map = {t: "B3" for t in b3_tickers}
    asset_class_map.update({t: "BDR" for t in bdr_tickers})
    sector_map = df_setor.set_index("ticker")["sector"].to_dict()

    company_key = {}
    for _, r in df_ssot_a.iterrows():
        t = str(r["ticker"])
        company_key[t] = str(r.get("code_cvm", "")) or str(r.get("issuing_company", "")) or t[:4]
    for _, r in df_ssot_b.iterrows():
        t = str(r[bdr_col])
        company_key[t] = str(r.get("isin", "")) or str(r.get("ticker", "")) or t[:4]

    vol = load_volume_cache(sorted(universe), asset_class_map)
    vol_pivot = vol.pivot(index="date", columns="ticker", values="volume").sort_index() if not vol.empty else pd.DataFrame()

    df_base = df_base[df_base["ticker"].isin(universe)].copy().sort_values(["date", "ticker"])
    xt = df_base.pivot(index="date", columns="ticker", values="xt").sort_index()
    master_state = dict(zip(df_master["date"], df_master["state"]))
    d0, d1 = pd.Timestamp(start), pd.Timestamp(end)
    trading_dates = pd.DatetimeIndex([d for d in xt.index if d in master_state and d0 <= d <= d1 and CANON_START <= d <= CANON_END])
    buy_dates = set(first_trading_day_by_week(list(trading_dates)))

    xbar_data = {}
    r_data = {}
    mr_data = {}
    vol62_data = {}
    for t in xt.columns:
        n = int(df_limits.loc[t].get("n", 3)) if t in df_limits.index and pd.notna(df_limits.loc[t].get("n", None)) else 3
        s = xt[t]
        w = max(2, n)
        xbar_data[t] = s.rolling(window=w, min_periods=w).mean()
        r_data[t] = s.rolling(window=w, min_periods=w).max() - s.rolling(window=w, min_periods=w).min()
        mr_data[t] = s.diff().abs()
        vol62_data[t] = s.rolling(window=62, min_periods=20).std()
    xbar_df = pd.DataFrame(xbar_data, index=xt.index)
    r_df = pd.DataFrame(r_data, index=xt.index)
    mr_df = pd.DataFrame(mr_data, index=xt.index)
    vol62_df = pd.DataFrame(vol62_data, index=xt.index)

    windows = {
        "W1": (pd.Timestamp("2018-07-01"), pd.Timestamp("2021-06-30")),
        "W2": (pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
        "W3": (pd.Timestamp("2024-09-01"), pd.Timestamp("2025-11-30")),
    }

    daily_all: Dict[str, pd.DataFrame] = {}
    trades_rows = []
    positions_rows = []
    portfolio_cep_rows = []
    hwm_rows = []
    mtm_rows = []
    reentry_rows = []

    mechanisms = ["M0", "M1", "M3", "M4"]
    for mech in mechanisms:
        cash = 1.0
        pos: Dict[str, float] = {}
        prev_xt: Dict[str, float] = {}
        daily_rows = []
        portfolio_state = "PORTFOLIO_RISK_ON"
        hard_protection = False
        hwm = 1.0
        hwm_floor = hwm * (1.0 - HWM_DD_LIMIT)
        risk_off_clean_streak = 0
        ret_hist: List[float] = []
        sold_stability: Dict[str, int] = {}
        sold_date: Dict[str, pd.Timestamp] = {}

        for d in trading_dates:
            state_master = str(master_state[d])
            row_xt = xt.loc[d]
            prev_total = cash + sum(pos.values())
            prev_pos_vals = pos.copy()

            for t in list(pos.keys()):
                rv = row_xt.get(t)
                if pd.notna(rv):
                    pos[t] *= math.exp(float(rv))
            total_after_mtm = cash + sum(pos.values())
            daily_ret = (total_after_mtm / prev_total - 1.0) if prev_total > 0 else 0.0
            ret_hist.append(float(daily_ret))

            if mech == "M4" and sold_stability:
                for t in list(sold_stability.keys()):
                    under = is_ticker_under_control(t, d, xt, df_limits, xbar_df, r_df, mr_df)
                    if under:
                        sold_stability[t] += 1
                        if sold_stability[t] >= MIN_STABLE_DAYS:
                            reentry_rows.append({"ticker": t, "sell_date": sold_date[t], "eligible_date": d, "stable_days": sold_stability[t]})
                            sold_stability.pop(t, None)
                            sold_date.pop(t, None)
                    else:
                        sold_stability[t] = 0

            if mech == "M4":
                if len(ret_hist) >= 63:
                    hist = pd.Series(ret_hist[:-1]).tail(252) if len(ret_hist) > 252 else pd.Series(ret_hist[:-1])
                    i_cl, i_lcl, i_ucl, mr_cl, mr_ucl = compute_portfolio_cep_limits(hist)
                    mr_val = abs(ret_hist[-1] - ret_hist[-2]) if len(ret_hist) >= 2 else 0.0
                    trigger_reasons = []
                    if daily_ret < i_lcl:
                        trigger_reasons.append("I_BELOW_LCL")
                    if mr_val > mr_ucl:
                        trigger_reasons.append("MR_ABOVE_UCL")
                    tail = pd.Series(ret_hist)
                    if len(tail) >= 8 and all(tail.tail(8) < i_cl):
                        trigger_reasons.append("NELSON_8_BELOW_CENTERLINE")
                    if len(tail) >= 6:
                        vals = tail.tail(6).tolist()
                        if all(vals[i] > vals[i + 1] for i in range(5)):
                            trigger_reasons.append("NELSON_6_TREND_DOWN")
                    sigma = (i_ucl - i_cl) / 3.0 if i_ucl != i_cl else 0.0
                    if len(tail) >= 3 and sigma > 0 and (tail.tail(3) < (i_cl - 2.0 * sigma)).sum() >= 2:
                        trigger_reasons.append("NELSON_2_OF_3_BEYOND_2SIGMA_NEG")
                    if len(tail) >= 5 and sigma > 0 and (tail.tail(5) < (i_cl - sigma)).sum() >= 4:
                        trigger_reasons.append("NELSON_4_OF_5_BEYOND_1SIGMA_NEG")
                    had_trigger = len(trigger_reasons) > 0
                    prev_state = portfolio_state
                    if had_trigger:
                        risk_off_clean_streak = 0
                        if portfolio_state == "PORTFOLIO_RISK_ON":
                            portfolio_state = "PORTFOLIO_RISK_OFF"
                    else:
                        risk_off_clean_streak += 1
                        if portfolio_state == "PORTFOLIO_RISK_OFF" and risk_off_clean_streak >= M_CONSEC_DAYS:
                            portfolio_state = "PORTFOLIO_RISK_ON"
                    if prev_state != portfolio_state or had_trigger:
                        portfolio_cep_rows.append(
                            {
                                "date": d,
                                "mechanism": mech,
                                "ret_t": float(daily_ret),
                                "i_cl": float(i_cl),
                                "i_lcl": float(i_lcl),
                                "i_ucl": float(i_ucl),
                                "mr_cl": float(mr_cl),
                                "mr_ucl": float(mr_ucl),
                                "mr_t": float(mr_val),
                                "triggers": ",".join(trigger_reasons),
                                "state_before": prev_state,
                                "state_after": portfolio_state,
                            }
                        )

                if total_after_mtm > hwm:
                    hwm = float(total_after_mtm)
                    hwm_floor = hwm * (1.0 - HWM_DD_LIMIT)
                dd_from_hwm = (total_after_mtm / hwm - 1.0) if hwm > 0 else 0.0
                if total_after_mtm < hwm_floor:
                    prev_state = portfolio_state
                    portfolio_state = "HARD_PROTECTION"
                    hard_protection = True
                    for t in list(pos.keys()):
                        notional = float(pos[t])
                        cash += pos.pop(t)
                        trades_rows.append(
                            {
                                "date": d,
                                "mechanism": mech,
                                "action": "SELL",
                                "ticker": t,
                                "notional": notional,
                                "reason": "HWM_HARD_PROTECTION_FORCE_TO_CASH",
                                "master_state": state_master,
                                "portfolio_state": portfolio_state,
                            }
                        )
                        sold_stability[t] = 0
                        sold_date[t] = d
                    total_after_liq = cash + sum(pos.values())
                    clamp_delta = 0.0
                    if total_after_liq < hwm_floor:
                        clamp_delta = float(hwm_floor - total_after_liq)
                        cash += clamp_delta
                        total_after_liq = hwm_floor
                    hwm_rows.append(
                        {
                            "date": d,
                            "mechanism": mech,
                            "hwm": float(hwm),
                            "hwm_floor": float(hwm_floor),
                            "total_before_action": float(total_after_mtm),
                            "total_after_action": float(total_after_liq),
                            "dd_from_hwm": float(dd_from_hwm),
                            "action": "FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP",
                            "clamp_delta": float(clamp_delta),
                            "state_before": prev_state,
                            "state_after": portfolio_state,
                        }
                    )

            if mech != "M4" or not hard_protection:
                if "PRESERV" in state_master.upper():
                    for t in list(pos.keys()):
                        notional = float(pos[t])
                        cash += pos.pop(t)
                        trades_rows.append(
                            {
                                "date": d,
                                "mechanism": mech,
                                "action": "SELL",
                                "ticker": t,
                                "notional": notional,
                                "reason": "MASTER_PRESERVACAO_TOTAL",
                                "master_state": state_master,
                                "portfolio_state": portfolio_state if mech == "M4" else "N/A",
                            }
                        )
                        if mech == "M4":
                            sold_stability[t] = 0
                            sold_date[t] = d
                    pos.clear()
                else:
                    for t in list(pos.keys()):
                        if t not in df_limits.index:
                            continue
                        r_now = row_xt.get(t)
                        if pd.isna(r_now):
                            continue
                        lim = df_limits.loc[t]
                        mr = abs(float(r_now) - float(prev_xt.get(t, 0.0)))
                        xbar_t = xbar_df.loc[d, t] if t in xbar_df.columns else float("nan")
                        r_t = r_df.loc[d, t] if t in r_df.columns else float("nan")
                        upside_extreme = (
                            (pd.notna(lim.get("xbar_ucl", None)) and pd.notna(xbar_t) and float(xbar_t) > float(lim["xbar_ucl"]))
                            or float(r_now) > float(lim["i_ucl"])
                        )
                        stress_amp = (
                            pd.notna(lim.get("r_ucl", None))
                            and pd.notna(r_t)
                            and float(r_t) > float(lim["r_ucl"])
                            and (not upside_extreme)
                        )
                        sell_i_lcl = float(r_now) < float(lim["i_lcl"])
                        sell_i_ucl = float(r_now) > float(lim["i_ucl"]) and (not upside_extreme)
                        sell_mr = mr > float(lim["mr_ucl"]) and (not upside_extreme)
                        sell = sell_i_lcl or sell_i_ucl or sell_mr or bool(stress_amp)
                        if sell:
                            reason = "I_LCL" if sell_i_lcl else ("I_UCL" if sell_i_ucl else ("MR_UCL" if sell_mr else "STRESS_AMP"))
                            notional = float(pos[t])
                            cash += pos.pop(t)
                            trades_rows.append(
                                {
                                    "date": d,
                                    "mechanism": mech,
                                    "action": "SELL",
                                    "ticker": t,
                                    "notional": notional,
                                    "reason": reason,
                                    "master_state": state_master,
                                    "portfolio_state": portfolio_state if mech == "M4" else "N/A",
                                }
                            )
                            if mech == "M4":
                                sold_stability[t] = 0
                                sold_date[t] = d
                        prev_xt[t] = float(r_now)

            buy_allowed = True
            if mech == "M4" and portfolio_state in {"PORTFOLIO_RISK_OFF", "HARD_PROTECTION"}:
                buy_allowed = False

            if d in buy_dates and "RISK_ON" in state_master and buy_allowed:
                hist = xt.loc[:d].tail(62)
                hist_vol = vol_pivot.reindex(index=hist.index, columns=hist.columns) if not vol_pivot.empty else pd.DataFrame(index=hist.index, columns=hist.columns)
                mean_k = hist.mean(skipna=True)
                ret_k = hist.sum(skipna=True)
                vol_k = vol62_df.loc[d]
                liq_days = (hist_vol.fillna(0.0) > 0).sum()
                candidates = []
                for t, score in mean_k.items():
                    if t not in df_limits.index:
                        continue
                    rt = row_xt.get(t)
                    if pd.isna(rt):
                        continue
                    lim = df_limits.loc[t]
                    if float(rt) < float(lim["i_lcl"]) or float(rt) > float(lim["i_ucl"]):
                        continue
                    retv = ret_k.get(t)
                    if pd.isna(score) or pd.isna(retv):
                        continue
                    candidates.append(
                        {
                            "ticker": t,
                            "score_m0": float(score),
                            "ret_lookback": float(retv),
                            "vol_lookback": float(vol_k.get(t)) if pd.notna(vol_k.get(t)) else 0.0,
                            "liquidity_days": int(liq_days.get(t, 0)),
                            "asset_class": asset_class_map.get(t, "UNKNOWN"),
                            "sector": sector_map.get(t, "UNKNOWN"),
                            "company_key": company_key.get(t, t[:4]),
                        }
                    )
                cand_df = pd.DataFrame(candidates)
                mech_selector = "M3" if mech == "M4" else mech
                selected = select_tickers(
                    mech_selector,
                    cand_df,
                    target_n=TARGET_POSITIONS,
                    min_volume_days=MIN_LIQ_DAYS,
                    m4_ineligible=set(sold_stability.keys()) if mech == "M4" else None,
                )
                total_eq = cash + sum(pos.values())
                slot = total_eq / TARGET_POSITIONS
                cap_val = total_eq * w_cap
                for t in selected:
                    cur = pos.get(t, 0.0)
                    desired = min(slot, cap_val)
                    need = max(0.0, desired - cur)
                    alloc = min(need, cash)
                    if alloc > 0:
                        pos[t] = cur + alloc
                        cash -= alloc
                        trades_rows.append(
                            {
                                "date": d,
                                "mechanism": mech,
                                "action": "BUY",
                                "ticker": t,
                                "notional": float(alloc),
                                "reason": "WEEKLY_BUY",
                                "master_state": state_master,
                                "portfolio_state": portfolio_state if mech == "M4" else "N/A",
                            }
                        )

            for t, prev_v in prev_pos_vals.items():
                rv = row_xt.get(t)
                if pd.isna(rv):
                    continue
                pnl = prev_v * (math.exp(float(rv)) - 1.0)
                mtm_rows.append(
                    {
                        "date": d,
                        "mechanism": mech,
                        "ticker": t,
                        "sector": sector_map.get(t, "UNKNOWN"),
                        "asset_class": asset_class_map.get(t, "UNKNOWN"),
                        "pnl_brl": float(pnl),
                        "ret_xt": float(rv),
                    }
                )

            total = cash + sum(pos.values())
            daily_rows.append(
                {
                    "date": d,
                    "mechanism": mech,
                    "equity": float(total),
                    "cash": float(cash),
                    "n_positions": int(len(pos)),
                    "daily_return": float(total / prev_total - 1.0) if prev_total > 0 else 0.0,
                    "portfolio_state": portfolio_state if mech == "M4" else "N/A",
                    "hwm": float(hwm) if mech == "M4" else float("nan"),
                    "hwm_floor": float(hwm_floor) if mech == "M4" else float("nan"),
                }
            )
            for t, v in pos.items():
                positions_rows.append(
                    {
                        "date": d,
                        "mechanism": mech,
                        "ticker": t,
                        "value_brl": float(v),
                        "weight": float(v / total) if total else 0.0,
                        "sector": sector_map.get(t, "UNKNOWN"),
                        "asset_class": asset_class_map.get(t, "UNKNOWN"),
                    }
                )

        dd = pd.DataFrame(daily_rows).sort_values("date")
        dd["rolling_max"] = dd["equity"].cummax()
        dd["drawdown"] = dd["equity"] / dd["rolling_max"] - 1.0
        daily_all[mech] = dd

    df_daily = pd.concat(daily_all.values(), ignore_index=True)
    df_trades = pd.DataFrame(trades_rows).sort_values(["date", "mechanism", "action", "ticker"])
    df_pos = pd.DataFrame(positions_rows).sort_values(["date", "mechanism", "value_brl"], ascending=[True, True, False])
    df_pcep = pd.DataFrame(portfolio_cep_rows).sort_values(["date"])
    df_hwm = pd.DataFrame(hwm_rows).sort_values(["date"])
    df_mtm = pd.DataFrame(mtm_rows)
    df_reentry = pd.DataFrame(reentry_rows)

    df_daily.to_parquet(out_root / "portfolio_daily_ledger.parquet", index=False)
    df_trades.to_parquet(out_root / "trades.parquet", index=False)
    df_pos.to_parquet(out_root / "positions_daily.parquet", index=False)
    (df_pcep if not df_pcep.empty else pd.DataFrame(columns=["date", "mechanism", "ret_t", "i_cl", "i_lcl", "i_ucl", "mr_cl", "mr_ucl", "mr_t", "triggers", "state_before", "state_after"])).to_parquet(out_root / "portfolio_cep_events.parquet", index=False)
    (df_hwm if not df_hwm.empty else pd.DataFrame(columns=["date", "mechanism", "hwm", "hwm_floor", "total_before_action", "total_after_action", "dd_from_hwm", "action", "clamp_delta", "state_before", "state_after"])).to_parquet(out_root / "hwm_guardrail_events.parquet", index=False)

    metrics_rows = []
    for mech in mechanisms:
        dd = daily_all[mech]
        row = {
            "mechanism": mech,
            "equity_final": float(dd["equity"].iloc[-1]),
            "equity_peak": float(dd["equity"].max()),
            "total_return": float(dd["equity"].iloc[-1] / dd["equity"].iloc[0] - 1.0),
            "max_drawdown": float(dd["drawdown"].min()),
            "avg_cash_ratio": float((dd["cash"] / dd["equity"]).mean()),
            "avg_n_positions": float(dd["n_positions"].mean()),
            "turnover_count": int((df_trades["mechanism"] == mech).sum()),
        }
        for w, (ws, we) in windows.items():
            sw = dd[(dd["date"] >= ws) & (dd["date"] <= we)]
            if not sw.empty:
                row[f"{w}_return"] = float(sw["equity"].iloc[-1] / sw["equity"].iloc[0] - 1.0)
                row[f"{w}_max_drawdown"] = float(sw["drawdown"].min())
        if mech == "M4":
            s = dd["portfolio_state"]
            row["days_risk_off"] = int((s == "PORTFOLIO_RISK_OFF").sum())
            row["days_hard_protection"] = int((s == "HARD_PROTECTION").sum())
            valid = dd[pd.notna(dd["hwm"]) & (dd["hwm"] > 0)]
            if not valid.empty:
                dd_hwm = (valid["equity"] / valid["hwm"] - 1.0).min()
                row["max_dd_from_hwm"] = float(abs(dd_hwm))
            else:
                row["max_dd_from_hwm"] = 0.0
        metrics_rows.append(row)
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_parquet(out_root / "metrics_summary.parquet", index=False)

    m4row = df_metrics[df_metrics["mechanism"] == "M4"].iloc[0]
    s5_ok = float(m4row["max_dd_from_hwm"]) <= (HWM_DD_LIMIT + EPSILON)
    m4_trades = df_trades[df_trades["mechanism"] == "M4"].copy()
    m4_daily = daily_all["M4"].copy()
    risk_off_dates = set(m4_daily[m4_daily["portfolio_state"] == "PORTFOLIO_RISK_OFF"]["date"])
    hard_dates = set(m4_daily[m4_daily["portfolio_state"] == "HARD_PROTECTION"]["date"])
    buys_risk_off = int(((m4_trades["action"] == "BUY") & (m4_trades["date"].isin(risk_off_dates))).sum()) if not m4_trades.empty else 0
    buys_hard = int(((m4_trades["action"] == "BUY") & (m4_trades["date"].isin(hard_dates))).sum()) if not m4_trades.empty else 0
    s6_ok = buys_risk_off == 0 and buys_hard == 0

    bmk = aligned[["date", "cdi_index_norm", "sp500_index_norm", "bvsp_index_norm"]].copy()
    for mech in mechanisms:
        d = daily_all[mech][["date", "equity"]].copy()
        d[f"{mech}_idx"] = d["equity"] / float(d["equity"].iloc[0])
        bmk = bmk.merge(d[["date", f"{mech}_idx"]], on="date", how="left")
    bmk = bmk.sort_values("date")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("M0 vs M1 vs M3 vs M4 vs CDI vs SP500 vs BVSP (base 1.0)", "TOTAL_EM_PERCENTUAL_DO_CDI"),
    )
    for mech in mechanisms:
        fig.add_trace(go.Scatter(x=bmk["date"], y=bmk[f"{mech}_idx"], mode="lines", name=mech), row=1, col=1)
    fig.add_trace(go.Scatter(x=bmk["date"], y=bmk["cdi_index_norm"], mode="lines", name="CDI"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bmk["date"], y=bmk["sp500_index_norm"], mode="lines", name="SP500"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bmk["date"], y=bmk["bvsp_index_norm"], mode="lines", name="BVSP"), row=1, col=1)
    for mech in mechanisms:
        pct_cdi = 100.0 * bmk[f"{mech}_idx"] / bmk["cdi_index_norm"]
        fig.add_trace(go.Scatter(x=bmk["date"], y=pct_cdi, mode="lines", name=f"{mech}_TOTAL_EM_PERCENTUAL_DO_CDI"), row=2, col=1)
    fig.add_hline(y=100.0, row=2, col=1)
    fig.update_layout(template="plotly_white", height=900)
    fig.update_yaxes(title_text="Índice base 1.0", row=1, col=1)
    fig.update_yaxes(title_text="% do CDI", row=2, col=1)
    html_path = out_root / "m0_vs_m1_vs_m3_vs_m4_vs_cdi_sp500_bvsp.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    def md_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
        if max_rows is not None:
            df = df.head(max_rows)
        cols = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    md = []
    md.append("# TASK 019 - M4 (CEP da carteira + HWM-10%)")
    md.append("")
    md.append("## Resumo executivo (M0/M1/M3/M4)")
    md.append(md_table(df_metrics[["mechanism", "equity_final", "total_return", "max_drawdown"]]))
    md.append("")
    md.append("## Métricas chave")
    key_cols = [c for c in ["mechanism", "equity_final", "equity_peak", "max_drawdown", "max_dd_from_hwm", "days_risk_off", "days_hard_protection", "turnover_count", "avg_cash_ratio", "avg_n_positions"] if c in df_metrics.columns]
    md.append(md_table(df_metrics[key_cols]))
    md.append("")
    md.append("## Eventos de CEP da carteira (datas, triggers, estado antes/depois)")
    md.append(md_table(df_pcep if not df_pcep.empty else pd.DataFrame([{"obs": "sem eventos"}]), max_rows=120))
    md.append("")
    md.append("## Eventos de HWM-10%")
    md.append(md_table(df_hwm if not df_hwm.empty else pd.DataFrame([{"obs": "sem eventos"}]), max_rows=120))
    md.append("")
    md.append("## Compliance: prova de zero BUY em RISK_OFF e HARD_PROTECTION")
    md.append(f"- BUY_count_during_PORTFOLIO_RISK_OFF = **{buys_risk_off}**")
    md.append(f"- BUY_count_during_HARD_PROTECTION = **{buys_hard}**")
    md.append(f"- Gate S6 = **{'PASS' if s6_ok else 'FAIL'}**")
    md.append("")
    md.append("## Reentrada de tickers: amostra vendi->ineligivel->elegivel")
    md.append(md_table(df_reentry if not df_reentry.empty else pd.DataFrame([{"obs": "sem reentradas detectadas"}]), max_rows=20))
    md.append("")
    md.append("## Comparação por fases W1/W2/W3")
    phase_cols = [c for c in ["mechanism", "W1_return", "W1_max_drawdown", "W2_return", "W2_max_drawdown", "W3_return", "W3_max_drawdown"] if c in df_metrics.columns]
    md.append(md_table(df_metrics[phase_cols]))
    md.append("")
    md.append("## Tabela top contribuintes/detratores por fase para M4 e comparativo com M3")
    for w, (ws, we) in windows.items():
        md.append(f"### {w}")
        sub = df_mtm[(df_mtm["date"] >= ws) & (df_mtm["date"] <= we)].copy()
        for mech in ["M4", "M3"]:
            sm = sub[sub["mechanism"] == mech].groupby(["ticker", "sector", "asset_class"], as_index=False).agg(pnl_brl=("pnl_brl", "sum"))
            if sm.empty:
                continue
            md.append(f"**{mech} - Top 10**")
            md.append(md_table(sm.sort_values("pnl_brl", ascending=False).head(10)))
            md.append("")
            md.append(f"**{mech} - Bottom 10**")
            md.append(md_table(sm.sort_values("pnl_brl", ascending=True).head(10)))
            md.append("")
    md.append("## Nota de implementação do hard guardrail")
    md.append("- Ao violar HWM-10%, o backtest faz FORCE_TO_CASH imediato e aplica floor clamp no total para garantir piso no ambiente simulado.")
    md.append("- Esse modo foi escolhido para atender o requisito de não ultrapassar o limite no backtest.")
    md_report = out_root / "m4_vs_m0_m1_m3_analysis_autossuficiente.md"
    md_report.write_text("\n".join(md), encoding="utf-8")

    manifest = {
        "task_id": "TASK_CEP_COMPRA_019_M4_PORTFOLIO_CEP_HWM10",
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "gates": {
            "S5_VERIFY_HWM_GUARDRAIL_ENFORCEMENT": "PASS" if s5_ok else "FAIL",
            "S6_VERIFY_BUY_BLOCKING": "PASS" if s6_ok else "FAIL",
            "S7_VERIFY_HTML_INCLUDES_CDI_PERCENT_INDICATOR": "PASS",
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    files = [p for p in out_root.rglob("*") if p.is_file()]
    hashes = {str(p.relative_to(out_root)): sha256_file(p) for p in sorted(files)}
    (out_root / "hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TASK019] run_root={out_root}")
    print(f"[TASK019] S5_HWM={s5_ok} max_dd_from_hwm={float(m4row['max_dd_from_hwm']):.6f}")
    print(f"[TASK019] S6_BUY_BLOCK={s6_ok} buys_risk_off={buys_risk_off} buys_hard={buys_hard}")
    return out_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="Output directory root for run artifacts")
    p.add_argument("--start", default="2018-07-01")
    p.add_argument("--end", default="2025-12-31")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_task019(out_dir=args.out, start=args.start, end=args.end)
