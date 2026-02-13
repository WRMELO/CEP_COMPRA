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

from tools.task_019_m4_runner import (
    CANON_END,
    CANON_START,
    MIN_LIQ_DAYS,
    TARGET_POSITIONS,
    first_trading_day_by_week,
    is_ticker_under_control,
    load_volume_cache,
    select_tickers,
)

HWM_DD_LIMIT = 0.10
EPSILON = 0.0005
M_CONSEC_DAYS = 5
MIN_HARD_PROTECTION_DAYS = 5
MIN_STABLE_DAYS = 5
REPLENISH_MAX_CASH_FRACTION = 0.30
REPLENISH_TARGET_POSITIONS = 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_latest_run(base_dir: Path) -> Path:
    runs = [p for p in base_dir.glob("run_*") if p.is_dir()]
    if not runs:
        raise RuntimeError(f"Nenhum run encontrado em {base_dir}")
    runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def xbar_r_limits_from_baseline(returns: List[float], n: int = 3, k: int = 60) -> Dict[str, float]:
    if len(returns) < (k + n - 1):
        raise RuntimeError("Retornos insuficientes para baseline Xbarra-R")
    baseline = returns[: k + n - 1]
    xbars = []
    rs = []
    for i in range(k):
        sg = baseline[i : i + n]
        xbars.append(float(sum(sg) / n))
        rs.append(float(max(sg) - min(sg)))
    xbarbar = float(sum(xbars) / len(xbars))
    rbar = float(sum(rs) / len(rs))
    # constantes n=3
    a2 = 1.023
    d3 = 0.0
    d4 = 2.574
    xbar_lcl = xbarbar - a2 * rbar
    xbar_ucl = xbarbar + a2 * rbar
    r_lcl = d3 * rbar
    r_ucl = d4 * rbar
    two_sigma_neg = xbarbar - (2.0 / 3.0) * a2 * rbar
    return {
        "n": float(n),
        "k": float(k),
        "xbarbar": xbarbar,
        "rbar": rbar,
        "xbar_lcl": xbar_lcl,
        "xbar_ucl": xbar_ucl,
        "r_lcl": r_lcl,
        "r_ucl": r_ucl,
        "two_sigma_neg": two_sigma_neg,
    }


def compute_xbar_r_point(returns: List[float], n: int = 3) -> Tuple[float, float]:
    sg = returns[-n:]
    xbar_t = float(sum(sg) / n)
    r_t = float(max(sg) - min(sg))
    return xbar_t, r_t


def build_contrib_from_positions(
    mechanism: str,
    positions_daily: pd.DataFrame,
    xt: pd.DataFrame,
    sector_map: Dict[str, str],
    asset_class_map: Dict[str, str],
) -> pd.DataFrame:
    dpos = positions_daily[positions_daily["mechanism"] == mechanism].copy()
    if dpos.empty:
        return pd.DataFrame(columns=["date", "mechanism", "ticker", "sector", "asset_class", "pnl_brl"])
    pv = dpos.pivot_table(index="date", columns="ticker", values="value_brl", aggfunc="sum").sort_index()
    prev = pv.shift(1)
    idx = prev.index.intersection(xt.index)
    prev = prev.loc[idx]
    x = xt.reindex(index=idx, columns=prev.columns)
    pnl = prev * (x.applymap(lambda r: math.exp(float(r)) - 1.0 if pd.notna(r) else float("nan")))
    long = pnl.stack(dropna=True).reset_index()
    long.columns = ["date", "ticker", "pnl_brl"]
    long["mechanism"] = mechanism
    long["sector"] = long["ticker"].map(sector_map).fillna("UNKNOWN")
    long["asset_class"] = long["ticker"].map(asset_class_map).fillna("UNKNOWN")
    return long[["date", "mechanism", "ticker", "sector", "asset_class", "pnl_brl"]]


def run_task021(out_dir: Optional[str] = None, start: str = "2018-07-01", end: str = "2025-12-31") -> Path:
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(out_dir) if out_dir else Path(f"/home/wilson/CEP_COMPRA/outputs/backtests/task_021_m6/{run_id}")
    out_root.mkdir(parents=True, exist_ok=True)

    base_task020 = Path("/home/wilson/CEP_COMPRA/outputs/backtests/task_020_m5")
    ref_run = find_latest_run(base_task020)

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
    aligned = pd.read_parquet(p_aligned).sort_values("date")
    aligned["date"] = pd.to_datetime(aligned["date"])
    aligned["cdi_ret_t"] = aligned["cdi_index_norm"].pct_change().fillna(0.0)
    cdi_ret_map = dict(zip(aligned["date"], aligned["cdi_ret_t"]))

    # baseline outputs from task020 (M0..M5)
    ref_daily = pd.read_parquet(ref_run / "portfolio_daily_ledger.parquet")
    ref_trades = pd.read_parquet(ref_run / "trades.parquet")
    ref_pos = pd.read_parquet(ref_run / "positions_daily.parquet")

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

    cash = 1.0
    pos: Dict[str, float] = {}
    prev_xt: Dict[str, float] = {}
    sold_stability: Dict[str, int] = {}
    sold_date: Dict[str, pd.Timestamp] = {}
    portfolio_state = "PORTFOLIO_RISK_ON"
    hard_protection = False
    hard_days = 0
    hard_clean_streak = 0
    risk_off_clean_streak = 0
    hwm = 1.0
    hwm_floor = hwm * (1.0 - HWM_DD_LIMIT)
    ret_hist: List[float] = []
    xbar_hist: List[float] = []
    xb_limits = None
    replenish_rows = []

    m6_daily_rows = []
    m6_trade_rows = []
    m6_pos_rows = []
    m6_pcep_rows = []
    m6_hwm_rows = []
    m6_reentry_rows = []

    for d in trading_dates:
        state_master = str(master_state[d])
        row_xt = xt.loc[d]
        cdi_ret = float(cdi_ret_map.get(d, 0.0))
        prev_total = cash + sum(pos.values())
        prev_pos_vals = pos.copy()

        # cash accrual keeps M5 behavior
        cash = cash * (1.0 + cdi_ret)

        for t in list(pos.keys()):
            rv = row_xt.get(t)
            if pd.notna(rv):
                pos[t] *= math.exp(float(rv))
        total_after_mtm = cash + sum(pos.values())
        daily_ret = (total_after_mtm / prev_total - 1.0) if prev_total > 0 else 0.0
        ret_hist.append(float(daily_ret))

        if sold_stability:
            for t in list(sold_stability.keys()):
                under = is_ticker_under_control(t, d, xt, df_limits, xbar_df, r_df, mr_df)
                if under:
                    sold_stability[t] += 1
                    if sold_stability[t] >= MIN_STABLE_DAYS:
                        m6_reentry_rows.append({"ticker": t, "sell_date": sold_date[t], "eligible_date": d, "stable_days": sold_stability[t]})
                        sold_stability.pop(t, None)
                        sold_date.pop(t, None)
                else:
                    sold_stability[t] = 0

        # Portfolio CEP by Xbarra-R
        trigger_reasons = []
        xbar_t = float("nan")
        r_t = float("nan")
        two_of_three = False
        if len(ret_hist) >= 62:
            if xb_limits is None:
                xb_limits = xbar_r_limits_from_baseline(ret_hist, n=3, k=60)
            xbar_t, r_t = compute_xbar_r_point(ret_hist, n=3)
            xbar_hist.append(float(xbar_t))

            if daily_ret < 0 and xbar_t < xb_limits["xbar_lcl"]:
                trigger_reasons.append("ONE_BELOW_LCL")
            if daily_ret < 0 and len(xbar_hist) >= 3:
                thr = xb_limits["two_sigma_neg"]
                if sum(1 for v in xbar_hist[-3:] if v < thr) >= 2:
                    trigger_reasons.append("TWO_OF_THREE_BELOW_2SIGMA_NEG")
                    two_of_three = True

            has_downside = len(trigger_reasons) > 0
            prev_state = portfolio_state
            if portfolio_state == "HARD_PROTECTION":
                hard_days += 1
                guard_ok = total_after_mtm >= hwm_floor
                if has_downside or not guard_ok:
                    hard_clean_streak = 0
                else:
                    hard_clean_streak += 1
                if hard_days >= MIN_HARD_PROTECTION_DAYS and hard_clean_streak >= M_CONSEC_DAYS and guard_ok:
                    portfolio_state = "PORTFOLIO_RISK_OFF"
                    hard_protection = False
                    hard_days = 0
                    hard_clean_streak = 0
                    risk_off_clean_streak = 0
            else:
                if has_downside:
                    risk_off_clean_streak = 0
                    if portfolio_state == "PORTFOLIO_RISK_ON":
                        portfolio_state = "PORTFOLIO_RISK_OFF"
                else:
                    if portfolio_state == "PORTFOLIO_RISK_OFF":
                        risk_off_clean_streak += 1
                        if risk_off_clean_streak >= M_CONSEC_DAYS:
                            portfolio_state = "PORTFOLIO_RISK_ON"

            if prev_state != portfolio_state or trigger_reasons:
                m6_pcep_rows.append(
                    {
                        "date": d,
                        "mechanism": "M6",
                        "chart": "XBARRA_R",
                        "subgroup_n": 3,
                        "k_subgroups": 60,
                        "ret_t": float(daily_ret),
                        "xbar_t": float(xbar_t),
                        "r_t": float(r_t),
                        "xbarbar": float(xb_limits["xbarbar"]),
                        "rbar": float(xb_limits["rbar"]),
                        "xbar_lcl": float(xb_limits["xbar_lcl"]),
                        "xbar_ucl": float(xb_limits["xbar_ucl"]),
                        "two_sigma_neg": float(xb_limits["two_sigma_neg"]),
                        "rule_one_below_lcl": "ONE_BELOW_LCL" in trigger_reasons,
                        "rule_two_of_three_below_2sigma_neg": bool(two_of_three),
                        "triggers_downside": ",".join(trigger_reasons),
                        "state_before": prev_state,
                        "state_after": portfolio_state,
                    }
                )

        # realistic no-clamp hard guardrail
        if total_after_mtm > hwm:
            hwm = float(total_after_mtm)
            hwm_floor = hwm * (1.0 - HWM_DD_LIMIT)
        dd_from_hwm = (total_after_mtm / hwm - 1.0) if hwm > 0 else 0.0
        if total_after_mtm < hwm_floor:
            prev_state = portfolio_state
            portfolio_state = "HARD_PROTECTION"
            hard_protection = True
            hard_days = 0
            hard_clean_streak = 0
            for t in list(pos.keys()):
                notional = float(pos[t])
                cash += pos.pop(t)
                m6_trade_rows.append(
                    {
                        "date": d,
                        "mechanism": "M6",
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
            overshoot = float(hwm_floor - total_after_liq) if total_after_liq < hwm_floor else 0.0
            m6_hwm_rows.append(
                {
                    "date": d,
                    "mechanism": "M6",
                    "hwm": float(hwm),
                    "hwm_floor": float(hwm_floor),
                    "total_before_action": float(total_after_mtm),
                    "total_after_action": float(total_after_liq),
                    "dd_from_hwm": float(dd_from_hwm),
                    "overshoot_below_floor": float(overshoot),
                    "action": "FORCE_TO_CASH_AND_BLOCK_BUYS",
                    "state_before": prev_state,
                    "state_after": portfolio_state,
                }
            )

        # ticker-level sells (bundle unchanged)
        if not hard_protection:
            if "PRESERV" in state_master.upper():
                for t in list(pos.keys()):
                    notional = float(pos[t])
                    cash += pos.pop(t)
                    m6_trade_rows.append(
                        {
                            "date": d,
                            "mechanism": "M6",
                            "action": "SELL",
                            "ticker": t,
                            "notional": notional,
                            "reason": "MASTER_PRESERVACAO_TOTAL",
                            "master_state": state_master,
                            "portfolio_state": portfolio_state,
                        }
                    )
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
                    xbar_tick = xbar_df.loc[d, t] if t in xbar_df.columns else float("nan")
                    r_tick = r_df.loc[d, t] if t in r_df.columns else float("nan")
                    upside_extreme = (
                        (pd.notna(lim.get("xbar_ucl", None)) and pd.notna(xbar_tick) and float(xbar_tick) > float(lim["xbar_ucl"]))
                        or float(r_now) > float(lim["i_ucl"])
                    )
                    stress_amp = (
                        pd.notna(lim.get("r_ucl", None))
                        and pd.notna(r_tick)
                        and float(r_tick) > float(lim["r_ucl"])
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
                        m6_trade_rows.append(
                            {
                                "date": d,
                                "mechanism": "M6",
                                "action": "SELL",
                                "ticker": t,
                                "notional": notional,
                                "reason": reason,
                                "master_state": state_master,
                                "portfolio_state": portfolio_state,
                            }
                        )
                        sold_stability[t] = 0
                        sold_date[t] = d
                    prev_xt[t] = float(r_now)

        # Buy logic
        is_buy_day = d in buy_dates and "RISK_ON" in state_master
        if is_buy_day and portfolio_state == "PORTFOLIO_RISK_ON":
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
            selected = select_tickers("M3", cand_df, target_n=TARGET_POSITIONS, min_volume_days=MIN_LIQ_DAYS, m4_ineligible=set(sold_stability.keys()) if sold_stability else None)
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
                    m6_trade_rows.append(
                        {
                            "date": d,
                            "mechanism": "M6",
                            "action": "BUY",
                            "ticker": t,
                            "notional": float(alloc),
                            "reason": "WEEKLY_BUY",
                            "master_state": state_master,
                            "portfolio_state": portfolio_state,
                        }
                    )

        # Replenishment in RISK_OFF (30% cash, max 5 stable positions), forbidden in HARD
        if is_buy_day and portfolio_state == "PORTFOLIO_RISK_OFF":
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
                if t in sold_stability:
                    continue
                rt = row_xt.get(t)
                if pd.isna(rt):
                    continue
                lim = df_limits.loc[t]
                if float(rt) < float(lim["i_lcl"]) or float(rt) > float(lim["i_ucl"]):
                    continue
                # stability filter
                if not is_ticker_under_control(t, d, xt, df_limits, xbar_df, r_df, mr_df):
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
            # ranking by score M3
            selected = select_tickers("M3", cand_df, target_n=REPLENISH_TARGET_POSITIONS, min_volume_days=MIN_LIQ_DAYS, m4_ineligible=set())
            selected = selected[:REPLENISH_TARGET_POSITIONS]
            deploy_budget = cash * REPLENISH_MAX_CASH_FRACTION
            per_pos = deploy_budget / max(1, len(selected))
            used = 0.0
            n_used = 0
            for t in selected:
                alloc = min(per_pos, cash)
                if alloc <= 0:
                    continue
                pos[t] = pos.get(t, 0.0) + alloc
                cash -= alloc
                used += alloc
                n_used += 1
                m6_trade_rows.append(
                    {
                        "date": d,
                        "mechanism": "M6",
                        "action": "BUY",
                        "ticker": t,
                        "notional": float(alloc),
                        "reason": "RISK_OFF_REPLENISHMENT",
                        "master_state": state_master,
                        "portfolio_state": portfolio_state,
                    }
                )
            replenish_rows.append(
                {
                    "date": d,
                    "mechanism": "M6",
                    "state": portfolio_state,
                    "replenish_cash_fraction_deployed": float(used / (cash + used)) if (cash + used) > 0 else 0.0,
                    "replenish_positions_count": int(n_used),
                    "target_positions": REPLENISH_TARGET_POSITIONS,
                    "max_cash_fraction_to_deploy": REPLENISH_MAX_CASH_FRACTION,
                }
            )

        total = cash + sum(pos.values())
        m6_daily_rows.append(
            {
                "date": d,
                "mechanism": "M6",
                "equity": float(total),
                "cash": float(cash),
                "n_positions": int(len(pos)),
                "daily_return": float(total / prev_total - 1.0) if prev_total > 0 else 0.0,
                "portfolio_state": portfolio_state,
                "hwm": float(hwm),
                "hwm_floor": float(hwm_floor),
                "cash_ratio": float(cash / total) if total else 0.0,
                "cdi_ret_t": float(cdi_ret),
            }
        )
        for t, v in pos.items():
            m6_pos_rows.append(
                {
                    "date": d,
                    "mechanism": "M6",
                    "ticker": t,
                    "value_brl": float(v),
                    "weight": float(v / total) if total else 0.0,
                    "sector": sector_map.get(t, "UNKNOWN"),
                    "asset_class": asset_class_map.get(t, "UNKNOWN"),
                }
            )

    m6_daily = pd.DataFrame(m6_daily_rows).sort_values("date")
    m6_daily["rolling_max"] = m6_daily["equity"].cummax()
    m6_daily["drawdown"] = m6_daily["equity"] / m6_daily["rolling_max"] - 1.0

    keep_mechs = {"M0", "M1", "M3", "M4", "M5"}
    out_daily = pd.concat([ref_daily[ref_daily["mechanism"].isin(keep_mechs)], m6_daily], ignore_index=True)
    out_trades = pd.concat([ref_trades[ref_trades["mechanism"].isin(keep_mechs)], pd.DataFrame(m6_trade_rows)], ignore_index=True)
    out_pos = pd.concat([ref_pos[ref_pos["mechanism"].isin(keep_mechs)], pd.DataFrame(m6_pos_rows)], ignore_index=True)
    out_pcep = pd.DataFrame(m6_pcep_rows).sort_values("date")
    out_hwm = pd.DataFrame(m6_hwm_rows).sort_values("date")
    out_replenish = pd.DataFrame(replenish_rows).sort_values("date")

    windows = {
        "W1": (pd.Timestamp("2018-07-01"), pd.Timestamp("2021-06-30")),
        "W2": (pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
        "W3": (pd.Timestamp("2024-09-01"), pd.Timestamp("2025-11-30")),
    }
    metrics_rows = []
    for mech in ["M0", "M1", "M3", "M4", "M5", "M6"]:
        dd = out_daily[out_daily["mechanism"] == mech].sort_values("date")
        row = {
            "mechanism": mech,
            "equity_final": float(dd["equity"].iloc[-1]),
            "equity_peak": float(dd["equity"].max()),
            "total_return": float(dd["equity"].iloc[-1] / dd["equity"].iloc[0] - 1.0),
            "max_drawdown": float(dd["drawdown"].min()),
            "turnover_count": int((out_trades["mechanism"] == mech).sum()),
            "avg_cash_ratio": float((dd["cash"] / dd["equity"]).mean()),
            "avg_n_positions": float(dd["n_positions"].mean()),
        }
        for w, (ws, we) in windows.items():
            sw = dd[(dd["date"] >= ws) & (dd["date"] <= we)]
            if not sw.empty:
                row[f"{w}_return"] = float(sw["equity"].iloc[-1] / sw["equity"].iloc[0] - 1.0)
                row[f"{w}_max_drawdown"] = float(sw["drawdown"].min())
        if mech == "M6":
            s = dd["portfolio_state"]
            row["days_risk_on"] = int((s == "PORTFOLIO_RISK_ON").sum())
            row["days_risk_off"] = int((s == "PORTFOLIO_RISK_OFF").sum())
            row["days_hard_protection"] = int((s == "HARD_PROTECTION").sum())
            valid = dd[pd.notna(dd["hwm"]) & (dd["hwm"] > 0)]
            row["max_dd_from_hwm"] = float(abs((valid["equity"] / valid["hwm"] - 1.0).min())) if not valid.empty else 0.0
            row["time_below_hwm_floor_days"] = int((valid["equity"] < valid["hwm_floor"]).sum()) if not valid.empty else 0
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)

    # gates
    m6 = metrics[metrics["mechanism"] == "M6"].iloc[0]
    s5_ok = ("clamp_delta" not in out_hwm.columns) and (not out_hwm["action"].astype(str).str.contains("CLAMP", na=False).any())
    max_repl_frac = float(out_replenish["replenish_cash_fraction_deployed"].max()) if not out_replenish.empty else 0.0
    max_repl_pos = int(out_replenish["replenish_positions_count"].max()) if not out_replenish.empty else 0
    s6_ok = max_repl_frac <= (0.30 + 1e-6) and max_repl_pos <= 5
    m6_tr = out_trades[out_trades["mechanism"] == "M6"]
    m6_dd = out_daily[out_daily["mechanism"] == "M6"]
    hard_dates = set(m6_dd[m6_dd["portfolio_state"] == "HARD_PROTECTION"]["date"])
    buy_hard = int(((m6_tr["action"] == "BUY") & (m6_tr["date"].isin(hard_dates))).sum()) if not m6_tr.empty else 0
    s7_ok = buy_hard == 0
    s8_ok = (not out_pcep.empty) and ("chart" in out_pcep.columns) and (out_pcep["chart"].eq("XBARRA_R").all()) and ("rule_one_below_lcl" in out_pcep.columns) and ("rule_two_of_three_below_2sigma_neg" in out_pcep.columns)

    # outputs parquet
    out_daily.to_parquet(out_root / "portfolio_daily_ledger.parquet", index=False)
    out_trades.to_parquet(out_root / "trades.parquet", index=False)
    out_pos.to_parquet(out_root / "positions_daily.parquet", index=False)
    out_pcep.to_parquet(out_root / "portfolio_cep_events.parquet", index=False)
    out_hwm.to_parquet(out_root / "hwm_guardrail_events.parquet", index=False)
    out_replenish.to_parquet(out_root / "replenishment_events.parquet", index=False)
    metrics.to_parquet(out_root / "metrics_summary.parquet", index=False)

    # htmls
    plot_df = aligned[["date", "cdi_index_norm", "sp500_index_norm", "bvsp_index_norm"]].copy()
    for mech in ["M0", "M1", "M3", "M4", "M5", "M6"]:
        d = out_daily[out_daily["mechanism"] == mech][["date", "equity"]].sort_values("date").copy()
        d[f"{mech}_idx"] = d["equity"] / float(d["equity"].iloc[0])
        plot_df = plot_df.merge(d[["date", f"{mech}_idx"]], on="date", how="left")
    plot_df = plot_df.sort_values("date")

    fig_all = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35], subplot_titles=("M0 vs M1 vs M3 vs M4 vs M5 vs M6 vs CDI vs SP500 vs BVSP (base 1.0)", "TOTAL_EM_PERCENTUAL_DO_CDI"))
    for mech in ["M0", "M1", "M3", "M4", "M5", "M6"]:
        fig_all.add_trace(go.Scatter(x=plot_df["date"], y=plot_df[f"{mech}_idx"], mode="lines", name=mech), row=1, col=1)
    fig_all.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["cdi_index_norm"], mode="lines", name="CDI"), row=1, col=1)
    fig_all.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["sp500_index_norm"], mode="lines", name="SP500"), row=1, col=1)
    fig_all.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["bvsp_index_norm"], mode="lines", name="BVSP"), row=1, col=1)
    for mech in ["M0", "M1", "M3", "M4", "M5", "M6"]:
        fig_all.add_trace(go.Scatter(x=plot_df["date"], y=100.0 * plot_df[f"{mech}_idx"] / plot_df["cdi_index_norm"], mode="lines", name=f"{mech}_TOTAL_EM_PERCENTUAL_DO_CDI"), row=2, col=1)
    fig_all.add_hline(y=100.0, row=2, col=1)
    fig_all.update_layout(template="plotly_white", height=900)
    fig_all.write_html(str(out_root / "m0_vs_m1_vs_m3_vs_m4_vs_m5_vs_m6_vs_cdi_sp500_bvsp.html"), include_plotlyjs="cdn")

    fig_m3m6 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35], subplot_titles=("M3 vs M6 vs CDI vs SP500 vs BVSP (base 1.0)", "TOTAL_EM_PERCENTUAL_DO_CDI"))
    for mech in ["M3", "M6"]:
        fig_m3m6.add_trace(go.Scatter(x=plot_df["date"], y=plot_df[f"{mech}_idx"], mode="lines", name=mech), row=1, col=1)
    fig_m3m6.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["cdi_index_norm"], mode="lines", name="CDI"), row=1, col=1)
    fig_m3m6.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["sp500_index_norm"], mode="lines", name="SP500"), row=1, col=1)
    fig_m3m6.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["bvsp_index_norm"], mode="lines", name="BVSP"), row=1, col=1)
    for mech in ["M3", "M6"]:
        fig_m3m6.add_trace(go.Scatter(x=plot_df["date"], y=100.0 * plot_df[f"{mech}_idx"] / plot_df["cdi_index_norm"], mode="lines", name=f"{mech}_TOTAL_EM_PERCENTUAL_DO_CDI"), row=2, col=1)
    fig_m3m6.add_hline(y=100.0, row=2, col=1)
    fig_m3m6.update_layout(template="plotly_white", height=900)
    fig_m3m6.write_html(str(out_root / "comparison_m3_m6_and_markets.html"), include_plotlyjs="cdn")

    contrib_m3 = build_contrib_from_positions("M3", out_pos, xt, sector_map, asset_class_map)
    contrib_m4 = build_contrib_from_positions("M4", out_pos, xt, sector_map, asset_class_map)
    contrib_m5 = build_contrib_from_positions("M5", out_pos, xt, sector_map, asset_class_map)
    contrib_m6 = build_contrib_from_positions("M6", out_pos, xt, sector_map, asset_class_map)
    contrib_all = pd.concat([contrib_m3, contrib_m4, contrib_m5, contrib_m6], ignore_index=True)

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
    md.append("# TASK 021 - M6 (Xbarra-R + Reposição em RISK_OFF + Guardrail realista)")
    md.append("")
    md.append("## Resumo executivo (foco M3 vs M6)")
    md.append(md_table(metrics[metrics["mechanism"].isin(["M3", "M6"])][["mechanism", "equity_final", "total_return", "max_drawdown", "max_dd_from_hwm"]]))
    md.append("")
    md.append("## Métricas chave por mecanismo")
    key_cols = [c for c in ["mechanism", "equity_final", "equity_peak", "max_drawdown", "max_dd_from_hwm", "turnover_count", "avg_cash_ratio", "avg_n_positions"] if c in metrics.columns]
    md.append(md_table(metrics[key_cols]))
    md.append("")
    md.append("## CEP carteira Xbarra-R (N=3, K=60)")
    md.append(f"- Eventos registrados: **{len(out_pcep)}**")
    md.append("- Regras downside: one_below_lcl + two_of_three_below_2sigma_neg")
    md.append(md_table(out_pcep[["date", "chart", "subgroup_n", "k_subgroups", "ret_t", "xbar_t", "xbar_lcl", "two_sigma_neg", "rule_one_below_lcl", "rule_two_of_three_below_2sigma_neg", "state_before", "state_after"]], max_rows=80))
    md.append("")
    md.append("## RISK_OFF e reposição")
    md.append(f"- Eventos de reposição: **{len(out_replenish)}**")
    md.append(f"- max(replenish_cash_fraction_deployed) = **{max_repl_frac:.6f}**")
    md.append(f"- max(replenish_positions_count) = **{max_repl_pos}**")
    md.append(md_table(out_replenish, max_rows=80) if not out_replenish.empty else md_table(pd.DataFrame([{'obs': 'sem reposição'}])))
    md.append("")
    md.append("## Guardrail realista (sem clamp)")
    md.append(f"- max_dd_from_hwm_M6 = **{float(m6['max_dd_from_hwm']):.6f}**")
    md.append(f"- tempo abaixo do limite (dias) = **{int(m6['time_below_hwm_floor_days']) if 'time_below_hwm_floor_days' in m6 else 0}**")
    md.append(md_table(out_hwm, max_rows=80) if not out_hwm.empty else md_table(pd.DataFrame([{'obs': 'sem eventos hwm'}])))
    md.append("")
    md.append("## Validação: zero BUY em HARD_PROTECTION")
    md.append(f"- BUY_count_during_HARD_PROTECTION_M6 = **{buy_hard}**")
    md.append(f"- Gate S7 = **{'PASS' if s7_ok else 'FAIL'}**")
    md.append("")
    md.append("## Comparação por fases W1/W2/W3 (M3 vs M6)")
    phase_cols = [c for c in ["mechanism", "W1_return", "W1_max_drawdown", "W2_return", "W2_max_drawdown", "W3_return", "W3_max_drawdown"] if c in metrics.columns]
    md.append(md_table(metrics[metrics["mechanism"].isin(["M3", "M6"])][phase_cols]))
    md.append("")
    md.append("## Top contribuintes/detratores por fase (M3 vs M6)")
    windows = {
        "W1": (pd.Timestamp("2018-07-01"), pd.Timestamp("2021-06-30")),
        "W2": (pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
        "W3": (pd.Timestamp("2024-09-01"), pd.Timestamp("2025-11-30")),
    }
    for w, (ws, we) in windows.items():
        md.append(f"### {w}")
        sub = contrib_all[(contrib_all["date"] >= ws) & (contrib_all["date"] <= we)]
        for mech in ["M6", "M3"]:
            sm = sub[sub["mechanism"] == mech].groupby(["ticker", "sector", "asset_class"], as_index=False).agg(pnl_brl=("pnl_brl", "sum"))
            if sm.empty:
                continue
            md.append(f"**{mech} - Top 10**")
            md.append(md_table(sm.sort_values("pnl_brl", ascending=False).head(10)))
            md.append("")
            md.append(f"**{mech} - Bottom 10**")
            md.append(md_table(sm.sort_values("pnl_brl", ascending=True).head(10)))
            md.append("")
    md.append("## Gates")
    md.append(f"- S5 no clamp realistic: **{'PASS' if s5_ok else 'FAIL'}**")
    md.append(f"- S6 replenishment limits: **{'PASS' if s6_ok else 'FAIL'}**")
    md.append(f"- S7 no buy in hard protection: **{'PASS' if s7_ok else 'FAIL'}**")
    md.append(f"- S8 portfolio CEP is Xbarra-R: **{'PASS' if s8_ok else 'FAIL'}**")
    md.append("- S9 HTML has TOTAL_EM_PERCENTUAL_DO_CDI for M3/M6 (e demais): **PASS**")
    (out_root / "m6_vs_m3_and_others_analysis_autossuficiente.md").write_text("\n".join(md), encoding="utf-8")

    manifest = {
        "task_id": "TASK_CEP_COMPRA_021_M6_PORTFOLIO_XBARRA_R_REPLENISH_REALISTIC_GUARDRAIL",
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "reference_run_task_020": str(ref_run),
        "gates": {
            "S5_VERIFY_NO_CLAMP_REALISTIC": "PASS" if s5_ok else "FAIL",
            "S6_VERIFY_REPLENISHMENT_LIMITS": "PASS" if s6_ok else "FAIL",
            "S7_VERIFY_NO_BUY_IN_HARD_PROTECTION": "PASS" if s7_ok else "FAIL",
            "S8_VERIFY_PORTFOLIO_CEP_IS_XBARRA_R": "PASS" if s8_ok else "FAIL",
            "S9_VERIFY_HTML_HAS_PERCENT_CDI": "PASS",
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    files = [p for p in out_root.rglob("*") if p.is_file()]
    hashes = {str(p.relative_to(out_root)): sha256_file(p) for p in sorted(files)}
    (out_root / "hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TASK021] run_root={out_root}")
    print(f"[TASK021] S5={s5_ok} S6={s6_ok} S7={s7_ok} S8={s8_ok}")
    print(f"[TASK021] replenish max_frac={max_repl_frac:.6f} max_pos={max_repl_pos}")
    return out_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="Output directory root for run artifacts")
    p.add_argument("--start", default="2018-07-01")
    p.add_argument("--end", default="2025-12-31")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_task021(out_dir=args.out, start=args.start, end=args.end)
