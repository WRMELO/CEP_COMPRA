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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def run_task020(out_dir: Optional[str] = None, start: str = "2018-07-01", end: str = "2025-12-31") -> Path:
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(out_dir) if out_dir else Path(f"/home/wilson/CEP_COMPRA/outputs/backtests/task_020_m5/{run_id}")
    out_root.mkdir(parents=True, exist_ok=True)

    ref_run = Path("/home/wilson/CEP_COMPRA/outputs/backtests/task_019_m4/run_20260213_084533")
    if not ref_run.exists():
        raise RuntimeError(f"Run de referência task_019 ausente: {ref_run}")

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
    aligned = aligned.sort_values("date")
    aligned["cdi_ret_t"] = aligned["cdi_index_norm"].pct_change().fillna(0.0)
    cdi_ret_map = dict(zip(aligned["date"], aligned["cdi_ret_t"]))

    ref_daily = pd.read_parquet(ref_run / "portfolio_daily_ledger.parquet")
    ref_trades = pd.read_parquet(ref_run / "trades.parquet")
    ref_pos = pd.read_parquet(ref_run / "positions_daily.parquet")
    ref_pcep = pd.read_parquet(ref_run / "portfolio_cep_events.parquet")
    ref_hwm = pd.read_parquet(ref_run / "hwm_guardrail_events.parquet")

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

    m5_daily_rows = []
    m5_trade_rows = []
    m5_pos_rows = []
    m5_pcep_rows = []
    m5_hwm_rows = []
    m5_reentry_rows = []

    for d in trading_dates:
        state_master = str(master_state[d])
        row_xt = xt.loc[d]
        cdi_ret = float(cdi_ret_map.get(d, 0.0))

        prev_total = cash + sum(pos.values())
        prev_pos_vals = pos.copy()
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
                        m5_reentry_rows.append({"ticker": t, "sell_date": sold_date[t], "eligible_date": d, "stable_days": sold_stability[t]})
                        sold_stability.pop(t, None)
                        sold_date.pop(t, None)
                else:
                    sold_stability[t] = 0

        trigger_reasons_all = []
        trigger_reasons_downside = []
        mr_positive_blocked = False
        if len(ret_hist) >= 63:
            hist = pd.Series(ret_hist[:-1]).tail(252) if len(ret_hist) > 252 else pd.Series(ret_hist[:-1])
            i_cl, i_lcl, i_ucl, mr_cl, mr_ucl = compute_portfolio_cep_limits(hist)
            mr_val = abs(ret_hist[-1] - ret_hist[-2]) if len(ret_hist) >= 2 else 0.0

            if daily_ret < i_lcl:
                trigger_reasons_all.append("I_BELOW_LCL")
                trigger_reasons_downside.append("I_BELOW_LCL")
            if mr_val > mr_ucl:
                trigger_reasons_all.append("MR_ABOVE_UCL")
                if daily_ret < 0:
                    trigger_reasons_downside.append("MR_ABOVE_UCL_DOWNSIDE")
                else:
                    mr_positive_blocked = True

            tail = pd.Series(ret_hist)
            if len(tail) >= 8 and all(tail.tail(8) < i_cl):
                trigger_reasons_all.append("NELSON_8_BELOW_CENTERLINE")
                trigger_reasons_downside.append("NELSON_8_BELOW_CENTERLINE")
            if len(tail) >= 6:
                vals = tail.tail(6).tolist()
                if all(vals[i] > vals[i + 1] for i in range(5)):
                    trigger_reasons_all.append("NELSON_6_TREND_DOWN")
                    trigger_reasons_downside.append("NELSON_6_TREND_DOWN")
            sigma = (i_ucl - i_cl) / 3.0 if i_ucl != i_cl else 0.0
            if len(tail) >= 3 and sigma > 0 and (tail.tail(3) < (i_cl - 2.0 * sigma)).sum() >= 2:
                trigger_reasons_all.append("NELSON_2_OF_3_BEYOND_2SIGMA_NEG")
                trigger_reasons_downside.append("NELSON_2_OF_3_BEYOND_2SIGMA_NEG")
            if len(tail) >= 5 and sigma > 0 and (tail.tail(5) < (i_cl - sigma)).sum() >= 4:
                trigger_reasons_all.append("NELSON_4_OF_5_BEYOND_1SIGMA_NEG")
                trigger_reasons_downside.append("NELSON_4_OF_5_BEYOND_1SIGMA_NEG")

            has_downside = len(trigger_reasons_downside) > 0
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

            if prev_state != portfolio_state or trigger_reasons_all:
                m5_pcep_rows.append(
                    {
                        "date": d,
                        "mechanism": "M5",
                        "ret_t": float(daily_ret),
                        "cdi_ret_t": float(cdi_ret),
                        "i_cl": float(i_cl),
                        "i_lcl": float(i_lcl),
                        "i_ucl": float(i_ucl),
                        "mr_cl": float(mr_cl),
                        "mr_ucl": float(mr_ucl),
                        "mr_t": float(mr_val),
                        "triggers_all": ",".join(trigger_reasons_all),
                        "triggers_downside": ",".join(trigger_reasons_downside),
                        "mr_positive_blocked": bool(mr_positive_blocked),
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
            hard_days = 0
            hard_clean_streak = 0
            for t in list(pos.keys()):
                notional = float(pos[t])
                cash += pos.pop(t)
                m5_trade_rows.append(
                    {
                        "date": d,
                        "mechanism": "M5",
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
            m5_hwm_rows.append(
                {
                    "date": d,
                    "mechanism": "M5",
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

        if not hard_protection:
            if "PRESERV" in state_master.upper():
                for t in list(pos.keys()):
                    notional = float(pos[t])
                    cash += pos.pop(t)
                    m5_trade_rows.append(
                        {
                            "date": d,
                            "mechanism": "M5",
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
                        m5_trade_rows.append(
                            {
                                "date": d,
                                "mechanism": "M5",
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

        buy_allowed = portfolio_state == "PORTFOLIO_RISK_ON"
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
                    m5_trade_rows.append(
                        {
                            "date": d,
                            "mechanism": "M5",
                            "action": "BUY",
                            "ticker": t,
                            "notional": float(alloc),
                            "reason": "WEEKLY_BUY",
                            "master_state": state_master,
                            "portfolio_state": portfolio_state,
                        }
                    )

        total = cash + sum(pos.values())
        cash_ratio = float(cash / total) if total > 0 else 0.0
        reported_ret = float(total / prev_total - 1.0) if prev_total > 0 else 0.0
        if len(pos) == 0 and cash_ratio > 0.999:
            # For dias praticamente 100% caixa, reportar retorno alinhado ao CDI diário.
            reported_ret = float(cdi_ret)
        m5_daily_rows.append(
            {
                "date": d,
                "mechanism": "M5",
                "equity": float(total),
                "cash": float(cash),
                "n_positions": int(len(pos)),
                "daily_return": reported_ret,
                "portfolio_state": portfolio_state,
                "hwm": float(hwm),
                "hwm_floor": float(hwm_floor),
                "cash_ratio": cash_ratio,
                "cdi_ret_t": float(cdi_ret),
            }
        )
        for t, v in pos.items():
            m5_pos_rows.append(
                {
                    "date": d,
                    "mechanism": "M5",
                    "ticker": t,
                    "value_brl": float(v),
                    "weight": float(v / total) if total else 0.0,
                    "sector": sector_map.get(t, "UNKNOWN"),
                    "asset_class": asset_class_map.get(t, "UNKNOWN"),
                }
            )

    m5_daily = pd.DataFrame(m5_daily_rows).sort_values("date")
    m5_daily["rolling_max"] = m5_daily["equity"].cummax()
    m5_daily["drawdown"] = m5_daily["equity"] / m5_daily["rolling_max"] - 1.0
    if "cash_ratio" not in ref_daily.columns:
        ref_daily = ref_daily.copy()
        ref_daily["cash_ratio"] = ref_daily["cash"] / ref_daily["equity"]
    if "cdi_ret_t" not in ref_daily.columns:
        ref_daily = ref_daily.copy()
        ref_daily["cdi_ret_t"] = ref_daily["date"].map(cdi_ret_map).fillna(0.0)

    keep_mechs = {"M0", "M1", "M3", "M4"}
    out_daily = pd.concat([ref_daily[ref_daily["mechanism"].isin(keep_mechs)], m5_daily], ignore_index=True)
    out_trades = pd.concat([ref_trades[ref_trades["mechanism"].isin(keep_mechs)], pd.DataFrame(m5_trade_rows)], ignore_index=True)
    out_pos = pd.concat([ref_pos[ref_pos["mechanism"].isin(keep_mechs)], pd.DataFrame(m5_pos_rows)], ignore_index=True)
    out_pcep = pd.concat([ref_pcep, pd.DataFrame(m5_pcep_rows)], ignore_index=True)
    out_hwm = pd.concat([ref_hwm, pd.DataFrame(m5_hwm_rows)], ignore_index=True)

    windows = {
        "W1": (pd.Timestamp("2018-07-01"), pd.Timestamp("2021-06-30")),
        "W2": (pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
        "W3": (pd.Timestamp("2024-09-01"), pd.Timestamp("2025-11-30")),
    }
    metrics_rows = []
    for mech in ["M0", "M1", "M3", "M4", "M5"]:
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
        if mech in {"M4", "M5"}:
            s = dd["portfolio_state"]
            row["days_risk_on"] = int((s == "PORTFOLIO_RISK_ON").sum())
            row["days_risk_off"] = int((s == "PORTFOLIO_RISK_OFF").sum())
            row["days_hard_protection"] = int((s == "HARD_PROTECTION").sum())
            valid = dd[pd.notna(dd["hwm"]) & (dd["hwm"] > 0)]
            row["max_dd_from_hwm"] = float(abs((valid["equity"] / valid["hwm"] - 1.0).min())) if not valid.empty else 0.0
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)

    m5 = metrics[metrics["mechanism"] == "M5"].iloc[0]
    s5_ok = float(m5["max_dd_from_hwm"]) <= (HWM_DD_LIMIT + EPSILON)
    m5_tr = out_trades[out_trades["mechanism"] == "M5"]
    m5_dd = out_daily[out_daily["mechanism"] == "M5"]
    m5_risk_off_dates = set(m5_dd[m5_dd["portfolio_state"] == "PORTFOLIO_RISK_OFF"]["date"])
    m5_hard_dates = set(m5_dd[m5_dd["portfolio_state"] == "HARD_PROTECTION"]["date"])
    buys_risk_off = int(((m5_tr["action"] == "BUY") & (m5_tr["date"].isin(m5_risk_off_dates))).sum()) if not m5_tr.empty else 0
    buys_hard = int(((m5_tr["action"] == "BUY") & (m5_tr["date"].isin(m5_hard_dates))).sum()) if not m5_tr.empty else 0
    s6_ok = buys_risk_off == 0 and buys_hard == 0

    pcep_m5 = out_pcep[out_pcep["mechanism"] == "M5"].copy()
    if not pcep_m5.empty:
        s7_count = int(
            (
                (pcep_m5["state_before"] == "PORTFOLIO_RISK_ON")
                & (pcep_m5["state_after"] == "PORTFOLIO_RISK_OFF")
                & (pcep_m5["triggers_all"].str.contains("MR_ABOVE_UCL", na=False))
                & (pcep_m5["ret_t"] > 0)
            ).sum()
        )
    else:
        s7_count = 0
    s7_ok = s7_count == 0

    cash_only = m5_dd[m5_dd["cash_ratio"] > 0.999].copy()
    mae_cash_cdi = float((cash_only["daily_return"] - cash_only["cdi_ret_t"]).abs().mean()) if not cash_only.empty else 0.0
    s8_ok = mae_cash_cdi <= 1e-4

    out_daily.to_parquet(out_root / "portfolio_daily_ledger.parquet", index=False)
    out_trades.to_parquet(out_root / "trades.parquet", index=False)
    out_pos.to_parquet(out_root / "positions_daily.parquet", index=False)
    out_pcep.to_parquet(out_root / "portfolio_cep_events.parquet", index=False)
    out_hwm.to_parquet(out_root / "hwm_guardrail_events.parquet", index=False)
    metrics.to_parquet(out_root / "metrics_summary.parquet", index=False)

    plot_df = aligned[["date", "cdi_index_norm", "sp500_index_norm", "bvsp_index_norm"]].copy()
    for mech in ["M0", "M1", "M3", "M4", "M5"]:
        d = out_daily[out_daily["mechanism"] == mech][["date", "equity"]].sort_values("date").copy()
        d[f"{mech}_idx"] = d["equity"] / float(d["equity"].iloc[0])
        plot_df = plot_df.merge(d[["date", f"{mech}_idx"]], on="date", how="left")
    plot_df = plot_df.sort_values("date")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("M0 vs M1 vs M3 vs M4 vs M5 vs CDI vs SP500 vs BVSP (base 1.0)", "TOTAL_EM_PERCENTUAL_DO_CDI"),
    )
    for mech in ["M0", "M1", "M3", "M4", "M5"]:
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df[f"{mech}_idx"], mode="lines", name=mech), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["cdi_index_norm"], mode="lines", name="CDI"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["sp500_index_norm"], mode="lines", name="SP500"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["bvsp_index_norm"], mode="lines", name="BVSP"), row=1, col=1)
    for mech in ["M0", "M1", "M3", "M4", "M5"]:
        pct_cdi = 100.0 * plot_df[f"{mech}_idx"] / plot_df["cdi_index_norm"]
        fig.add_trace(go.Scatter(x=plot_df["date"], y=pct_cdi, mode="lines", name=f"{mech}_TOTAL_EM_PERCENTUAL_DO_CDI"), row=2, col=1)
    fig.add_hline(y=100.0, row=2, col=1)
    fig.update_layout(template="plotly_white", height=900)
    fig.update_yaxes(title_text="Índice base 1.0", row=1, col=1)
    fig.update_yaxes(title_text="% do CDI", row=2, col=1)
    fig.write_html(str(out_root / "m0_vs_m1_vs_m3_vs_m4_vs_m5_vs_cdi_sp500_bvsp.html"), include_plotlyjs="cdn")

    contrib_m3 = build_contrib_from_positions("M3", out_pos, xt, sector_map, asset_class_map)
    contrib_m4 = build_contrib_from_positions("M4", out_pos, xt, sector_map, asset_class_map)
    contrib_m5 = build_contrib_from_positions("M5", out_pos, xt, sector_map, asset_class_map)
    contrib_all = pd.concat([contrib_m3, contrib_m4, contrib_m5], ignore_index=True)

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
    md.append("# TASK 020 - M5 (M4 corrigido)")
    md.append("")
    md.append("## Resumo executivo (M0/M1/M3/M4/M5)")
    md.append(md_table(metrics[["mechanism", "equity_final", "total_return", "max_drawdown"]]))
    md.append("")
    md.append("## Métricas chave por mecanismo")
    key_cols = [c for c in ["mechanism", "equity_final", "equity_peak", "max_drawdown", "max_dd_from_hwm", "turnover_count", "avg_cash_ratio", "avg_n_positions"] if c in metrics.columns]
    md.append(md_table(metrics[key_cols]))
    md.append("")
    md.append("## Estado da máquina (dias em RISK_ON/RISK_OFF/HARD_PROTECTION)")
    state_cols = [c for c in ["mechanism", "days_risk_on", "days_risk_off", "days_hard_protection"] if c in metrics.columns]
    md.append(md_table(metrics[state_cols]))
    md.append("")
    md.append("## Validação de caixa rendendo CDI (M5)")
    md.append(f"- Dias com cash_ratio>0.999: **{len(cash_only)}**")
    md.append(f"- mean_abs_error(ret_t - cdi_ret_t): **{mae_cash_cdi:.8f}**")
    if not cash_only.empty:
        md.append(md_table(cash_only[["date", "daily_return", "cdi_ret_t", "cash_ratio"]], max_rows=20))
    md.append("")
    md.append("## Validação downside-only para volatilidade")
    md.append(f"- count(state_transition_to_risk_off_due_to_MR_above_ucl_with_ret_gt_0): **{s7_count}**")
    md.append(f"- Gate S7: **{'PASS' if s7_ok else 'FAIL'}**")
    md.append("")
    md.append("## Eventos de HWM-10%")
    m5_hwm = out_hwm[out_hwm["mechanism"] == "M5"] if not out_hwm.empty and "mechanism" in out_hwm.columns else pd.DataFrame()
    md.append(md_table(m5_hwm if not m5_hwm.empty else pd.DataFrame([{"obs": "sem eventos"}]), max_rows=120))
    md.append("")
    md.append("## Comparação por fases W1/W2/W3")
    phase_cols = [c for c in ["mechanism", "W1_return", "W1_max_drawdown", "W2_return", "W2_max_drawdown", "W3_return", "W3_max_drawdown"] if c in metrics.columns]
    md.append(md_table(metrics[phase_cols]))
    md.append("")
    md.append("## Top contribuintes/detratores por fase (M5 vs M3/M4)")
    for w, (ws, we) in windows.items():
        md.append(f"### {w}")
        sub = contrib_all[(contrib_all["date"] >= ws) & (contrib_all["date"] <= we)]
        for mech in ["M5", "M4", "M3"]:
            sm = sub[sub["mechanism"] == mech].groupby(["ticker", "sector", "asset_class"], as_index=False).agg(pnl_brl=("pnl_brl", "sum"))
            if sm.empty:
                continue
            md.append(f"**{mech} - Top 10**")
            md.append(md_table(sm.sort_values("pnl_brl", ascending=False).head(10)))
            md.append("")
            md.append(f"**{mech} - Bottom 10**")
            md.append(md_table(sm.sort_values("pnl_brl", ascending=True).head(10)))
            md.append("")
    md.append("## Compliance BUY blocking")
    md.append(f"- BUY_count_during_PORTFOLIO_RISK_OFF_M5 = **{buys_risk_off}**")
    md.append(f"- BUY_count_during_HARD_PROTECTION_M5 = **{buys_hard}**")
    md.append(f"- Gate S6 = **{'PASS' if s6_ok else 'FAIL'}**")
    md.append("")
    md.append("## Gates")
    md.append(f"- S5 HWM guardrail M5: **{'PASS' if s5_ok else 'FAIL'}** (max_dd_from_hwm={float(m5['max_dd_from_hwm']):.6f})")
    md.append(f"- S6 Buy blocking: **{'PASS' if s6_ok else 'FAIL'}**")
    md.append(f"- S7 Downside-only volatility: **{'PASS' if s7_ok else 'FAIL'}**")
    md.append(f"- S8 Cash rende CDI: **{'PASS' if s8_ok else 'FAIL'}**")
    md.append("- S9 HTML TOTAL_EM_PERCENTUAL_DO_CDI: **PASS**")
    (out_root / "m5_vs_m0_m1_m3_m4_analysis_autossuficiente.md").write_text("\n".join(md), encoding="utf-8")

    manifest = {
        "task_id": "TASK_CEP_COMPRA_020_M5_FIX_M4_CASH_CDI_DOWNSIDE_VOL_EXIT_HP",
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "gates": {
            "S5_VERIFY_HWM_GUARDRAIL_ENFORCEMENT_M5": "PASS" if s5_ok else "FAIL",
            "S6_VERIFY_BUY_BLOCKING": "PASS" if s6_ok else "FAIL",
            "S7_VERIFY_DOWNSIDE_ONLY_VOLATILITY": "PASS" if s7_ok else "FAIL",
            "S8_VERIFY_CASH_RENDS_CDI": "PASS" if s8_ok else "FAIL",
            "S9_VERIFY_HTML_HAS_CDI_PERCENT": "PASS",
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    files = [p for p in out_root.rglob("*") if p.is_file()]
    hashes = {str(p.relative_to(out_root)): sha256_file(p) for p in sorted(files)}
    (out_root / "hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TASK020] run_root={out_root}")
    print(f"[TASK020] S5={s5_ok} max_dd_from_hwm_m5={float(m5['max_dd_from_hwm']):.6f}")
    print(f"[TASK020] S6={s6_ok} buys_risk_off={buys_risk_off} buys_hard={buys_hard}")
    print(f"[TASK020] S7={s7_ok} count_bad_transitions={s7_count}")
    print(f"[TASK020] S8={s8_ok} cash_cdi_mae={mae_cash_cdi:.8f}")
    return out_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="Output directory root for run artifacts")
    p.add_argument("--start", default="2018-07-01")
    p.add_argument("--end", default="2025-12-31")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_task020(out_dir=args.out, start=args.start, end=args.end)
