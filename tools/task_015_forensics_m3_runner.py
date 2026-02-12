#!/usr/bin/env python3
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go


CANON_START = pd.Timestamp("2018-07-01")
CANON_END = pd.Timestamp("2025-12-31")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_canonical_window(df: pd.DataFrame, date_col: str, name: str) -> None:
    dmin = pd.to_datetime(df[date_col]).min()
    dmax = pd.to_datetime(df[date_col]).max()
    if dmin < CANON_START or dmax > CANON_END:
        raise RuntimeError(f"FAIL_WINDOW {name}: min={dmin} max={dmax} fora de 2018-07-01..2025-12-31")


def first_trading_day_by_week(dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
    out = []
    seen = set()
    for d in dates:
        key = (d.isocalendar().year, d.isocalendar().week)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


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
    df = pd.DataFrame(rows, columns=["ticker", "date", "volume"])
    return df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")


def select_tickers(
    mech: str,
    cand_df: pd.DataFrame,
    target_n: int,
    min_volume_days: int,
) -> Tuple[List[str], pd.DataFrame]:
    if cand_df.empty:
        return [], pd.DataFrame(columns=["ticker", "mechanism", "decision", "reason"])

    # base ranking fields available in pipeline
    work = cand_df.copy()
    work["liquidity_ok"] = work["liquidity_days"] >= min_volume_days
    work["sector"] = work["sector"].fillna("UNKNOWN").replace("", "UNKNOWN")

    if mech == "M0":
        ranked = work.sort_values(["score_m0", "ret_lookback", "ticker"], ascending=[False, False, True])
        selected = ranked["ticker"].tolist()[:target_n]
        log = [{"ticker": t, "mechanism": mech, "decision": "selected", "reason": "ranking_m0"} for t in selected]
        return selected, pd.DataFrame(log)

    # M1/M3 invariants
    work = work[work["liquidity_ok"]].copy()
    work = (
        work.sort_values(["ret_lookback", "score_m0", "ticker"], ascending=[False, False, True])
        .drop_duplicates(subset=["company_key"], keep="first")
        .copy()
    )

    if mech == "M1":
        work = work.sort_values(["score_m0", "ret_lookback", "ticker"], ascending=[False, False, True])
    else:
        # M3: score combinado apenas com medidas existentes (retorno+score-vol)
        for c in ["score_m0", "ret_lookback", "vol_lookback"]:
            if c not in work.columns:
                raise RuntimeError(f"M3 bloqueado: coluna ausente {c}")
        def z(v: pd.Series) -> pd.Series:
            s = v.std(ddof=0)
            if pd.isna(s) or s == 0:
                return pd.Series([0.0] * len(v), index=v.index)
            return (v - v.mean()) / s
        work["score_m3"] = z(work["score_m0"]) + z(work["ret_lookback"]) - z(work["vol_lookback"])
        work = work.sort_values(["score_m3", "score_m0", "ret_lookback", "ticker"], ascending=[False, False, False, True])

    chosen: List[str] = []
    logs: List[Dict] = []
    sec_count: Dict[str, int] = {}
    b3_count = 0
    bdr_count = 0

    def can_add(row: pd.Series) -> Tuple[bool, str]:
        nonlocal b3_count, bdr_count
        sec = row["sector"]
        cls = row["asset_class"]
        if sec_count.get(sec, 0) >= 2:
            return False, "sector_cap"
        if cls == "B3" and b3_count >= 8:
            return False, "b3_max"
        if cls == "BDR" and bdr_count >= 5:
            return False, "bdr_max"
        return True, "ok"

    def add(row: pd.Series, phase: str) -> None:
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
        logs.append({"ticker": t, "mechanism": mech, "decision": "selected", "reason": phase})

    for _, row in work[work["asset_class"] == "B3"].iterrows():
        if len(chosen) >= target_n or b3_count >= 5:
            break
        ok, reason = can_add(row)
        if ok:
            add(row, "min_b3")
        else:
            logs.append({"ticker": row["ticker"], "mechanism": mech, "decision": "skipped", "reason": reason})

    for _, row in work[work["asset_class"] == "BDR"].iterrows():
        if len(chosen) >= target_n or bdr_count >= 2:
            break
        if row["ticker"] in chosen:
            continue
        ok, reason = can_add(row)
        if ok:
            add(row, "min_bdr")
        else:
            logs.append({"ticker": row["ticker"], "mechanism": mech, "decision": "skipped", "reason": reason})

    for _, row in work.iterrows():
        if len(chosen) >= target_n:
            break
        if row["ticker"] in chosen:
            continue
        ok, reason = can_add(row)
        if ok:
            add(row, "fill")
        else:
            logs.append({"ticker": row["ticker"], "mechanism": mech, "decision": "skipped", "reason": reason})

    return chosen[:target_n], pd.DataFrame(logs)


def main() -> int:
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    out_for = Path(f"/home/wilson/CEP_COMPRA/outputs/forensics/task_015/{run_id}")
    out_for_data = out_for / "data"
    out_bt = Path(f"/home/wilson/CEP_COMPRA/outputs/backtests/task_015_m3/{run_id}")
    out_bt_con = out_bt / "consolidated"
    out_bt_plots = out_bt / "plots"
    for p in [out_for, out_for_data, out_bt, out_bt_con, out_bt_plots]:
        p.mkdir(parents=True, exist_ok=True)

    # Inputs
    task012_root = Path("/home/wilson/CEP_COMPRA/outputs/backtests/task_012/run_20260212_114129")
    p_m0_eq = task012_root / "raw/M0_equity_curve.parquet"
    p_m1_eq = task012_root / "raw/M1_equity_curve.parquet"
    p_m0_w = task012_root / "raw/M0_weekly_decisions.parquet"
    p_m1_w = task012_root / "raw/M1_weekly_decisions.parquet"
    p_aligned_012 = task012_root / "consolidated/series_alinhadas_plot.parquet"
    p_setor = Path("/home/wilson/CEP_COMPRA/outputs/ssot/setores/ssot_latest/setores_ticker_latest.parquet")

    for p in [p_m0_eq, p_m1_eq, p_m0_w, p_m1_w, p_aligned_012, p_setor]:
        if not p.exists():
            raise RuntimeError(f"Input ausente: {p}")

    # Assert janela canônica inputs task_012
    assert_canonical_window(pd.read_parquet(p_m0_eq), "date", "task012_m0_equity")
    assert_canonical_window(pd.read_parquet(p_m1_eq), "date", "task012_m1_equity")
    assert_canonical_window(pd.read_parquet(p_m0_w), "date", "task012_m0_weekly")
    assert_canonical_window(pd.read_parquet(p_m1_w), "date", "task012_m1_weekly")

    p_base = Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/base_operacional/base_operacional_xt.csv")
    p_limits = Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/merged/limits_per_ticker.csv")
    p_master = Path("/home/wilson/CEP_NA_BOLSA/outputs/experimentos/fase1_calibracao/exp/20260209/dataset_sizing/master_states.csv")
    p_ssot_a = Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/acoes/b3/20260204/ssot_acoes_b3.csv")
    p_ssot_b = Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/bdr/b3/20260204/ssot_bdr_b3.csv")
    p_sizing = Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos/060_selected_config_sizing_v2.json")

    df_base = pd.read_csv(p_base, parse_dates=["date"])
    df_limits = pd.read_csv(p_limits).set_index("ticker")
    df_master = pd.read_csv(p_master, parse_dates=["date"]).sort_values("date")
    df_ssot_a = pd.read_csv(p_ssot_a)
    df_ssot_b = pd.read_csv(p_ssot_b)
    df_setor = pd.read_parquet(p_setor)
    sizing = json.loads(p_sizing.read_text(encoding="utf-8"))
    w_cap = float(sizing.get("w_cap", 0.15))

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

    vol = load_volume_cache(sorted(list(universe)), asset_class_map)
    vol_pivot = vol.pivot(index="date", columns="ticker", values="volume").sort_index() if not vol.empty else pd.DataFrame()

    df_base = df_base[df_base["ticker"].isin(universe)].copy().sort_values(["date", "ticker"])
    xt = df_base.pivot(index="date", columns="ticker", values="xt").sort_index()
    master_state = dict(zip(df_master["date"], df_master["state"]))
    trading_dates = pd.DatetimeIndex([d for d in xt.index if d in master_state and CANON_START <= d <= CANON_END])
    buy_dates = set(first_trading_day_by_week(list(trading_dates)))

    # rolling xbar/r por ticker
    xbar_df = {}
    r_df = {}
    vol_df = {}
    for t in xt.columns:
        n = int(df_limits.loc[t].get("n", 3)) if t in df_limits.index and pd.notna(df_limits.loc[t].get("n", None)) else 3
        s = xt[t]
        xbar_df[t] = s.rolling(window=max(2, n), min_periods=max(2, n)).mean()
        r_df[t] = s.rolling(window=max(2, n), min_periods=max(2, n)).max() - s.rolling(window=max(2, n), min_periods=max(2, n)).min()
        vol_df[t] = s.rolling(window=62, min_periods=20).std()
    xbar_df = pd.DataFrame(xbar_df, index=xt.index)
    r_df = pd.DataFrame(r_df, index=xt.index)
    vol62_df = pd.DataFrame(vol_df, index=xt.index)

    windows = {
        "W1_outperformance": (pd.Timestamp("2018-07-01"), pd.Timestamp("2021-06-30")),
        "W2_underperformance": (pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
        "W2a_drawdown_aug21": (pd.Timestamp("2021-08-01"), pd.Timestamp("2021-09-30")),
        "W2b_drawdown_may22": (pd.Timestamp("2022-05-01"), pd.Timestamp("2022-06-30")),
        "W3_late_decline": (pd.Timestamp("2024-09-01"), pd.Timestamp("2025-11-30")),
    }

    daily_all = {}
    weekly_holdings = []
    contrib_rows = []
    trigger_rows = []
    decision_logs = []

    for mech in ["M0", "M1", "M3"]:
        cash = 1.0
        pos: Dict[str, float] = {}
        prev_xt: Dict[str, float] = {}
        daily_rows = []

        for d in trading_dates:
            state = str(master_state[d])
            row_xt = xt.loc[d]
            prev_equity = cash + sum(pos.values())
            sell_turnover = 0.0
            buy_turnover = 0.0

            prev_pos_vals = {t: v for t, v in pos.items()}
            for t in list(pos.keys()):
                rv = row_xt.get(t)
                if pd.notna(rv):
                    pos[t] *= math.exp(float(rv))

            if "PRESERV" in state.upper():
                for t, v in pos.items():
                    sell_turnover += v
                cash += sum(pos.values())
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
                    trigger_rows.append(
                        {
                            "date": d,
                            "mechanism": mech,
                            "ticker": t,
                            "master_state": state,
                            "upside_extreme": bool(upside_extreme),
                            "stress_amp": bool(stress_amp),
                            "trigger_i_lcl": bool(sell_i_lcl),
                            "trigger_i_ucl": bool(sell_i_ucl),
                            "trigger_mr_ucl": bool(sell_mr),
                            "sold": bool(sell),
                        }
                    )
                    if sell:
                        sell_turnover += pos[t]
                        cash += pos.pop(t)
                    prev_xt[t] = float(r_now)

            if d in buy_dates and "RISK_ON" in state:
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
                selected, dec_log = select_tickers(mech, cand_df, target_n=10, min_volume_days=50)
                if not dec_log.empty:
                    dec_log["date"] = d
                    decision_logs.append(dec_log)

                total_equity = cash + sum(pos.values())
                slot = total_equity / 10.0
                cap_value = total_equity * float(sizing.get("w_cap", 0.15))
                for t in selected:
                    cur = pos.get(t, 0.0)
                    desired = min(slot, cap_value)
                    need = max(0.0, desired - cur)
                    alloc = min(need, cash)
                    if alloc > 0:
                        pos[t] = cur + alloc
                        cash -= alloc
                        buy_turnover += alloc

                # weekly holdings snapshot
                tot = cash + sum(pos.values())
                for t, v in sorted(pos.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                    rt = row_xt.get(t)
                    under_control = False
                    if t in df_limits.index and pd.notna(rt):
                        lim = df_limits.loc[t]
                        under_control = float(lim["i_lcl"]) <= float(rt) <= float(lim["i_ucl"])
                    weekly_holdings.append(
                        {
                            "event_date": d,
                            "mechanism": mech,
                            "ticker": t,
                            "value_brl": float(v),
                            "weight": (float(v) / float(tot)) if tot else 0.0,
                            "sector": sector_map.get(t, "UNKNOWN"),
                            "asset_class": asset_class_map.get(t, "UNKNOWN"),
                            "under_control": bool(under_control),
                        }
                    )

            # ticker contribution (pnl) from previous holdings
            for t, prev_v in prev_pos_vals.items():
                rv = row_xt.get(t)
                if pd.isna(rv):
                    continue
                pnl = prev_v * (math.exp(float(rv)) - 1.0)
                contrib_rows.append(
                    {
                        "date": d,
                        "mechanism": mech,
                        "ticker": t,
                        "asset_class": asset_class_map.get(t, "UNKNOWN"),
                        "sector": sector_map.get(t, "UNKNOWN"),
                        "pnl_brl": float(pnl),
                        "ret_xt": float(rv),
                        "equity_prev": float(prev_equity),
                    }
                )

            equity = cash + sum(pos.values())
            daily_rows.append(
                {
                    "date": d,
                    "mechanism": mech,
                    "equity": float(equity),
                    "cash": float(cash),
                    "n_positions": int(len(pos)),
                    "turnover_brl": float(sell_turnover + buy_turnover),
                    "turnover_ratio": float((sell_turnover + buy_turnover) / prev_equity) if prev_equity > 0 else 0.0,
                }
            )

        daily_df = pd.DataFrame(daily_rows).sort_values("date")
        daily_df["rolling_max"] = daily_df["equity"].cummax()
        daily_df["drawdown"] = (daily_df["equity"] / daily_df["rolling_max"]) - 1.0
        daily_df["daily_return"] = daily_df["equity"].pct_change()
        assert_canonical_window(daily_df, "date", f"sim_{mech}_daily")
        daily_all[mech] = daily_df

    # Save weekly holdings
    df_wh = pd.DataFrame(weekly_holdings)
    wh_m0 = df_wh[df_wh["mechanism"] == "M0"].copy()
    wh_m1 = df_wh[df_wh["mechanism"] == "M1"].copy()
    p_wh_m0 = out_for_data / "weekly_holdings_m0.parquet"
    p_wh_m1 = out_for_data / "weekly_holdings_m1.parquet"
    wh_m0.to_parquet(p_wh_m0, index=False)
    wh_m1.to_parquet(p_wh_m1, index=False)

    # Contribution by ticker/phase
    contrib = pd.DataFrame(contrib_rows)
    w_rows = []
    for wname, (ws, we) in windows.items():
        sub = contrib[(contrib["date"] >= ws) & (contrib["date"] <= we)].copy()
        if sub.empty:
            continue
        g = (
            sub.groupby(["mechanism", "ticker", "asset_class", "sector"], as_index=False)
            .agg(pnl_brl=("pnl_brl", "sum"))
        )
        g["window"] = wname
        # concentration
        for mech in g["mechanism"].unique():
            gm = g[g["mechanism"] == mech].copy()
            denom = gm["pnl_brl"].abs().sum()
            if denom == 0:
                hhi = 0.0
            else:
                hhi = float(((gm["pnl_brl"].abs() / denom) ** 2).sum())
            g.loc[g["mechanism"] == mech, "hhi_ticker"] = hhi
        w_rows.append(g)
    df_ct = pd.concat(w_rows, ignore_index=True) if w_rows else pd.DataFrame()
    p_ct = out_for_data / "contribuicao_ticker_fases.parquet"
    df_ct.to_parquet(p_ct, index=False)

    # Contribution by sector/assetclass
    rows_sa = []
    for wname, (ws, we) in windows.items():
        sub = contrib[(contrib["date"] >= ws) & (contrib["date"] <= we)].copy()
        if sub.empty:
            continue
        g = sub.groupby(["mechanism", "asset_class", "sector"], as_index=False).agg(pnl_brl=("pnl_brl", "sum"))
        g["window"] = wname
        for mech in g["mechanism"].unique():
            gm = g[g["mechanism"] == mech].copy()
            denom = gm["pnl_brl"].abs().sum()
            hhi_sector = float(((gm.groupby("sector")["pnl_brl"].sum().abs() / denom) ** 2).sum()) if denom else 0.0
            g.loc[g["mechanism"] == mech, "hhi_sector"] = hhi_sector
        rows_sa.append(g)
    df_sa = pd.concat(rows_sa, ignore_index=True) if rows_sa else pd.DataFrame()
    p_sa = out_for_data / "contribuicao_setor_assetclass_fases.parquet"
    df_sa.to_parquet(p_sa, index=False)

    # Audit sell triggers by windows
    trig = pd.DataFrame(trigger_rows)
    daily_all_df = pd.concat(daily_all.values(), ignore_index=True)
    a_rows = []
    for wname, (ws, we) in windows.items():
        tr = trig[(trig["date"] >= ws) & (trig["date"] <= we)].copy()
        dd = daily_all_df[(daily_all_df["date"] >= ws) & (daily_all_df["date"] <= we)].copy()
        for mech in ["M0", "M1", "M3"]:
            tm = tr[tr["mechanism"] == mech]
            dm = dd[dd["mechanism"] == mech]
            if dm.empty:
                continue
            a_rows.append(
                {
                    "window": wname,
                    "mechanism": mech,
                    "trigger_i_lcl_count": int(tm["trigger_i_lcl"].sum()) if not tm.empty else 0,
                    "trigger_i_ucl_count": int(tm["trigger_i_ucl"].sum()) if not tm.empty else 0,
                    "trigger_mr_ucl_count": int(tm["trigger_mr_ucl"].sum()) if not tm.empty else 0,
                    "stress_amp_count": int(tm["stress_amp"].sum()) if not tm.empty else 0,
                    "upside_extreme_count": int(tm["upside_extreme"].sum()) if not tm.empty else 0,
                    "sold_count": int(tm["sold"].sum()) if not tm.empty else 0,
                    "avg_turnover_ratio": float(dm["turnover_ratio"].mean()),
                    "avg_cash_ratio": float((dm["cash"] / dm["equity"]).mean()),
                    "avg_n_positions": float(dm["n_positions"].mean()),
                    "max_drawdown": float(dm["drawdown"].min()),
                }
            )
    df_audit = pd.DataFrame(a_rows)
    p_audit = out_for_data / "audit_sell_triggers_windows.parquet"
    df_audit.to_parquet(p_audit, index=False)

    # Relatório forense
    rel = out_for / "relatorio_fases_m0_m1.md"
    lines = [
        "# Task 015 - Relatorio forense por fases (M0 vs M1)",
        "",
        "## Janela canônica",
        f"- Todas as séries usadas passaram no assert: {CANON_START.date()}..{CANON_END.date()}",
        "",
        "## Evidências principais",
        f"- `weekly_holdings_m0.parquet`: {p_wh_m0}",
        f"- `weekly_holdings_m1.parquet`: {p_wh_m1}",
        f"- `contribuicao_ticker_fases.parquet`: {p_ct}",
        f"- `contribuicao_setor_assetclass_fases.parquet`: {p_sa}",
        f"- `audit_sell_triggers_windows.parquet`: {p_audit}",
        "",
        "## Síntese por janela",
    ]
    for wname in windows:
        lines.append(f"### {wname}")
        for mech in ["M0", "M1"]:
            dm = daily_all[mech]
            ws, we = windows[wname]
            sw = dm[(dm["date"] >= ws) & (dm["date"] <= we)]
            if sw.empty:
                continue
            ret = float(sw["equity"].iloc[-1] / sw["equity"].iloc[0] - 1.0)
            mdd = float(sw["drawdown"].min())
            lines.append(f"- {mech}: retorno={ret:.4f}, max_drawdown={mdd:.4f}")
        am = df_audit[df_audit["window"] == wname]
        if not am.empty:
            for mech in ["M0", "M1"]:
                r = am[am["mechanism"] == mech]
                if r.empty:
                    continue
                rr = r.iloc[0]
                lines.append(
                    f"- {mech} audit: sold={int(rr['sold_count'])}, i_ucl={int(rr['trigger_i_ucl_count'])}, "
                    f"mr_ucl={int(rr['trigger_mr_ucl_count'])}, cash_ratio={rr['avg_cash_ratio']:.4f}, "
                    f"turnover={rr['avg_turnover_ratio']:.4f}"
                )
        lines.append("")
    rel.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Proposta M3
    p_prop = Path("/home/wilson/CEP_COMPRA/docs/emendas/PROPOSTA_M3_CEP_COMPRA_V1_4_NAO_INTEGRADA.md")
    p_prop.parent.mkdir(parents=True, exist_ok=True)
    p_prop.write_text(
        "\n".join(
            [
                "# PROPOSTA M3 CEP_COMPRA v1.4 (não integrada)",
                "",
                "## Objetivo",
                "Combinar critérios de ranking já existentes no pipeline, preservando gating sob controle.",
                "",
                "## Ranking M3 (somente features existentes)",
                "- score_m3 = z(score_m0) + z(ret_lookback_62) - z(vol_lookback_62)",
                "- score_m0: média de xt no lookback (base M0)",
                "- ret_lookback_62: soma de xt no lookback",
                "- vol_lookback_62: desvio-padrão de xt no lookback",
                "",
                "## Invariantes mantidos",
                "- volume > 0 em pelo menos 50/62 dias (excludente)",
                "- uma classe por empresa",
                "- cap por setor 20% (UNKNOWN conta)",
                "- mix B3 entre 50% e 80%",
                "- gating de venda com upside_extreme e stress_amp sob controle",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Backtest comparativo M0/M1/M3
    metrics = []
    for mech in ["M0", "M1", "M3"]:
        d = daily_all[mech]
        metrics.append(
            {
                "mechanism": mech,
                "equity_final": float(d["equity"].iloc[-1]),
                "total_return": float(d["equity"].iloc[-1] / d["equity"].iloc[0] - 1.0),
                "max_drawdown": float(d["drawdown"].min()),
                "avg_cash_ratio": float((d["cash"] / d["equity"]).mean()),
                "avg_n_positions": float(d["n_positions"].mean()),
            }
        )
    df_metrics = pd.DataFrame(metrics).sort_values("equity_final", ascending=False).reset_index(drop=True)
    df_metrics["rank"] = range(1, len(df_metrics) + 1)
    p_metrics = out_bt_con / "metricas_consolidadas.parquet"
    df_metrics.to_parquet(p_metrics, index=False)

    # Plot using local benchmark from task_012 aligned series
    bmk = pd.read_parquet(p_aligned_012)[["date", "cdi_index_norm", "sp500_index_norm", "bvsp_index_norm"]].copy()
    bmk["date"] = pd.to_datetime(bmk["date"])
    for mech in ["M0", "M1", "M3"]:
        d = daily_all[mech][["date", "equity"]].copy()
        d[f"{mech}_index"] = d["equity"] / float(d["equity"].iloc[0])
        bmk = bmk.merge(d[["date", f"{mech}_index"]], on="date", how="left")
    bmk = bmk.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bmk["date"], y=bmk["M0_index"], mode="lines", name="M0"))
    fig.add_trace(go.Scatter(x=bmk["date"], y=bmk["M1_index"], mode="lines", name="M1"))
    fig.add_trace(go.Scatter(x=bmk["date"], y=bmk["M3_index"], mode="lines", name="M3"))
    fig.add_trace(go.Scatter(x=bmk["date"], y=bmk["cdi_index_norm"], mode="lines", name="CDI"))
    fig.add_trace(go.Scatter(x=bmk["date"], y=bmk["sp500_index_norm"], mode="lines", name="S&P500"))
    fig.add_trace(go.Scatter(x=bmk["date"], y=bmk["bvsp_index_norm"], mode="lines", name="^BVSP"))
    fig.update_layout(
        title="Task 015 - M0 vs M1 vs M3 vs CDI vs SP500 vs BVSP (base 1.0)",
        xaxis_title="Data",
        yaxis_title="Índice normalizado",
        template="plotly_white",
    )
    p_plot = out_bt_plots / "m0_vs_m1_vs_m3_vs_cdi_sp500_bvsp.html"
    fig.write_html(str(p_plot), include_plotlyjs="cdn")

    # manifests/hashes
    manifest_for = {
        "task_id": "TASK_CEP_COMPRA_015_FORENSICA_FASES_M0_M1_E_PROPOSTA_M3_COM_GATING_SOB_CONTROLE",
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "outputs": {
            "relatorio_fases": str(rel),
            "weekly_holdings_m0": str(p_wh_m0),
            "weekly_holdings_m1": str(p_wh_m1),
            "contribuicao_ticker_fases": str(p_ct),
            "contribuicao_setor_assetclass_fases": str(p_sa),
            "audit_sell_triggers_windows": str(p_audit),
        },
    }
    (out_for / "manifest.json").write_text(json.dumps(manifest_for, indent=2, ensure_ascii=False), encoding="utf-8")
    hashes_for = {str(p.relative_to(out_for)): sha256_file(p) for p in sorted([x for x in out_for.rglob("*") if x.is_file()])}
    (out_for / "hashes.json").write_text(json.dumps(hashes_for, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_bt = {
        "task_id": "TASK_CEP_COMPRA_015_FORENSICA_FASES_M0_M1_E_PROPOSTA_M3_COM_GATING_SOB_CONTROLE",
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "outputs": {
            "metricas_consolidadas": str(p_metrics),
            "plot_html": str(p_plot),
        },
    }
    (out_bt / "manifest.json").write_text(json.dumps(manifest_bt, indent=2, ensure_ascii=False), encoding="utf-8")
    hashes_bt = {str(p.relative_to(out_bt)): sha256_file(p) for p in sorted([x for x in out_bt.rglob("*") if x.is_file()])}
    (out_bt / "hashes.json").write_text(json.dumps(hashes_bt, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TASK015] run_id={run_id}")
    print(f"[TASK015] forensics_root={out_for}")
    print(f"[TASK015] m3_backtest_root={out_bt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
