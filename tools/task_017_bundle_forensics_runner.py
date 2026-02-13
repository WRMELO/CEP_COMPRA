#!/usr/bin/env python3
import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


CANON_START = pd.Timestamp("2018-07-01")
CANON_END = pd.Timestamp("2025-12-31")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_window(df: pd.DataFrame, date_col: str, name: str) -> None:
    d = pd.to_datetime(df[date_col], errors="coerce")
    if d.min() < CANON_START or d.max() > CANON_END:
        raise RuntimeError(f"FAIL_WINDOW {name}: min={d.min()} max={d.max()}")


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
    return pd.DataFrame(rows, columns=["ticker", "date", "volume"]).sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")


def select_tickers(
    mech: str,
    cand_df: pd.DataFrame,
    target_n: int,
    min_volume_days: int,
) -> Tuple[List[str], List[Dict]]:
    logs: List[Dict] = []
    if cand_df.empty:
        return [], logs

    work = cand_df.copy()
    work["liquidity_ok"] = work["liquidity_days"] >= min_volume_days
    work["sector"] = work["sector"].fillna("UNKNOWN").replace("", "UNKNOWN")

    if mech == "M0":
        ranked = work.sort_values(["score_m0", "ret_lookback", "ticker"], ascending=[False, False, True])
        selected = ranked["ticker"].tolist()[:target_n]
        for t in selected:
            logs.append({"ticker": t, "decision": "selected", "reason": "ranking_m0"})
        return selected, logs

    work = work[work["liquidity_ok"]].copy()
    work = (
        work.sort_values(["ret_lookback", "score_m0", "ticker"], ascending=[False, False, True])
        .drop_duplicates(subset=["company_key"], keep="first")
        .copy()
    )
    if mech == "M1":
        work = work.sort_values(["score_m0", "ret_lookback", "ticker"], ascending=[False, False, True])
    else:
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
        logs.append({"ticker": t, "decision": "selected", "reason": phase})

    for _, row in work[work["asset_class"] == "B3"].iterrows():
        if len(chosen) >= target_n or b3_count >= 5:
            break
        ok, reason = can_add(row)
        if ok:
            add(row, "min_b3")
        else:
            logs.append({"ticker": row["ticker"], "decision": "skipped", "reason": reason})

    for _, row in work[work["asset_class"] == "BDR"].iterrows():
        if len(chosen) >= target_n or bdr_count >= 2:
            break
        if row["ticker"] in chosen:
            continue
        ok, reason = can_add(row)
        if ok:
            add(row, "min_bdr")
        else:
            logs.append({"ticker": row["ticker"], "decision": "skipped", "reason": reason})

    for _, row in work.iterrows():
        if len(chosen) >= target_n:
            break
        if row["ticker"] in chosen:
            continue
        ok, reason = can_add(row)
        if ok:
            add(row, "fill")
        else:
            logs.append({"ticker": row["ticker"], "decision": "skipped", "reason": reason})

    return chosen[:target_n], logs


def main() -> int:
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(f"/home/wilson/CEP_COMPRA/outputs/reports/task_017/{run_id}")
    out_data = out_root / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    # Validate base inputs exist
    p_for = Path("/home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309")
    p_bt_m3 = Path("/home/wilson/CEP_COMPRA/outputs/backtests/task_015_m3/run_20260212_121309")
    p_bt_012 = Path("/home/wilson/CEP_COMPRA/outputs/backtests/task_012/run_20260212_114129")
    for p in [p_for, p_bt_m3, p_bt_012]:
        if not p.exists():
            raise RuntimeError(f"Input run ausente: {p}")

    # Core data
    p_base = Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/base_operacional/base_operacional_xt.csv")
    p_limits = Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/merged/limits_per_ticker.csv")
    p_master = Path("/home/wilson/CEP_NA_BOLSA/outputs/experimentos/fase1_calibracao/exp/20260209/dataset_sizing/master_states.csv")
    p_ssot_a = Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/acoes/b3/20260204/ssot_acoes_b3.csv")
    p_ssot_b = Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/bdr/b3/20260204/ssot_bdr_b3.csv")
    p_setor = Path("/home/wilson/CEP_COMPRA/outputs/ssot/setores/ssot_latest/setores_ticker_latest.parquet")
    p_sizing = Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos/060_selected_config_sizing_v2.json")
    for p in [p_base, p_limits, p_master, p_ssot_a, p_ssot_b, p_setor, p_sizing]:
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

    # Precompute rolling stats
    xbar = {}
    rroll = {}
    vol62 = {}
    for t in xt.columns:
        n = int(df_limits.loc[t].get("n", 3)) if t in df_limits.index and pd.notna(df_limits.loc[t].get("n", None)) else 3
        s = xt[t]
        xbar[t] = s.rolling(window=max(2, n), min_periods=max(2, n)).mean()
        rroll[t] = s.rolling(window=max(2, n), min_periods=max(2, n)).max() - s.rolling(window=max(2, n), min_periods=max(2, n)).min()
        vol62[t] = s.rolling(window=62, min_periods=20).std()
    xbar_df = pd.DataFrame(xbar, index=xt.index)
    r_df = pd.DataFrame(rroll, index=xt.index)
    vol62_df = pd.DataFrame(vol62, index=xt.index)

    windows = {
        "W1": (pd.Timestamp("2018-07-01"), pd.Timestamp("2021-06-30")),
        "W2": (pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
        "W3": (pd.Timestamp("2024-09-01"), pd.Timestamp("2025-11-30")),
    }
    events = {
        "queda_agosto_2021": (pd.Timestamp("2021-08-01"), pd.Timestamp("2021-08-31")),
        "queda_maio_2022": (pd.Timestamp("2022-05-01"), pd.Timestamp("2022-05-31")),
        "salto_2025_11_07": (pd.Timestamp("2025-11-07"), pd.Timestamp("2025-11-07")),
    }

    daily_mech = {}
    holdings_weekly = []
    ledger = []
    mtm_rows = []

    for mech in ["M0", "M1", "M3"]:
        cash = 1.0
        pos: Dict[str, float] = {}
        prev_xt: Dict[str, float] = {}
        daily_rows = []

        for d in trading_dates:
            state = str(master_state[d])
            row_xt = xt.loc[d]
            prev_equity = cash + sum(pos.values())
            prev_pos = pos.copy()

            # mark-to-market and record mtm
            for t in list(pos.keys()):
                rv = row_xt.get(t)
                if pd.notna(rv):
                    old = pos[t]
                    new = old * math.exp(float(rv))
                    pos[t] = new
                    mtm_rows.append(
                        {
                            "date": d,
                            "mechanism": mech,
                            "ticker": t,
                            "position_value_prev": float(old),
                            "position_value_curr": float(new),
                            "ret_xt": float(rv),
                            "pnl_brl": float(new - old),
                            "asset_class": asset_class_map.get(t, "UNKNOWN"),
                            "sector": sector_map.get(t, "UNKNOWN"),
                        }
                    )

            # daily sells
            if "PRESERV" in state.upper():
                for t, v in list(pos.items()):
                    cash += v
                    ledger.append(
                        {
                            "date": d,
                            "mechanism": mech,
                            "action": "SELL",
                            "ticker": t,
                            "qty": None,
                            "price": None,
                            "notional": float(v),
                            "reason": "MASTER_PRESERVACAO_TOTAL",
                            "master_state": state,
                        }
                    )
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
                        notional = pos[t]
                        cash += pos.pop(t)
                        reason = "I_LCL" if sell_i_lcl else ("I_UCL" if sell_i_ucl else ("MR_UCL" if sell_mr else "STRESS_AMP"))
                        ledger.append(
                            {
                                "date": d,
                                "mechanism": mech,
                                "action": "SELL",
                                "ticker": t,
                                "qty": None,
                                "price": None,
                                "notional": float(notional),
                                "reason": reason,
                                "master_state": state,
                            }
                        )
                    prev_xt[t] = float(r_now)

            # weekly buy
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
                selected, log_sel = select_tickers(mech, cand_df, target_n=10, min_volume_days=50)

                total_equity = cash + sum(pos.values())
                slot = total_equity / 10.0
                cap_value = total_equity * w_cap
                for t in selected:
                    cur = pos.get(t, 0.0)
                    desired = min(slot, cap_value)
                    need = max(0.0, desired - cur)
                    alloc = min(need, cash)
                    if alloc > 0:
                        pos[t] = cur + alloc
                        cash -= alloc
                        ledger.append(
                            {
                                "date": d,
                                "mechanism": mech,
                                "action": "BUY",
                                "ticker": t,
                                "qty": None,
                                "price": None,
                                "notional": float(alloc),
                                "reason": "WEEKLY_BUY",
                                "master_state": state,
                            }
                        )

                tot = cash + sum(pos.values())
                for t, v in sorted(pos.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                    holdings_weekly.append(
                        {
                            "event_date": d,
                            "mechanism": mech,
                            "ticker": t,
                            "value_brl": float(v),
                            "weight": float(v / tot) if tot else 0.0,
                            "sector": sector_map.get(t, "UNKNOWN"),
                            "asset_class": asset_class_map.get(t, "UNKNOWN"),
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
                    "daily_return": float(equity / prev_equity - 1.0) if prev_equity > 0 else 0.0,
                }
            )

        df_daily = pd.DataFrame(daily_rows).sort_values("date")
        df_daily["rolling_max"] = df_daily["equity"].cummax()
        df_daily["drawdown"] = (df_daily["equity"] / df_daily["rolling_max"]) - 1.0
        assert_window(df_daily, "date", f"daily_{mech}")
        daily_mech[mech] = df_daily

    # Save ledgers, mtm, holdings and daily portfolio by mechanism
    ledger_df = pd.DataFrame(ledger).sort_values(["date", "mechanism", "action", "ticker"])
    mtm_df = pd.DataFrame(mtm_rows).sort_values(["date", "mechanism", "ticker"])
    hold_df = pd.DataFrame(holdings_weekly).sort_values(["event_date", "mechanism", "value_brl"], ascending=[True, True, False])

    for mech in ["M0", "M1", "M3"]:
        l = ledger_df[ledger_df["mechanism"] == mech].copy()
        m = mtm_df[mtm_df["mechanism"] == mech].copy()
        h = hold_df[hold_df["mechanism"] == mech].copy()
        d = daily_mech[mech].copy()
        l.to_parquet(out_data / f"ledger_trades_{mech.lower()}.parquet", index=False)
        m.to_parquet(out_data / f"mtm_daily_by_ticker_{mech.lower()}.parquet", index=False)
        h.to_parquet(out_data / f"holdings_weekly_{mech.lower()}.parquet", index=False)
        d.to_parquet(out_data / f"daily_portfolio_{mech.lower()}.parquet", index=False)

    # Master state daily
    ms = df_master[(df_master["date"] >= CANON_START) & (df_master["date"] <= CANON_END)].copy()
    ms.to_parquet(out_data / "master_state_daily.parquet", index=False)

    # Decomposition windows and events
    decomp_rows = []
    for label, (ws, we) in {**windows, **events}.items():
        sub = mtm_df[(mtm_df["date"] >= ws) & (mtm_df["date"] <= we)].copy()
        if sub.empty:
            continue
        g = (
            sub.groupby(["mechanism", "ticker", "sector", "asset_class"], as_index=False)
            .agg(pnl_brl=("pnl_brl", "sum"))
        )
        g["label"] = label
        decomp_rows.append(g)
    decomp = pd.concat(decomp_rows, ignore_index=True) if decomp_rows else pd.DataFrame()
    decomp_win = decomp[decomp["label"].isin(windows.keys())].copy()
    decomp_evt = decomp[decomp["label"].isin(events.keys())].copy()
    decomp_win.to_parquet(out_data / "decomp_pnl_windows.parquet", index=False)
    decomp_evt.to_parquet(out_data / "decomp_pnl_event_months.parquet", index=False)

    # Markdown causal report
    md_lines = []
    md_lines.append("# Pacote forense ampliado M0/M1/M3")
    md_lines.append("")
    md_lines.append("## Escopo e causalidade")
    md_lines.append("- Todas as afirmações abaixo referenciam tabelas Parquet deste pacote.")
    md_lines.append("- Janela canônica validada: 2018-07-01..2025-12-31.")
    md_lines.append("")
    for label, (ws, we) in {**windows, **events}.items():
        md_lines.append(f"## {label} ({ws.date()}..{we.date()})")
        dsub = decomp[(decomp["label"] == label)].copy()
        if dsub.empty:
            md_lines.append("- Sem dados no recorte.")
            md_lines.append("")
            continue
        for mech in ["M0", "M1", "M3"]:
            m = dsub[dsub["mechanism"] == mech].copy()
            if m.empty:
                continue
            top_pos = m.sort_values("pnl_brl", ascending=False).head(3)[["ticker", "pnl_brl"]]
            top_neg = m.sort_values("pnl_brl", ascending=True).head(3)[["ticker", "pnl_brl"]]
            md_lines.append(f"- {mech} top contribuintes: " + ", ".join([f"{r.ticker}({r.pnl_brl:.4f})" for r in top_pos.itertuples(index=False)]))
            md_lines.append(f"- {mech} top detratores: " + ", ".join([f"{r.ticker}({r.pnl_brl:.4f})" for r in top_neg.itertuples(index=False)]))
            daily_slice = daily_mech[mech][(daily_mech[mech]["date"] >= ws) & (daily_mech[mech]["date"] <= we)].copy()
            trades_slice = ledger_df[(ledger_df["mechanism"] == mech) & (ledger_df["date"] >= ws) & (ledger_df["date"] <= we)].copy()
            avg_cash_ratio = float((daily_slice["cash"] / daily_slice["equity"]).mean()) if not daily_slice.empty else 0.0
            avg_n_positions = float(daily_slice["n_positions"].mean()) if not daily_slice.empty else 0.0
            avg_equity = float(daily_slice["equity"].mean()) if not daily_slice.empty else 0.0
            days = max(1, int(len(daily_slice)))
            turnover_ratio = float(trades_slice["notional"].sum()) / float(avg_equity * days) if avg_equity > 0 else 0.0
            buy_count = int((trades_slice["action"] == "BUY").sum())
            sell_count = int((trades_slice["action"] == "SELL").sum())
            sell_reasons = (
                trades_slice[trades_slice["action"] == "SELL"]["reason"].value_counts().head(2).to_dict()
                if not trades_slice.empty else {}
            )
            md_lines.append(
                f"- {mech} risco/execução: turnover={turnover_ratio:.4f}, cash_ratio={avg_cash_ratio:.4f}, "
                f"n_positions={avg_n_positions:.2f}, buys={buy_count}, sells={sell_count}"
            )
            if sell_reasons:
                sr = ", ".join([f"{k}:{v}" for k, v in sell_reasons.items()])
                md_lines.append(f"- {mech} decisões de venda dominantes: {sr}")
        md_lines.append("- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.")
        md_lines.append("- Evidência operacional: `data/ledger_trades_*.parquet`, `data/daily_portfolio_*.parquet`.")
        md_lines.append("")

    md_lines.append("## Diferenças operacionais (giro/proteção)")
    for mech in ["M0", "M1", "M3"]:
        d = daily_mech[mech]
        md_lines.append(
            f"- {mech}: avg_cash={float((d['cash']/d['equity']).mean()):.4f}, avg_n_positions={float(d['n_positions'].mean()):.2f}, max_drawdown={float(d['drawdown'].min()):.4f}"
        )
    md_lines.append("- Evidência: `data/master_state_daily.parquet`, `data/ledger_trades_*.parquet`, `data/holdings_weekly_*.parquet`.")
    md_lines.append("")

    p_md = out_root / "pacote_forense_ampliado.md"
    p_md.write_text("\n".join(md_lines), encoding="utf-8")

    # Optional zip for chat upload
    zip_base = out_root / "data_bundle_task_017"
    shutil.make_archive(str(zip_base), "zip", root_dir=str(out_data))

    # manifest/hashes
    manifest = {
        "task_id": "TASK_CEP_COMPRA_017_BUNDLE_FORENSE_AMPLIADO_M0_M1_M3_PARA_EXPLICAR_FASES",
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "inputs": {
            "run_forensics_base": str(p_for),
            "run_backtest_m3": str(p_bt_m3),
            "run_backtest_m0_m1_v12": str(p_bt_012),
        },
        "outputs": {
            "markdown": str(p_md),
            "data_dir": str(out_data),
            "zip": str(zip_base) + ".zip",
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    files = [p for p in out_root.rglob("*") if p.is_file()]
    hashes = {str(p.relative_to(out_root)): sha256_file(p) for p in sorted(files)}
    (out_root / "hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TASK017] run_id={run_id}")
    print(f"[TASK017] output_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
