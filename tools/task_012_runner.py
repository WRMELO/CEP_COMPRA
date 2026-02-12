#!/usr/bin/env python3
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_trading_day_by_week(dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
    out: List[pd.Timestamp] = []
    seen = set()
    for d in dates:
        key = (d.isocalendar().year, d.isocalendar().week)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def fetch_cdi_sgs(start_br: str, end_br: str) -> pd.DataFrame:
    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
        f"?formato=json&dataInicial={start_br}&dataFinal={end_br}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["rate"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df[["date", "rate"]].dropna().sort_values("date")
    df["factor"] = 1.0 + (df["rate"] / 100.0)
    df["cdi_index"] = df["factor"].cumprod()
    return df[["date", "cdi_index"]]


def fetch_sp500_stooq() -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=%5Espx&i=d"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    rows = []
    for line in r.text.strip().splitlines()[1:]:
        p = line.split(",")
        if len(p) >= 5:
            rows.append((p[0], p[4]))
    df = pd.DataFrame(rows, columns=["date", "sp500_close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sp500_close"] = pd.to_numeric(df["sp500_close"], errors="coerce")
    df = df.dropna().sort_values("date")
    df["sp500_index"] = df["sp500_close"] / float(df["sp500_close"].iloc[0])
    return df[["date", "sp500_index"]]


def fetch_bvsp_fallback(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/precos_brutos/ibov/brapi/20260204/precos_brutos_ibov.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["symbol"] == "^BVSP"].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    df["bvsp_index"] = df["close"] / float(df["close"].iloc[0])
    return df[["date", "bvsp_index"]]


def load_volume_cache(
    tickers: List[str],
    asset_class_map: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    base = Path("/home/wilson/CEP_NA_BOLSA/data/raw/market")
    for t in tickers:
        cls = asset_class_map.get(t, "UNKNOWN")
        if cls == "B3":
            p = base / "acoes/brapi/20260204" / f"{t}.json"
        else:
            p = base / "bdr/brapi/20260204" / f"{t}.json"
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            hist = payload.get("results", [{}])[0].get("historicalDataPrice", [])
            for h in hist:
                d = pd.to_datetime(h.get("date"), unit="s", errors="coerce")
                v = h.get("volume")
                if pd.isna(d):
                    continue
                rows.append((t, d.normalize(), float(v) if v is not None else 0.0))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "volume"])
    vol = pd.DataFrame(rows, columns=["ticker", "date", "volume"])
    vol = vol.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return vol


def build_sector_ssot(
    df_ssot_a: pd.DataFrame,
    df_ssot_b: pd.DataFrame,
    run_date: str,
) -> Tuple[pd.DataFrame, Dict]:
    # BRAPI local raw (sectors geralmente ausentes): tentativa primária
    brapi_rows = []
    raw_base = Path("/home/wilson/CEP_NA_BOLSA/data/raw/market")

    # Ações
    for _, r in df_ssot_a.iterrows():
        t = str(r["ticker"])
        p = raw_base / "acoes/brapi/20260204" / f"{t}.json"
        sector = None
        subsector = None
        company = None
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                res0 = payload.get("results", [{}])[0]
                sector = res0.get("sector")
                subsector = res0.get("subSector")
                company = res0.get("longName") or res0.get("shortName")
            except Exception:
                pass
        brapi_rows.append(
            {
                "ticker": t,
                "asset_class": "B3",
                "company_name": company if company else str(r.get("trading_name", "")),
                "company_group_key": str(r.get("code_cvm", "")) if pd.notna(r.get("code_cvm", None)) else "",
                "sector_raw": sector,
                "subsector_raw": subsector,
                "source_primary": "BRAPI_LOCAL_RAW",
            }
        )

    # BDR
    bdr_sym_col = "ticker_bdr" if "ticker_bdr" in df_ssot_b.columns else "ticker"
    for _, r in df_ssot_b.iterrows():
        t = str(r[bdr_sym_col])
        p = raw_base / "bdr/brapi/20260204" / f"{t}.json"
        sector = None
        subsector = None
        company = None
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                res0 = payload.get("results", [{}])[0]
                sector = res0.get("sector")
                subsector = res0.get("subSector")
                company = res0.get("longName") or res0.get("shortName")
            except Exception:
                pass
        brapi_rows.append(
            {
                "ticker": t,
                "asset_class": "BDR",
                "company_name": company if company else str(r.get("empresa", "")),
                "company_group_key": str(r.get("isin", "")),
                "sector_raw": sector,
                "subsector_raw": subsector,
                "source_primary": "BRAPI_LOCAL_RAW",
            }
        )

    ssot = pd.DataFrame(brapi_rows)

    # Fallback Ações: segment do raw SSOT B3 companies
    raw_companies = Path("/home/wilson/CEP_NA_BOLSA/data/raw/ssot/acoes/b3/20260204/ACOES_B3_20260204.json")
    company_seg = {}
    if raw_companies.exists():
        payload = json.loads(raw_companies.read_text(encoding="utf-8"))
        for c in payload.get("companies", []):
            key = str(c.get("issuingCompany", ""))
            company_seg[key] = c.get("segment")

    a_iss = df_ssot_a.set_index("ticker")["issuing_company"].astype(str).to_dict()
    bdr_setor = {}
    if "setor" in df_ssot_b.columns:
        for _, r in df_ssot_b.iterrows():
            t = str(r[bdr_sym_col])
            bdr_setor[t] = r.get("setor")

    source_used = []
    sector_norm = []
    subsector_norm = []
    for _, r in ssot.iterrows():
        t = r["ticker"]
        cls = r["asset_class"]
        sec = r["sector_raw"]
        sub = r["subsector_raw"]
        src = "BRAPI_LOCAL_RAW"
        if (sec is None or str(sec).strip() == "" or str(sec).strip() == "-") and cls == "B3":
            iss = a_iss.get(t, "")
            sec = company_seg.get(iss)
            src = "B3_LISTED_COMPANIES_SEGMENT_FALLBACK"
        if (sec is None or str(sec).strip() == "" or str(sec).strip() == "-") and cls == "BDR":
            sec = bdr_setor.get(t)
            src = "SSOT_BDR_SETOR_FALLBACK"
        if sec is None or str(sec).strip() == "" or str(sec).strip() == "-":
            sec = "UNKNOWN"
            src = src + "_TO_UNKNOWN"
        if sub is None or str(sub).strip() == "" or str(sub).strip() == "-":
            sub = "UNKNOWN"
        source_used.append(src)
        sector_norm.append(str(sec))
        subsector_norm.append(str(sub))
    ssot["sector"] = sector_norm
    ssot["subsector"] = subsector_norm
    ssot["source_used"] = source_used

    out_day = Path(f"/home/wilson/CEP_COMPRA/outputs/ssot/setores/{run_date}")
    out_day.mkdir(parents=True, exist_ok=True)
    out_latest = Path("/home/wilson/CEP_COMPRA/outputs/ssot/setores/ssot_latest")
    out_latest.mkdir(parents=True, exist_ok=True)

    p_parquet = out_day / "setores_ticker.parquet"
    ssot.to_parquet(p_parquet, index=False)
    p_latest = out_latest / "setores_ticker_latest.parquet"
    ssot.to_parquet(p_latest, index=False)

    coverage = (
        ssot.assign(known=ssot["sector"] != "UNKNOWN")
        .groupby("asset_class", as_index=False)
        .agg(total=("ticker", "count"), known=("known", "sum"))
    )
    coverage["unknown"] = coverage["total"] - coverage["known"]
    coverage["known_pct"] = coverage["known"] / coverage["total"]
    p_cov = out_day / "coverage_by_asset_class.parquet"
    coverage.to_parquet(p_cov, index=False)

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_date": run_date,
        "source_primary": "BRAPI_LOCAL_RAW_FILES",
        "fallbacks": [
            "B3 listed companies segment (acoes)",
            "SSOT BDR setor column (bdr)",
            "UNKNOWN default",
        ],
        "outputs": {
            "setores_ticker_parquet": str(p_parquet),
            "setores_ticker_latest_parquet": str(p_latest),
            "coverage_by_asset_class_parquet": str(p_cov),
        },
    }
    p_manifest = out_day / "manifest.json"
    p_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    hashes = {
        "setores_ticker.parquet": sha256_file(p_parquet),
        "setores_ticker_latest.parquet": sha256_file(p_latest),
        "coverage_by_asset_class.parquet": sha256_file(p_cov),
        "manifest.json": sha256_file(p_manifest),
    }
    p_hashes = out_day / "hashes.json"
    p_hashes.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")
    return ssot, {"day_dir": str(out_day), "latest_file": str(p_latest), "manifest": str(p_manifest), "hashes": str(p_hashes)}


def main() -> int:
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    run_date = now.strftime("%Y%m%d")
    out_root = Path(f"/home/wilson/CEP_COMPRA/outputs/backtests/task_012/{run_id}")
    out_raw = out_root / "raw"
    out_con = out_root / "consolidated"
    out_plots = out_root / "plots"
    out_logs = out_root / "logs"
    for p in [out_root, out_raw, out_con, out_plots, out_logs]:
        p.mkdir(parents=True, exist_ok=True)

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
    sizing = json.loads(p_sizing.read_text(encoding="utf-8"))
    w_cap = float(sizing.get("w_cap", 0.15))

    bdr_symbol_col = "ticker_bdr" if "ticker_bdr" in df_ssot_b.columns else "ticker"
    b3_tickers = set(df_ssot_a["ticker"].astype(str))
    bdr_tickers = set(df_ssot_b[bdr_symbol_col].astype(str))
    universe = b3_tickers | bdr_tickers

    asset_class_map: Dict[str, str] = {}
    for t in b3_tickers:
        asset_class_map[t] = "B3"
    for t in bdr_tickers:
        asset_class_map[t] = "BDR"

    # company key map
    company_key: Dict[str, str] = {}
    for _, r in df_ssot_a.iterrows():
        t = str(r["ticker"])
        ck = str(r.get("code_cvm", "")) if pd.notna(r.get("code_cvm", None)) else ""
        if not ck:
            ck = str(r.get("issuing_company", ""))
        if not ck:
            ck = t[:4]
        company_key[t] = ck
    for _, r in df_ssot_b.iterrows():
        t = str(r[bdr_symbol_col])
        ck = str(r.get("isin", ""))
        if not ck:
            ck = str(r.get("ticker", ""))
        if not ck:
            ck = t[:4]
        company_key[t] = ck

    # S2 SSOT setor
    ssot_setor, setor_meta = build_sector_ssot(df_ssot_a, df_ssot_b, run_date=run_date)
    sector_map = ssot_setor.set_index("ticker")["sector"].to_dict()

    # volume cache (para regra 50/62)
    vol = load_volume_cache(sorted(list(universe)), asset_class_map)
    vol_pivot = vol.pivot(index="date", columns="ticker", values="volume").sort_index() if not vol.empty else pd.DataFrame()

    # Base de backtest
    df_base = df_base[df_base["ticker"].isin(universe)].copy().sort_values(["date", "ticker"])
    xt = df_base.pivot(index="date", columns="ticker", values="xt").sort_index()
    master_state = dict(zip(df_master["date"], df_master["state"]))
    start_trade = pd.Timestamp("2018-07-01")
    end_trade = pd.Timestamp("2025-12-31")
    trading_dates = pd.DatetimeIndex([d for d in xt.index if d in master_state and start_trade <= d <= end_trade])
    buy_dates = set(first_trading_day_by_week(list(trading_dates)))

    # rolling xbar/r por ticker para regra upside/sell
    xbar_map = {}
    r_map = {}
    for t in xt.columns:
        if t in df_limits.index:
            n = int(df_limits.loc[t].get("n", 3))
        else:
            n = 3
        s = xt[t]
        xbar_map[t] = s.rolling(window=max(2, n), min_periods=max(2, n)).mean()
        r_map[t] = s.rolling(window=max(2, n), min_periods=max(2, n)).max() - s.rolling(window=max(2, n), min_periods=max(2, n)).min()
    xbar_df = pd.DataFrame(xbar_map, index=xt.index)
    r_df = pd.DataFrame(r_map, index=xt.index)

    lookback = 62
    min_volume_days = 50
    target_n = 10
    min_b3, max_b3 = 5, 8
    min_bdr, max_bdr = 2, 5
    sector_cap_positions = 2  # 20% de 10 posições

    metrics_rows = []
    m0m1_eq = {}

    for mech in ["M0", "M1"]:
        cash = 1.0
        pos: Dict[str, float] = {}
        prev_xt: Dict[str, float] = {}
        equity_rows = []
        decision_rows = []

        for d in trading_dates:
            row_xt = xt.loc[d]
            # mtm
            for t in list(pos.keys()):
                r = row_xt.get(t)
                if pd.notna(r):
                    pos[t] *= math.exp(float(r))

            state = str(master_state[d])
            if "PRESERV" in state.upper():
                cash += sum(pos.values())
                pos.clear()
            else:
                # SELL patch com exceção upside
                for t in list(pos.keys()):
                    if t not in df_limits.index:
                        continue
                    r_now = row_xt.get(t)
                    if pd.isna(r_now):
                        continue
                    lim = df_limits.loc[t]
                    mr = abs(float(r_now) - float(prev_xt.get(t, 0.0)))
                    xbar_t = xbar_df.loc[d, t] if (t in xbar_df.columns and d in xbar_df.index) else float("nan")
                    r_t = r_df.loc[d, t] if (t in r_df.columns and d in r_df.index) else float("nan")
                    upside_extreme = (
                        (pd.notna(xbar_t) and pd.notna(lim.get("xbar_ucl", None)) and float(xbar_t) > float(lim["xbar_ucl"]))
                        or (pd.notna(r_now) and pd.notna(lim.get("i_ucl", None)) and float(r_now) > float(lim["i_ucl"]))
                    )
                    stress_amp = (
                        pd.notna(r_t)
                        and pd.notna(lim.get("r_ucl", None))
                        and float(r_t) > float(lim["r_ucl"])
                        and (not upside_extreme)
                    )
                    sell_i_lcl = float(r_now) < float(lim["i_lcl"])
                    sell_i_ucl = float(r_now) > float(lim["i_ucl"]) and (not upside_extreme)
                    sell_mr = mr > float(lim["mr_ucl"]) and (not upside_extreme)
                    sell = sell_i_lcl or sell_i_ucl or sell_mr or bool(stress_amp)
                    if sell:
                        cash += pos.pop(t)
                    prev_xt[t] = float(r_now)

            if d in buy_dates and "RISK_ON" in state:
                hist_xt = xt.loc[:d].tail(lookback)
                if vol_pivot.empty:
                    hist_vol = pd.DataFrame(index=hist_xt.index, columns=hist_xt.columns)
                else:
                    hist_vol = vol_pivot.reindex(index=hist_xt.index, columns=hist_xt.columns)
                mean_k = hist_xt.mean(skipna=True)
                ret_k = hist_xt.sum(skipna=True)
                vol_count = (hist_vol.fillna(0.0) > 0).sum()

                candidates = []
                for t in mean_k.index:
                    if t not in df_limits.index:
                        continue
                    rt = row_xt.get(t)
                    if pd.isna(rt):
                        continue
                    lim = df_limits.loc[t]
                    if float(rt) < float(lim["i_lcl"]) or float(rt) > float(lim["i_ucl"]):
                        continue
                    score = float(mean_k.get(t, float("nan")))
                    retv = float(ret_k.get(t, float("nan")))
                    if pd.isna(score) or pd.isna(retv):
                        continue
                    liquidity_ok = int(vol_count.get(t, 0)) >= min_volume_days
                    cls = asset_class_map.get(t, "UNKNOWN")
                    sec = sector_map.get(t, "UNKNOWN")
                    ck = company_key.get(t, t[:4])
                    candidates.append(
                        {
                            "ticker": t,
                            "score": score,
                            "ret_lookback": retv,
                            "liquidity_days": int(vol_count.get(t, 0)),
                            "liquidity_ok": bool(liquidity_ok),
                            "asset_class": cls,
                            "sector": sec if sec else "UNKNOWN",
                            "company_key": ck,
                        }
                    )

                cand_df = pd.DataFrame(candidates)
                if mech == "M0":
                    ranked = cand_df.sort_values(["score", "ret_lookback", "ticker"], ascending=[False, False, True])
                    target_list = ranked["ticker"].tolist()[:target_n]
                else:
                    # M1 rules
                    ranked = cand_df[cand_df["liquidity_ok"]].copy()
                    # one class per company: keep best ret_lookback
                    ranked = (
                        ranked.sort_values(["ret_lookback", "score", "ticker"], ascending=[False, False, True])
                        .drop_duplicates(subset=["company_key"], keep="first")
                    )
                    ranked = ranked.sort_values(["score", "ret_lookback", "ticker"], ascending=[False, False, True]).reset_index(drop=True)

                    chosen: List[str] = []
                    sec_count: Dict[str, int] = {}
                    b3_count = 0
                    bdr_count = 0

                    def can_add(row) -> bool:
                        nonlocal b3_count, bdr_count
                        cls = row["asset_class"]
                        sec = row["sector"] if row["sector"] else "UNKNOWN"
                        if sec_count.get(sec, 0) >= sector_cap_positions:
                            return False
                        if cls == "B3" and b3_count >= max_b3:
                            return False
                        if cls == "BDR" and bdr_count >= max_bdr:
                            return False
                        return True

                    def add_row(row) -> None:
                        nonlocal b3_count, bdr_count
                        tck = row["ticker"]
                        cls = row["asset_class"]
                        sec = row["sector"] if row["sector"] else "UNKNOWN"
                        chosen.append(tck)
                        sec_count[sec] = sec_count.get(sec, 0) + 1
                        if cls == "B3":
                            b3_count += 1
                        elif cls == "BDR":
                            bdr_count += 1

                    # phase 1: fill minimum B3
                    for _, row in ranked[ranked["asset_class"] == "B3"].iterrows():
                        if len(chosen) >= target_n or b3_count >= min_b3:
                            break
                        if can_add(row):
                            add_row(row)
                    # phase 2: fill minimum BDR
                    for _, row in ranked[ranked["asset_class"] == "BDR"].iterrows():
                        if len(chosen) >= target_n or bdr_count >= min_bdr:
                            break
                        if row["ticker"] in chosen:
                            continue
                        if can_add(row):
                            add_row(row)
                    # phase 3: fill remaining greedily
                    for _, row in ranked.iterrows():
                        if len(chosen) >= target_n:
                            break
                        if row["ticker"] in chosen:
                            continue
                        if can_add(row):
                            add_row(row)

                    target_list = chosen[:target_n]

                total_equity = cash + sum(pos.values())
                slot = total_equity / target_n if target_n else 0.0
                cap_value = total_equity * w_cap

                for t in target_list:
                    cur = pos.get(t, 0.0)
                    desired = min(slot, cap_value)
                    need = max(0.0, desired - cur)
                    alloc = min(need, cash)
                    if alloc > 0:
                        pos[t] = cur + alloc
                        cash -= alloc

                decision_rows.append(
                    {
                        "date": d,
                        "mechanism": mech,
                        "state": state,
                        "selected_tickers": ",".join(target_list),
                        "n_selected": len(target_list),
                    }
                )

            equity = cash + sum(pos.values())
            equity_rows.append({"date": d, "equity": equity, "cash": cash, "n_positions": len(pos)})

        df_eq = pd.DataFrame(equity_rows).sort_values("date")
        df_eq["rolling_max"] = df_eq["equity"].cummax()
        df_eq["drawdown"] = (df_eq["equity"] / df_eq["rolling_max"]) - 1.0
        df_dec = pd.DataFrame(decision_rows)
        df_eq.to_parquet(out_raw / f"{mech}_equity_curve.parquet", index=False)
        df_dec.to_parquet(out_raw / f"{mech}_weekly_decisions.parquet", index=False)

        metrics_rows.append(
            {
                "mechanism": mech,
                "equity_final": float(df_eq["equity"].iloc[-1]),
                "total_return": float(df_eq["equity"].iloc[-1] - 1.0),
                "max_drawdown": float(df_eq["drawdown"].min()),
                "avg_cash_ratio": float((df_eq["cash"] / df_eq["equity"]).mean()),
                "avg_n_positions": float(df_eq["n_positions"].mean()),
            }
        )
        m0m1_eq[mech] = df_eq[["date", "equity"]].rename(columns={"equity": f"{mech}_equity"})

    # Consolidado
    df_metrics = pd.DataFrame(metrics_rows).sort_values("equity_final", ascending=False).reset_index(drop=True)
    df_metrics["rank"] = range(1, len(df_metrics) + 1)
    p_metrics = out_con / "metricas_consolidadas.parquet"
    p_rank = out_con / "ranking_final.parquet"
    df_metrics.to_parquet(p_metrics, index=False)
    df_metrics[["mechanism", "rank", "equity_final", "total_return", "max_drawdown"]].to_parquet(p_rank, index=False)

    # Plot comparativo
    df_plot = m0m1_eq["M0"].merge(m0m1_eq["M1"], on="date", how="inner").sort_values("date")
    for c in ["M0_equity", "M1_equity"]:
        df_plot[f"{c}_idx"] = df_plot[c] / float(df_plot[c].iloc[0])

    start = pd.Timestamp(df_plot["date"].min())
    end = pd.Timestamp(df_plot["date"].max())
    cdi = fetch_cdi_sgs(start.strftime("%d/%m/%Y"), end.strftime("%d/%m/%Y"))
    spx = fetch_sp500_stooq()
    bvsp = fetch_bvsp_fallback(start, end)

    al = df_plot[["date", "M0_equity_idx", "M1_equity_idx"]].copy()
    al = al.merge(cdi, on="date", how="left").merge(spx, on="date", how="left").merge(bvsp, on="date", how="left")
    al = al.sort_values("date")
    for c in ["cdi_index", "sp500_index", "bvsp_index"]:
        al[c] = al[c].ffill().bfill()
        al[f"{c}_norm"] = al[c] / float(al[c].iloc[0])
    p_al = out_con / "series_alinhadas_plot.parquet"
    al.to_parquet(p_al, index=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=al["date"], y=al["M0_equity_idx"], mode="lines", name="M0"))
    fig.add_trace(go.Scatter(x=al["date"], y=al["M1_equity_idx"], mode="lines", name="M1"))
    fig.add_trace(go.Scatter(x=al["date"], y=al["cdi_index_norm"], mode="lines", name="CDI"))
    fig.add_trace(go.Scatter(x=al["date"], y=al["sp500_index_norm"], mode="lines", name="S&P 500"))
    fig.add_trace(go.Scatter(x=al["date"], y=al["bvsp_index_norm"], mode="lines", name="^BVSP"))
    fig.update_layout(
        title="Task 012 - M0 vs M1 vs CDI vs S&P500 vs BVSP (base 1.0)",
        xaxis_title="Data",
        yaxis_title="Indice normalizado",
        template="plotly_white",
    )
    p_plot = out_plots / "m0_vs_m1_vs_cdi_sp500_bvsp.html"
    fig.write_html(str(p_plot), include_plotlyjs="cdn")

    manifest = {
        "task_id": "TASK_CEP_COMPRA_012_EMENDA_V13_REGRAS_E_BACKTEST_M1_COM_SSOT_SETOR",
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "parameters": {
            "lookback_days": lookback,
            "min_volume_days": min_volume_days,
            "portfolio_target_positions": target_n,
            "sector_max_weight": 0.20,
            "b3_min_weight": 0.50,
            "b3_max_weight": 0.80,
            "mechanisms": ["M0", "M1"],
        },
        "ssot_setor": setor_meta,
        "outputs": {
            "metricas_consolidadas_parquet": str(p_metrics),
            "ranking_final_parquet": str(p_rank),
            "plot_html": str(p_plot),
            "series_alinhadas_parquet": str(p_al),
        },
    }
    p_manifest = out_root / "manifest.json"
    p_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    files = [p for p in out_root.rglob("*") if p.is_file()]
    hashes = {str(p.relative_to(out_root)): sha256_file(p) for p in sorted(files)}
    p_hashes = out_root / "hashes.json"
    p_hashes.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TASK012] run_id={run_id}")
    print(f"[TASK012] output_root={out_root}")
    print(f"[TASK012] ssot_setor_day={setor_meta['day_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
