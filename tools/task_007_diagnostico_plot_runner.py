#!/usr/bin/env python3
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_run_dir(base: Path) -> Path:
    runs = sorted([p for p in base.glob("run_*") if p.is_dir()])
    if not runs:
        raise RuntimeError(f"Nenhum run encontrado em {base}")
    return runs[-1]


def fetch_cdi_sgs(start_br: str, end_br: str) -> Tuple[pd.DataFrame, Dict]:
    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
        f"?formato=json&dataInicial={start_br}&dataFinal={end_br}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["cdi_rate_pct"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df[["date", "cdi_rate_pct"]].dropna().sort_values("date")
    df["cdi_factor"] = 1.0 + (df["cdi_rate_pct"] / 100.0)
    df["cdi_index"] = df["cdi_factor"].cumprod()
    meta = {
        "provider": "Banco Central SGS",
        "series_code": 12,
        "url": url,
        "transform": "cdi_factor=1+taxa_pct/100; cdi_index=cumprod(cdi_factor)",
    }
    return df, meta


def fetch_sp500_stooq() -> Tuple[pd.DataFrame, Dict]:
    url = "https://stooq.com/q/d/l/?s=%5Espx&i=d"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    rows = []
    for line in lines[1:]:
        p = line.split(",")
        if len(p) < 5:
            continue
        rows.append((p[0], p[4]))
    df = pd.DataFrame(rows, columns=["date", "sp500_close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sp500_close"] = pd.to_numeric(df["sp500_close"], errors="coerce")
    df = df.dropna().sort_values("date")
    df["sp500_index"] = df["sp500_close"] / float(df["sp500_close"].iloc[0])
    meta = {
        "provider": "Stooq",
        "symbol": "^spx",
        "url": url,
        "transform": "sp500_index=close/close_inicial",
    }
    return df, meta


def fetch_bvsp_with_fallback(start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[pd.DataFrame, Dict]:
    # tentativa 1: BRAPI (pode requerer token)
    brapi_url = "https://brapi.dev/api/quote/%5EBVSP?range=10y&interval=1d"
    try:
        r = requests.get(brapi_url, timeout=30)
        if r.status_code == 200:
            payload = r.json()
            if payload.get("results"):
                hist = payload["results"][0].get("historicalDataPrice", [])
                if hist:
                    df = pd.DataFrame(hist)
                    df["date"] = pd.to_datetime(df["date"], unit="s")
                    df["bvsp_close"] = pd.to_numeric(df["close"], errors="coerce")
                    df = df[["date", "bvsp_close"]].dropna().sort_values("date")
                    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
                    if not df.empty:
                        df["bvsp_index"] = df["bvsp_close"] / float(df["bvsp_close"].iloc[0])
                        return df, {
                            "provider": "BRAPI",
                            "symbol": "^BVSP",
                            "url": brapi_url,
                            "transform": "bvsp_index=close/close_inicial",
                            "fallback_used": False,
                        }
    except Exception:
        pass

    # tentativa 2: Stooq (^bvsp)
    stooq_url = "https://stooq.com/q/d/l/?s=%5Ebvsp&i=d"
    try:
        r = requests.get(stooq_url, timeout=30)
        if r.status_code == 200 and "No data" not in r.text:
            lines = r.text.strip().splitlines()
            rows = []
            for line in lines[1:]:
                p = line.split(",")
                if len(p) < 5:
                    continue
                rows.append((p[0], p[4]))
            df = pd.DataFrame(rows, columns=["date", "bvsp_close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["bvsp_close"] = pd.to_numeric(df["bvsp_close"], errors="coerce")
            df = df.dropna().sort_values("date")
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
            if not df.empty:
                df["bvsp_index"] = df["bvsp_close"] / float(df["bvsp_close"].iloc[0])
                return df, {
                    "provider": "Stooq",
                    "symbol": "^bvsp",
                    "url": stooq_url,
                    "transform": "bvsp_index=close/close_inicial",
                    "fallback_used": False,
                }
    except Exception:
        pass

    # fallback: SSOT local IBOV com symbol ^BVSP
    local_path = Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/precos_brutos/ibov/brapi/20260204/precos_brutos_ibov.csv")
    if not local_path.exists():
        raise RuntimeError("Nao foi possivel obter BVSP por BRAPI/Stooq e fallback local nao existe")
    df = pd.read_csv(local_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["symbol"] == "^BVSP"].copy()
    df["bvsp_close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[["date", "bvsp_close"]].dropna().sort_values("date")
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    if df.empty:
        raise RuntimeError("Fallback local BVSP existe, mas sem dados no intervalo")
    df["bvsp_index"] = df["bvsp_close"] / float(df["bvsp_close"].iloc[0])
    return df, {
        "provider": "CEP_NA_BOLSA local fallback",
        "symbol": "^BVSP",
        "path": str(local_path),
        "reason": "BRAPI sem token e Stooq sem dados",
        "transform": "bvsp_index=close/close_inicial",
        "fallback_used": True,
    }


def main() -> int:
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(f"/home/wilson/CEP_COMPRA/outputs/plots/task_007/{run_id}")
    data_dir = out_root / "data"
    out_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    task006_base = Path("/home/wilson/CEP_COMPRA/outputs/plots/task_006")
    task006_latest = latest_run_dir(task006_base)

    portfolio_parquet = Path("/home/wilson/CEP_COMPRA/outputs/reports/task_005/run_20260211_211715/data/portfolio_daily.parquet")
    dfp = pd.read_parquet(portfolio_parquet)

    # map columns dynamically
    required = {"date", "position_value_brl", "cash_brl", "equity_brl"}
    if not required.issubset(set(dfp.columns)):
        raise RuntimeError(f"Colunas esperadas ausentes. Encontradas={list(dfp.columns)}")

    daily = (
        dfp.groupby("date", as_index=False)
        .agg(
            sum_positions=("position_value_brl", "sum"),
            cash_brl=("cash_brl", "max"),
            equity_brl=("equity_brl", "max"),
        )
        .sort_values("date")
    )
    daily["carteira_total_calc"] = daily["sum_positions"] + daily["cash_brl"]
    daily["equity_minus_calc"] = daily["equity_brl"] - daily["carteira_total_calc"]
    max_abs_diff = float(daily["equity_minus_calc"].abs().max())
    equity_includes_cash = max_abs_diff < 1e-6
    daily["carteira_total_brl"] = daily["equity_brl"] if equity_includes_cash else daily["carteira_total_calc"]
    daily["date_prev"] = daily["date"].shift(1)
    daily["carteira_total_prev"] = daily["carteira_total_brl"].shift(1)
    daily["equity_prev"] = daily["equity_brl"].shift(1)
    daily["cash_prev"] = daily["cash_brl"].shift(1)
    daily["sum_positions_prev"] = daily["sum_positions"].shift(1)
    daily["daily_return"] = daily["carteira_total_brl"].pct_change()
    daily["abs_daily_return"] = daily["daily_return"].abs()

    # top jumps
    topk = daily.dropna(subset=["abs_daily_return"]).nlargest(10, "abs_daily_return").copy()
    topk["delta_total"] = topk["carteira_total_brl"] - topk["carteira_total_prev"]
    topk["delta_equity"] = topk["equity_brl"] - topk["equity_prev"]
    topk["delta_cash"] = topk["cash_brl"] - topk["cash_prev"]
    topk["delta_positions"] = topk["sum_positions"] - topk["sum_positions_prev"]
    topk = topk[
        [
            "date_prev",
            "date",
            "carteira_total_prev",
            "carteira_total_brl",
            "equity_prev",
            "equity_brl",
            "cash_prev",
            "cash_brl",
            "delta_total",
            "delta_equity",
            "delta_cash",
            "delta_positions",
            "daily_return",
            "abs_daily_return",
        ]
    ].sort_values("abs_daily_return", ascending=False)

    major = topk.iloc[0]
    jump_date = pd.Timestamp(major["date"])
    w0 = jump_date - pd.Timedelta(days=60)
    w1 = jump_date + pd.Timedelta(days=60)
    jump_window = daily[(daily["date"] >= w0) & (daily["date"] <= w1)].copy()

    # Save diagnostic parquets
    p_carteira = data_dir / "carteira_total_daily.parquet"
    p_topk = data_dir / "jumps_topk.parquet"
    p_win = data_dir / "jump_window_60d.parquet"
    daily.to_parquet(p_carteira, index=False)
    topk.to_parquet(p_topk, index=False)
    jump_window.to_parquet(p_win, index=False)

    # Benchmarks
    start_date = pd.Timestamp("2018-07-02")
    end_date = pd.Timestamp("2025-12-31")
    cdi, cdi_meta = fetch_cdi_sgs("02/07/2018", "31/12/2025")
    cdi = cdi[(cdi["date"] >= start_date) & (cdi["date"] <= end_date)].copy()
    spx, spx_meta = fetch_sp500_stooq()
    spx = spx[(spx["date"] >= start_date) & (spx["date"] <= end_date)].copy()
    bvsp, bvsp_meta = fetch_bvsp_with_fallback(start_date, end_date)

    p_cdi = data_dir / "cdi_index.parquet"
    p_spx = data_dir / "sp500_index.parquet"
    p_bvsp = data_dir / "bvsp_index.parquet"
    cdi.to_parquet(p_cdi, index=False)
    spx.to_parquet(p_spx, index=False)
    bvsp.to_parquet(p_bvsp, index=False)

    # Align and normalize all to base 1.0 on first carteira day
    ref = daily[["date", "carteira_total_brl"]].copy()
    aligned = ref.merge(cdi[["date", "cdi_index"]], on="date", how="left")
    aligned = aligned.merge(spx[["date", "sp500_index"]], on="date", how="left")
    aligned = aligned.merge(bvsp[["date", "bvsp_index"]], on="date", how="left")
    aligned = aligned.sort_values("date")
    aligned["cdi_index"] = aligned["cdi_index"].ffill().bfill()
    aligned["sp500_index"] = aligned["sp500_index"].ffill().bfill()
    aligned["bvsp_index"] = aligned["bvsp_index"].ffill().bfill()

    base_cart = float(aligned["carteira_total_brl"].iloc[0])
    base_cdi = float(aligned["cdi_index"].iloc[0])
    base_spx = float(aligned["sp500_index"].iloc[0])
    base_bvsp = float(aligned["bvsp_index"].iloc[0])
    aligned["carteira_index"] = aligned["carteira_total_brl"] / base_cart
    aligned["cdi_index_norm"] = aligned["cdi_index"] / base_cdi
    aligned["sp500_index_norm"] = aligned["sp500_index"] / base_spx
    aligned["bvsp_index_norm"] = aligned["bvsp_index"] / base_bvsp

    p_aligned = data_dir / "series_alinhadas.parquet"
    aligned.to_parquet(p_aligned, index=False)

    # Diagnostic markdown
    p_master_daily = Path("/home/wilson/CEP_COMPRA/outputs/reports/task_005/run_20260211_211715/data/master_state_daily.parquet")
    df_master_daily = pd.read_parquet(p_master_daily)
    st_jump = df_master_daily[df_master_daily["date"] == jump_date]["state"].iloc[0] if not df_master_daily[df_master_daily["date"] == jump_date].empty else "N/A"
    prev_date = pd.Timestamp(major["date_prev"]) if pd.notna(major["date_prev"]) else None
    st_prev = df_master_daily[df_master_daily["date"] == prev_date]["state"].iloc[0] if (prev_date is not None and not df_master_daily[df_master_daily["date"] == prev_date].empty) else "N/A"

    delta_total = float(major["delta_total"])
    delta_cash = float(major["delta_cash"])
    delta_pos = float(major["delta_positions"])
    if abs(delta_cash) > abs(delta_total) * 1.02 and "RISK_ON" in str(st_jump):
        probable_cause = (
            "inconsistencia contabil provavel no runner: caixa sobe acima da variacao liquida com Master em RISK_ON; "
            "indicativo de descontinuidade tecnica, nao mudanca de regime"
        )
    elif abs(delta_cash) > abs(delta_pos):
        probable_cause = "movimento de caixa dominante (realocacao/liquidacao parcial)"
    else:
        probable_cause = "variacao majoritariamente por posicoes/equity"
    diag_md = out_root / "diagnostico_salto.md"
    diag_lines = [
        "# Diagnostico de descontinuidade da Carteira",
        "",
        f"- Run task_006 baseline selecionado: `{task006_latest}`",
        f"- Maior salto por abs_daily_return: {pd.Timestamp(major['date']).date()}",
        f"- date_prev: {pd.Timestamp(major['date_prev']).date() if pd.notna(major['date_prev']) else 'N/A'}",
        f"- daily_return: {float(major['daily_return']):.6f}",
        f"- delta_total: {float(major['delta_total']):.2f}",
        f"- delta_equity: {float(major['delta_equity']):.2f}",
        f"- delta_cash: {float(major['delta_cash']):.2f}",
        f"- delta_positions: {float(major['delta_positions']):.2f}",
        f"- master_state_prev: {st_prev}",
        f"- master_state_jump: {st_jump}",
        f"- Causa provavel: {probable_cause}.",
        "",
        "## Evidencias",
        f"- `jumps_topk.parquet`: `{p_topk}`",
        f"- `jump_window_60d.parquet`: `{p_win}`",
        f"- `carteira_total_daily.parquet`: `{p_carteira}`",
    ]
    diag_md.write_text("\n".join(diag_lines), encoding="utf-8")

    # Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aligned["date"], y=aligned["carteira_index"], mode="lines", name="Carteira"))
    fig.add_trace(go.Scatter(x=aligned["date"], y=aligned["cdi_index_norm"], mode="lines", name="CDI"))
    fig.add_trace(go.Scatter(x=aligned["date"], y=aligned["sp500_index_norm"], mode="lines", name="S&P 500"))
    fig.add_trace(go.Scatter(x=aligned["date"], y=aligned["bvsp_index_norm"], mode="lines", name="^BVSP"))
    fig.add_vline(x=jump_date, line_dash="dash", line_color="red")
    fig.add_annotation(
        x=jump_date,
        y=max(float(aligned["carteira_index"].max()), float(aligned["sp500_index_norm"].max())),
        text=(
            f"Maior salto\\n{jump_date.date()}\\n"
            f"ret={float(major['daily_return']):.2%}\\n"
            f"dTot={float(major['delta_total']):.0f} | dCash={float(major['delta_cash']):.0f}"
        ),
        showarrow=True,
        arrowhead=2,
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.update_layout(
        title="Carteira (equity+cash) vs CDI vs S&P 500 vs ^BVSP (base 1.0)",
        xaxis_title="Data",
        yaxis_title="Indice normalizado",
        template="plotly_white",
        hovermode="x unified",
    )
    html_path = out_root / "carteira_vs_cdi_vs_sp500_vs_bvsp.html"
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)

    manifest = {
        "task_id": "TASK_CEP_COMPRA_007_DIAGNOSTICO_SALTO_E_PLOTLY_COM_BVSP",
        "run_id": run_id,
        "timestamp_utc": now.isoformat(),
        "task006_baseline_run": str(task006_latest),
        "inputs": {
            "portfolio_daily_parquet": "/home/wilson/CEP_COMPRA/outputs/reports/task_005/run_20260211_211715/data/portfolio_daily.parquet",
        },
        "column_mapping": {
            "date": "date",
            "equity": "equity_brl",
            "cash": "cash_brl",
            "sum_positions": "position_value_brl(groupby sum)",
            "equity_includes_cash": equity_includes_cash,
            "equity_vs_calc_max_abs_diff": max_abs_diff,
        },
        "largest_jump": {
            "date_prev": str(pd.Timestamp(major["date_prev"]).date()) if pd.notna(major["date_prev"]) else None,
            "date_jump": str(pd.Timestamp(major["date"]).date()),
            "daily_return": float(major["daily_return"]),
            "delta_total": float(major["delta_total"]),
            "delta_equity": float(major["delta_equity"]),
            "delta_cash": float(major["delta_cash"]),
            "delta_positions": float(major["delta_positions"]),
            "master_state_prev": st_prev,
            "master_state_jump": st_jump,
            "probable_cause": probable_cause,
        },
        "sources": {
            "cdi": cdi_meta,
            "sp500": spx_meta,
            "bvsp": bvsp_meta,
        },
        "alignment_rule": "calendario da carteira; forward-fill benchmarks; bfill apenas no inicio se necessario",
        "outputs": {
            "plot_html": str(html_path),
            "series_alinhadas_parquet": str(p_aligned),
            "diagnostico_md": str(diag_md),
            "carteira_total_daily_parquet": str(p_carteira),
            "jumps_topk_parquet": str(p_topk),
            "jump_window_60d_parquet": str(p_win),
            "cdi_index_parquet": str(p_cdi),
            "sp500_index_parquet": str(p_spx),
            "bvsp_index_parquet": str(p_bvsp),
        },
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    hash_files = [p_carteira, p_topk, p_win, p_cdi, p_spx, p_bvsp, p_aligned, diag_md, html_path, manifest_path]
    hashes = [{"path": str(p), "sha256": sha256_file(p)} for p in hash_files]
    (out_root / "hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=True), encoding="utf-8")

    print(str(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
