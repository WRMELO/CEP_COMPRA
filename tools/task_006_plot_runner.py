#!/usr/bin/env python3
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_cdi_sgs(start_br: str, end_br: str) -> pd.DataFrame:
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
    # taxa diaria em percentual; indice acumulado base 1
    df["cdi_factor"] = 1.0 + (df["cdi_rate_pct"] / 100.0)
    df["cdi_index"] = df["cdi_factor"].cumprod()
    return df


def fetch_sp500_stooq() -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=%5Espx&i=d"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    # CSV simples: Date,Open,High,Low,Close,Volume
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        rows.append((parts[0], parts[4]))
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sp500_close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[["date", "sp500_close"]].dropna().sort_values("date")
    return df


def main() -> int:
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    root = Path(f"/home/wilson/CEP_COMPRA/outputs/plots/task_006/{run_id}")
    data_dir = root / "data"
    root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    portfolio_parquet = Path("/home/wilson/CEP_COMPRA/outputs/reports/task_005/run_20260211_211715/data/portfolio_daily.parquet")
    start_date = pd.Timestamp("2018-07-02")
    end_date = pd.Timestamp("2025-12-31")

    # S1: carregar carteira e definir serie carteira_total
    df_port = pd.read_parquet(portfolio_parquet)
    daily = (
        df_port.groupby("date", as_index=False)
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
    daily = daily[(daily["date"] >= start_date) & (daily["date"] <= end_date)].copy()
    daily_base = daily[["date", "carteira_total_brl", "cash_brl", "equity_brl", "sum_positions", "equity_minus_calc"]].copy()
    daily_base.to_parquet(data_dir / "carteira_total_base.parquet", index=False)

    # S2: buscar CDI e S&P500
    cdi = fetch_cdi_sgs("02/07/2018", "31/12/2025")
    cdi = cdi[(cdi["date"] >= start_date) & (cdi["date"] <= end_date)].copy()
    cdi.to_parquet(data_dir / "cdi_raw_index.parquet", index=False)

    spx = fetch_sp500_stooq()
    spx = spx[(spx["date"] >= start_date) & (spx["date"] <= end_date)].copy()
    spx["sp500_index"] = spx["sp500_close"] / float(spx["sp500_close"].iloc[0])
    spx.to_parquet(data_dir / "sp500_raw_index.parquet", index=False)

    # S3: alinhamento e normalizacao base 1.0 no primeiro dia da carteira
    aligned = daily_base[["date", "carteira_total_brl"]].merge(
        cdi[["date", "cdi_index"]], on="date", how="left"
    ).merge(
        spx[["date", "sp500_index"]], on="date", how="left"
    ).sort_values("date")
    # calendario referencia = carteira; benchmarks com ffill
    aligned["cdi_index"] = aligned["cdi_index"].ffill()
    aligned["sp500_index"] = aligned["sp500_index"].ffill()
    if aligned["cdi_index"].isna().any():
        aligned["cdi_index"] = aligned["cdi_index"].bfill()
    if aligned["sp500_index"].isna().any():
        aligned["sp500_index"] = aligned["sp500_index"].bfill()

    base_cart = float(aligned["carteira_total_brl"].iloc[0])
    base_cdi = float(aligned["cdi_index"].iloc[0])
    base_spx = float(aligned["sp500_index"].iloc[0])
    aligned["carteira_index"] = aligned["carteira_total_brl"] / base_cart
    aligned["cdi_index_norm"] = aligned["cdi_index"] / base_cdi
    aligned["sp500_index_norm"] = aligned["sp500_index"] / base_spx
    aligned.to_parquet(data_dir / "carteira_cdi_sp500_alinhado.parquet", index=False)

    # S4: plotly html
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aligned["date"], y=aligned["carteira_index"], mode="lines", name="Carteira"))
    fig.add_trace(go.Scatter(x=aligned["date"], y=aligned["cdi_index_norm"], mode="lines", name="CDI"))
    fig.add_trace(go.Scatter(x=aligned["date"], y=aligned["sp500_index_norm"], mode="lines", name="S&P 500"))
    fig.update_layout(
        title="Carteira (equity+cash) vs CDI vs S&P 500 (base 1.0)",
        xaxis_title="Data",
        yaxis_title="Indice normalizado",
        template="plotly_white",
        hovermode="x unified",
    )
    html_path = root / "carteira_vs_cdi_vs_sp500.html"
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)

    manifest = {
        "task_id": "TASK_CEP_COMPRA_006_PLOTLY_EQUITY_CASH_VS_CDI_VS_SP500",
        "run_id": run_id,
        "timestamp_utc": now.isoformat(),
        "sources": {
            "cdi": {
                "provider": "Banco Central SGS",
                "series_code": 12,
                "url": "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial=02/07/2018&dataFinal=31/12/2025",
            },
            "sp500": {
                "provider": "Stooq",
                "symbol": "^spx",
                "url": "https://stooq.com/q/d/l/?s=%5Espx&i=d",
            },
        },
        "column_mapping": {
            "portfolio_date": "date",
            "portfolio_equity": "equity_brl",
            "portfolio_cash": "cash_brl",
            "portfolio_positions_sum": "sum_positions",
            "portfolio_total_used": "carteira_total_brl",
            "equity_includes_cash": equity_includes_cash,
            "equity_vs_calc_max_abs_diff": max_abs_diff,
        },
        "alignment_rule": "calendario da carteira como referencia; benchmarks com forward-fill (e backward-fill apenas se necessario no inicio)",
        "outputs": {
            "plot_html": str(html_path),
            "aligned_parquet": str(data_dir / "carteira_cdi_sp500_alinhado.parquet"),
            "carteira_total_parquet": str(data_dir / "carteira_total_base.parquet"),
            "cdi_parquet": str(data_dir / "cdi_raw_index.parquet"),
            "sp500_parquet": str(data_dir / "sp500_raw_index.parquet"),
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    files_for_hash = [
        data_dir / "carteira_total_base.parquet",
        data_dir / "cdi_raw_index.parquet",
        data_dir / "sp500_raw_index.parquet",
        data_dir / "carteira_cdi_sp500_alinhado.parquet",
        html_path,
        manifest_path,
    ]
    hashes = [{"path": str(p), "sha256": sha256_file(p)} for p in files_for_hash]
    (root / "hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=True), encoding="utf-8")

    print(str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
