#!/usr/bin/env python3
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def next_trading_day_map(dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, pd.Timestamp]:
    out: Dict[pd.Timestamp, pd.Timestamp] = {}
    for i, d in enumerate(dates):
        out[d] = dates[i + 1] if i + 1 < len(dates) else pd.NaT
    return out


def main() -> int:
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(f"/home/wilson/CEP_COMPRA/outputs/forensics/task_009/{run_id}")
    data_dir = out_root / "data"
    out_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Inputs fixos da task
    p_manifest = Path("/home/wilson/CEP_COMPRA/docs/bundle_cep_na_bolsa/manifest.json")
    p_interface = Path("/home/wilson/CEP_COMPRA/docs/bundle_cep_na_bolsa/interface_contrato.md")
    p_master = Path("/home/wilson/CEP_COMPRA/outputs/reports/task_005/run_20260211_211715/data/master_state_daily.parquet")
    p_port = Path("/home/wilson/CEP_COMPRA/outputs/reports/task_005/run_20260211_211715/data/portfolio_daily.parquet")
    p_weekly = Path("/home/wilson/CEP_COMPRA/outputs/reports/task_005/run_20260211_211715/data/portfolio_weekly_events.parquet")
    p_audit = Path("/home/wilson/CEP_COMPRA/outputs/forensics/task_008/run_20260212_103613/data/portfolio_audit_daily.parquet")

    p_limits = Path(
        "/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/merged/limits_per_ticker.csv"
    )
    p_xt = Path(
        "/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/base_operacional/base_operacional_xt.csv"
    )
    p_rule_code = Path("/home/wilson/CEP_COMPRA/tools/task_005_report_runner.py")
    p_bundle_rule_code = Path("/home/wilson/CEP_NA_BOLSA/src/cep/runners/runner_backtest_fixed_limits_exp031_v1.py")
    p_constit = Path("/home/wilson/CEP_NA_BOLSA/docs/00_constituicao/CEP_NA_BOLSA_CONSTITUICAO_V2_20260204.md")

    if not p_manifest.exists():
        p_manifest = Path("/home/wilson/.cursor/worktrees/CEP_COMPRA/gxt/docs/bundle_cep_na_bolsa/manifest.json")
    if not p_interface.exists():
        p_interface = Path("/home/wilson/.cursor/worktrees/CEP_COMPRA/gxt/docs/bundle_cep_na_bolsa/interface_contrato.md")

    required = [
        p_manifest,
        p_interface,
        p_master,
        p_port,
        p_weekly,
        p_audit,
        p_limits,
        p_xt,
        p_rule_code,
        p_bundle_rule_code,
        p_constit,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Arquivos obrigatorios ausentes: {missing}")

    analysis_start = pd.Timestamp("2022-07-01")
    analysis_end = pd.Timestamp("2023-06-30")
    focus_start = pd.Timestamp("2022-10-26")
    focus_end = pd.Timestamp("2023-03-31")

    # Carga de dados
    df_port = pd.read_parquet(p_port).copy()
    df_port["date"] = pd.to_datetime(df_port["date"], errors="coerce")
    df_weekly = pd.read_parquet(p_weekly).copy()
    df_weekly["event_date"] = pd.to_datetime(df_weekly["event_date"], errors="coerce")
    df_master = pd.read_parquet(p_master).copy()
    df_master["date"] = pd.to_datetime(df_master["date"], errors="coerce")
    df_audit = pd.read_parquet(p_audit).copy()
    df_audit["date"] = pd.to_datetime(df_audit["date"], errors="coerce")

    df_limits = pd.read_csv(p_limits)
    df_xt = pd.read_csv(p_xt)
    df_xt["date"] = pd.to_datetime(df_xt["date"], errors="coerce")

    # recorte de periodo amplo
    in_wide = (df_xt["date"] >= analysis_start) & (df_xt["date"] <= analysis_end)
    df_xt_wide = df_xt[in_wide].copy()
    df_audit_wide = df_audit[(df_audit["date"] >= analysis_start) & (df_audit["date"] <= analysis_end)].copy()

    # Eventos de rompimento I/MR por ticker
    lim_cols = ["ticker", "i_lcl", "i_ucl", "mr_ucl"]
    lim = df_limits[lim_cols].copy()
    ev = df_xt_wide.merge(lim, on="ticker", how="inner")
    ev = ev.sort_values(["ticker", "date"]).copy()
    ev["mr_value"] = (ev["xt"] - ev.groupby("ticker")["xt"].shift(1)).abs()

    ev_i_lcl = ev[ev["xt"] < ev["i_lcl"]].copy()
    ev_i_lcl["tipo_grafico"] = "I"
    ev_i_lcl["lado"] = "LCL"
    ev_i_lcl["valor"] = ev_i_lcl["xt"]
    ev_i_lcl["limite"] = ev_i_lcl["i_lcl"]
    ev_i_lcl["flag"] = True

    ev_i_ucl = ev[ev["xt"] > ev["i_ucl"]].copy()
    ev_i_ucl["tipo_grafico"] = "I"
    ev_i_ucl["lado"] = "UCL"
    ev_i_ucl["valor"] = ev_i_ucl["xt"]
    ev_i_ucl["limite"] = ev_i_ucl["i_ucl"]
    ev_i_ucl["flag"] = True

    ev_mr_ucl = ev[ev["mr_value"] > ev["mr_ucl"]].copy()
    ev_mr_ucl["tipo_grafico"] = "MR"
    ev_mr_ucl["lado"] = "UCL"
    ev_mr_ucl["valor"] = ev_mr_ucl["mr_value"]
    ev_mr_ucl["limite"] = ev_mr_ucl["mr_ucl"]
    ev_mr_ucl["flag"] = True

    eventos = pd.concat([ev_i_lcl, ev_i_ucl, ev_mr_ucl], ignore_index=True)
    eventos = eventos[
        ["date", "ticker", "tipo_grafico", "lado", "valor", "limite", "flag", "xt", "mr_value", "close", "asset_class"]
    ].sort_values(["date", "ticker", "tipo_grafico", "lado"])
    p_eventos = data_dir / "eventos_rompimento.parquet"
    eventos.to_parquet(p_eventos, index=False)

    # Reconstrucao de SELL no runner da task_005: ticker some de um dia para o outro
    pos_day = (
        df_port[df_port["ticker"].notna()][["date", "ticker", "position_value_brl"]]
        .groupby(["date", "ticker"], as_index=False)
        .sum()
        .sort_values(["ticker", "date"])
    )
    pos_day["prev_value"] = pos_day.groupby("ticker")["position_value_brl"].shift(1)
    pos_day["prev_date"] = pos_day.groupby("ticker")["date"].shift(1)

    merged_next = pos_day[["date", "ticker"]].copy()
    merged_next["next_date"] = merged_next.groupby("ticker")["date"].shift(-1)
    merged_next["next_exists"] = merged_next["next_date"].notna()

    # ticker vendido em d se existia em d-1 e não existe em d
    trading_dates = sorted(df_audit_wide["date"].dropna().unique().tolist())
    next_map = next_trading_day_map(trading_dates)
    sells_rows = []
    for d in trading_dates[1:]:
        prev_d = trading_dates[trading_dates.index(d) - 1]
        prev_set = set(pos_day[pos_day["date"] == prev_d]["ticker"])
        curr_set = set(pos_day[pos_day["date"] == d]["ticker"])
        sold = sorted(prev_set - curr_set)
        prev_vals = pos_day[pos_day["date"] == prev_d].set_index("ticker")["position_value_brl"].to_dict()
        for t in sold:
            sells_rows.append(
                {"sell_date": d, "ticker": t, "sell_prev_position_value_brl": float(prev_vals.get(t, 0.0))}
            )
    df_sells = pd.DataFrame(sells_rows)
    if df_sells.empty:
        df_sells = pd.DataFrame(columns=["sell_date", "ticker", "sell_prev_position_value_brl"])

    # enrich eventos com sell same/next
    ev2 = eventos.copy()
    ev2["next_trading_date"] = ev2["date"].map(next_map)
    sells_same = df_sells.rename(columns={"sell_date": "date"})[["date", "ticker"]].copy()
    sells_same["sell_same_day"] = True
    sells_next = df_sells.rename(columns={"sell_date": "next_trading_date"})[["next_trading_date", "ticker"]].copy()
    sells_next["sell_next_day"] = True
    ev2 = ev2.merge(sells_same, on=["date", "ticker"], how="left")
    ev2 = ev2.merge(sells_next, on=["next_trading_date", "ticker"], how="left")
    ev2["sell_same_day"] = ev2["sell_same_day"].notna()
    ev2["sell_next_day"] = ev2["sell_next_day"].notna()
    ev2["sell_same_or_next"] = ev2["sell_same_day"] | ev2["sell_next_day"]

    held_daily = pos_day[["date", "ticker"]].copy()
    held_daily["held_on_date"] = True
    ev2 = ev2.merge(held_daily, on=["date", "ticker"], how="left")
    ev2["held_on_date"] = ev2["held_on_date"].notna()

    # métricas de impacto pós-evento
    a = df_audit_wide[["date", "cash_ratio", "n_positions", "total_audit_brl"]].copy().sort_values("date")
    a["next_date"] = a["date"].shift(-1)
    a["cash_ratio_next"] = a["cash_ratio"].shift(-1)
    a["n_positions_next"] = a["n_positions"].shift(-1)
    a["delta_cash_ratio_1d"] = a["cash_ratio_next"] - a["cash_ratio"]
    a["delta_n_positions_1d"] = a["n_positions_next"] - a["n_positions"]
    ev2 = ev2.merge(a[["date", "delta_cash_ratio_1d", "delta_n_positions_1d"]], on="date", how="left")

    p_eventos_sell = data_dir / "eventos_com_sell.parquet"
    ev2.to_parquet(p_eventos_sell, index=False)

    # Cobertura dos eventos para tickers efetivamente em carteira
    held_obs = (
        df_port[
            (df_port["date"] >= analysis_start)
            & (df_port["date"] <= analysis_end)
            & (df_port["ticker"].notna())
        ][["date", "ticker"]]
        .drop_duplicates()
        .copy()
    )
    held_cov = held_obs.merge(
        df_xt_wide[["date", "ticker", "xt"]], on=["date", "ticker"], how="left"
    ).merge(lim, on="ticker", how="left")
    held_cov_full = held_cov.dropna(subset=["xt", "i_lcl", "i_ucl", "mr_ucl"]).copy()
    held_cov_full = held_cov_full.sort_values(["ticker", "date"])
    held_cov_full["mr_value"] = (
        held_cov_full["xt"] - held_cov_full.groupby("ticker")["xt"].shift(1)
    ).abs()
    held_coverage = pd.DataFrame(
        [
            {
                "analysis_start": analysis_start,
                "analysis_end": analysis_end,
                "held_rows_total": int(len(held_cov)),
                "held_rows_xt_missing": int(held_cov["xt"].isna().sum()),
                "held_rows_with_full_data": int(len(held_cov_full)),
                "i_ucl_viol_in_held": int((held_cov_full["xt"] > held_cov_full["i_ucl"]).sum()),
                "i_lcl_viol_in_held": int((held_cov_full["xt"] < held_cov_full["i_lcl"]).sum()),
                "mr_ucl_viol_in_held": int((held_cov_full["mr_value"] > held_cov_full["mr_ucl"]).sum()),
            }
        ]
    )
    p_held_cov = data_dir / "held_coverage.parquet"
    held_coverage.to_parquet(p_held_cov, index=False)

    # turnover diário aproximado (sells + buys weekly allocations)
    sell_daily = df_sells.groupby("sell_date", as_index=False)["sell_prev_position_value_brl"].sum().rename(
        columns={"sell_date": "date", "sell_prev_position_value_brl": "sell_value_brl"}
    )
    buy_daily = (
        df_weekly.groupby("event_date", as_index=False)["allocation_brl"].sum().rename(
            columns={"event_date": "date", "allocation_brl": "buy_value_brl"}
        )
    )
    turn = df_audit_wide[["date", "total_audit_brl", "cash_ratio", "n_positions"]].copy()
    turn = turn.merge(sell_daily, on="date", how="left").merge(buy_daily, on="date", how="left")
    turn["sell_value_brl"] = turn["sell_value_brl"].fillna(0.0)
    turn["buy_value_brl"] = turn["buy_value_brl"].fillna(0.0)
    turn["turnover_ratio"] = (turn["sell_value_brl"] + turn["buy_value_brl"]) / turn["total_audit_brl"]

    def summarize_interval(start: pd.Timestamp, end: pd.Timestamp, label: str) -> List[Dict]:
        e_all = ev2[(ev2["date"] >= start) & (ev2["date"] <= end)].copy()
        t = turn[(turn["date"] >= start) & (turn["date"] <= end)].copy()
        rows = []
        for population_name, e in [("universo_total", e_all), ("somente_em_carteira", e_all[e_all["held_on_date"]])]:
            for tg, ld in [("I", "UCL"), ("I", "LCL"), ("MR", "UCL")]:
                sub = e[(e["tipo_grafico"] == tg) & (e["lado"] == ld)]
                n = int(len(sub))
                same = int(sub["sell_same_day"].sum()) if n else 0
                nextd = int(sub["sell_next_day"].sum()) if n else 0
                son = int(sub["sell_same_or_next"].sum()) if n else 0
                rows.append(
                    {
                        "intervalo": label,
                        "population": population_name,
                        "start": start,
                        "end": end,
                        "tipo_grafico": tg,
                        "lado": ld,
                        "event_count": n,
                        "sell_same_day_count": same,
                        "sell_next_day_count": nextd,
                        "sell_same_or_next_count": son,
                        "sell_same_or_next_rate": (son / n) if n else 0.0,
                        "avg_delta_cash_ratio_1d": float(sub["delta_cash_ratio_1d"].mean()) if n else 0.0,
                        "avg_delta_n_positions_1d": float(sub["delta_n_positions_1d"].mean()) if n else 0.0,
                    }
                )
        rows.append(
            {
                "intervalo": label,
                "population": "portfolio_diario",
                "start": start,
                "end": end,
                "tipo_grafico": "PORTFOLIO",
                "lado": "N/A",
                "event_count": int(len(e_all)),
                "sell_same_day_count": int(e_all["sell_same_day"].sum()) if len(e_all) else 0,
                "sell_next_day_count": int(e_all["sell_next_day"].sum()) if len(e_all) else 0,
                "sell_same_or_next_count": int(e_all["sell_same_or_next"].sum()) if len(e_all) else 0,
                "sell_same_or_next_rate": float(e_all["sell_same_or_next"].mean()) if len(e_all) else 0.0,
                "avg_delta_cash_ratio_1d": float(e_all["delta_cash_ratio_1d"].mean()) if len(e_all) else 0.0,
                "avg_delta_n_positions_1d": float(e_all["delta_n_positions_1d"].mean()) if len(e_all) else 0.0,
                "avg_turnover_ratio": float(t["turnover_ratio"].mean()) if len(t) else 0.0,
                "avg_cash_ratio": float(t["cash_ratio"].mean()) if len(t) else 0.0,
                "avg_n_positions": float(t["n_positions"].mean()) if len(t) else 0.0,
            }
        )
        return rows

    metric_rows = []
    metric_rows += summarize_interval(analysis_start, analysis_end, "amplo_2022-07_2023-06")
    metric_rows += summarize_interval(focus_start, focus_end, "foco_plateau_2022-10-26_2023-03-31")
    metricas = pd.DataFrame(metric_rows)
    p_metricas = data_dir / "metricas_impacto.parquet"
    metricas.to_parquet(p_metricas, index=False)

    # S1: relatório de regras de SELL
    regras_md = out_root / "regras_sell_xbar_r_i_mr.md"
    regras_md.write_text(
        "\n".join(
            [
                "# Regras de SELL ligadas a rompimentos (Xbar/R/I/MR)",
                "",
                "## Localização exata no código",
                "- `CEP_COMPRA/tools/task_005_report_runner.py` (bloco de venda diária congelada):",
                "  - `if float(r) < float(lim[\"i_lcl\"]) or float(r) > float(lim[\"i_ucl\"]) or mr > float(lim[\"mr_ucl\"]): cash += pos.pop(t)`",
                "- `CEP_NA_BOLSA/src/cep/runners/runner_backtest_fixed_limits_exp031_v1.py` (estado Master):",
                "  - `stress_amp_raw = r > limits_xbar_r[\"ucl_r\"]`",
                "  - `xbar_up = xbar > limits_xbar_r[\"ucl_xbar\"]`",
                "  - `stress_i_down = xt < limits_imr[\"lcl_i\"]`",
                "  - `stress_i_up = xt > limits_imr[\"ucl_i\"]`",
                "  - `upside_extreme = xbar_up | stress_i_up`",
                "  - `stress_amp = stress_amp_raw & (~upside_extreme)`",
                "",
                "## Definição de X_t (sem inferência)",
                "- `CEP_NA_BOLSA/docs/00_constituicao/CEP_NA_BOLSA_CONSTITUICAO_V2_20260204.md` define:",
                "  - `X_t = log(Close_t / Close_{t-1})`",
                "- `schema_base_operacional.json` registra as colunas da base operacional: `ticker,date,close,xt,asset_class`.",
                "",
                "## Resposta objetiva (SELL por rompimento positivo/UCL)",
                "- Xbar (UCL): **não há SELL direto no runner da carteira**; no Master, `xbar_up` atua como exceção de upside para não acionar preservação por amplitude.",
                "- R (UCL): no Master, `r > ucl_r` é gatilho de estresse de amplitude (`stress_amp_raw`), com exceção de upside.",
                "- I (UCL): no runner da carteira há SELL direto por `xt > i_ucl`.",
                "- MR (UCL): no runner da carteira há SELL direto por `mr > mr_ucl`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # S3: relatório conclusivo
    def get_count(intervalo: str, population: str, tg: str, ld: str, col: str) -> float:
        m = metricas[
            (metricas["intervalo"] == intervalo)
            & (metricas["population"] == population)
            & (metricas["tipo_grafico"] == tg)
            & (metricas["lado"] == ld)
        ]
        if m.empty:
            return 0.0
        return float(m.iloc[0][col])

    rel = out_root / "relatorio_task_009_plato_e_rompimentos.md"
    rel.write_text(
        "\n".join(
            [
                "# Task 009 - Verificação SELL por rompimento e impacto no plateau",
                "",
                "## Resposta direta",
                "- SELL por UCL em Xbar: **não** (no fluxo da carteira).",
                "- SELL por UCL em R: **indireto via estado Master** (estresse de amplitude), com exceção de upside.",
                "- SELL por UCL em I: **sim** (regra explícita no runner da carteira).",
                "- SELL por UCL em MR: **sim** (regra explícita no runner da carteira).",
                "",
                "## Contagens no intervalo amplo (2022-07-01..2023-06-30)",
                f"- I/UCL eventos (universo): `{int(get_count('amplo_2022-07_2023-06', 'universo_total', 'I', 'UCL', 'event_count'))}`",
                f"- I/UCL eventos (em carteira): `{int(get_count('amplo_2022-07_2023-06', 'somente_em_carteira', 'I', 'UCL', 'event_count'))}`",
                f"- MR/UCL eventos (universo): `{int(get_count('amplo_2022-07_2023-06', 'universo_total', 'MR', 'UCL', 'event_count'))}`",
                f"- MR/UCL eventos (em carteira): `{int(get_count('amplo_2022-07_2023-06', 'somente_em_carteira', 'MR', 'UCL', 'event_count'))}`",
                f"- Taxa SELL D0/D+1 (I/UCL, em carteira): `{get_count('amplo_2022-07_2023-06', 'somente_em_carteira', 'I', 'UCL', 'sell_same_or_next_rate'):.4f}`",
                f"- Taxa SELL D0/D+1 (MR/UCL, em carteira): `{get_count('amplo_2022-07_2023-06', 'somente_em_carteira', 'MR', 'UCL', 'sell_same_or_next_rate'):.4f}`",
                "",
                "## Contagens no intervalo foco plateau (2022-10-26..2023-03-31)",
                f"- I/UCL eventos (universo): `{int(get_count('foco_plateau_2022-10-26_2023-03-31', 'universo_total', 'I', 'UCL', 'event_count'))}`",
                f"- I/UCL eventos (em carteira): `{int(get_count('foco_plateau_2022-10-26_2023-03-31', 'somente_em_carteira', 'I', 'UCL', 'event_count'))}`",
                f"- MR/UCL eventos (universo): `{int(get_count('foco_plateau_2022-10-26_2023-03-31', 'universo_total', 'MR', 'UCL', 'event_count'))}`",
                f"- MR/UCL eventos (em carteira): `{int(get_count('foco_plateau_2022-10-26_2023-03-31', 'somente_em_carteira', 'MR', 'UCL', 'event_count'))}`",
                f"- Taxa SELL D0/D+1 (I/UCL, em carteira): `{get_count('foco_plateau_2022-10-26_2023-03-31', 'somente_em_carteira', 'I', 'UCL', 'sell_same_or_next_rate'):.4f}`",
                f"- Taxa SELL D0/D+1 (MR/UCL, em carteira): `{get_count('foco_plateau_2022-10-26_2023-03-31', 'somente_em_carteira', 'MR', 'UCL', 'sell_same_or_next_rate'):.4f}`",
                "",
                "## Conclusão sobre o plateau (evidencial)",
                "- O dataset de eventos mostra presença de rompimentos I/MR e associação com saídas de posição.",
                "- A explicação do plateau deve considerar a combinação de: regime Master (RISK_ON/OFF/PRESERVACAO), vendas por I/MR e performance dos ativos mantidos.",
                "- Não há evidência de SELL direto por `Xbar > UCL` no fluxo da carteira.",
                "",
                "## Cobertura de dados para tickers em carteira (intervalo amplo)",
                f"- held_rows_total: `{int(held_coverage.iloc[0]['held_rows_total'])}`",
                f"- held_rows_xt_missing: `{int(held_coverage.iloc[0]['held_rows_xt_missing'])}`",
                f"- held_rows_with_full_data: `{int(held_coverage.iloc[0]['held_rows_with_full_data'])}`",
                f"- i_ucl_viol_in_held: `{int(held_coverage.iloc[0]['i_ucl_viol_in_held'])}`",
                f"- i_lcl_viol_in_held: `{int(held_coverage.iloc[0]['i_lcl_viol_in_held'])}`",
                f"- mr_ucl_viol_in_held: `{int(held_coverage.iloc[0]['mr_ucl_viol_in_held'])}`",
                "",
                "## Artefatos",
                f"- `eventos_rompimento.parquet`: `{p_eventos}`",
                f"- `eventos_com_sell.parquet`: `{p_eventos_sell}`",
                f"- `metricas_impacto.parquet`: `{p_metricas}`",
                f"- `held_coverage.parquet`: `{p_held_cov}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "task_id": "TASK_CEP_COMPRA_009_VERIFICAR_SELL_POR_ROMPIMENTO_POSITIVO_E_IMPACTO_PLATO",
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "inputs": {
            "bundle_manifest": str(p_manifest),
            "bundle_interface": str(p_interface),
            "master_state_daily_parquet": str(p_master),
            "portfolio_daily_parquet": str(p_port),
            "portfolio_weekly_events_parquet": str(p_weekly),
            "audit_daily_parquet": str(p_audit),
            "limits_per_ticker_csv": str(p_limits),
            "base_operacional_xt_csv": str(p_xt),
        },
        "analysis_periods": {
            "analysis_start": str(analysis_start.date()),
            "analysis_end": str(analysis_end.date()),
            "focus_plateau_start": str(focus_start.date()),
            "focus_plateau_end": str(focus_end.date()),
        },
        "outputs": [
            "regras_sell_xbar_r_i_mr.md",
            "data/eventos_rompimento.parquet",
            "data/eventos_com_sell.parquet",
            "data/metricas_impacto.parquet",
            "data/held_coverage.parquet",
            "relatorio_task_009_plato_e_rompimentos.md",
        ],
    }
    p_manifest_out = out_root / "manifest.json"
    p_manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    file_list = [p for p in out_root.rglob("*") if p.is_file()]
    hashes = {str(p.relative_to(out_root)): sha256_file(p) for p in sorted(file_list)}
    p_hashes = out_root / "hashes.json"
    p_hashes.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TASK009] run_id={run_id}")
    print(f"[TASK009] output_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
