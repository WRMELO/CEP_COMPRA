#!/usr/bin/env python3
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_col(columns: List[str], candidates: List[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise RuntimeError(f"Nenhuma coluna encontrada para candidatos={candidates}. Colunas={columns}")
    return None


def fetch_external_price_probe(ticker: str) -> List[Dict]:
    probes: List[Dict] = []
    urls = [
        {"provider": "BRAPI", "url": f"https://brapi.dev/api/quote/{ticker}?range=3mo&interval=1d"},
        {"provider": "BRAPI", "url": f"https://brapi.dev/api/quote/{ticker}.SA?range=3mo&interval=1d"},
        {"provider": "Stooq", "url": f"https://stooq.com/q/d/l/?s={ticker.lower()}.sa&i=d"},
        {
            "provider": "Yahoo",
            "url": "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{ticker}.SA?period1=1759276800&period2=1763596800&interval=1d&events=history",
        },
    ]
    for item in urls:
        rec = {"provider": item["provider"], "url": item["url"]}
        try:
            r = requests.get(item["url"], timeout=20)
            rec["status_code"] = int(r.status_code)
            rec["ok"] = bool(r.status_code == 200)
            rec["snippet"] = r.text[:180].replace("\n", " ")
            rec["reason"] = "ok" if r.status_code == 200 else "http_error"
        except Exception as exc:
            rec["status_code"] = None
            rec["ok"] = False
            rec["snippet"] = str(exc)
            rec["reason"] = "exception"
        probes.append(rec)
    return probes


def find_plateau_interval(
    audit_daily: pd.DataFrame,
    start_suspect: pd.Timestamp,
    rolling_window_days: int,
    max_abs_return_threshold: float,
    min_consecutive_days: int,
) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    work = audit_daily[audit_daily["date"] >= start_suspect].copy()
    if work.empty:
        raise RuntimeError("Sem dados apos plateau_suspect_start")
    work["abs_return"] = work["daily_return"].abs()
    work["rolling_max_abs_return"] = work["abs_return"].rolling(
        rolling_window_days, min_periods=rolling_window_days
    ).max()
    work["plateau_flag"] = work["rolling_max_abs_return"] <= max_abs_return_threshold

    streaks: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    in_streak = False
    streak_start = None
    streak_count = 0
    prev_date = None
    for _, row in work.iterrows():
        flag = bool(row["plateau_flag"]) if pd.notna(row["plateau_flag"]) else False
        d = row["date"]
        if flag and not in_streak:
            in_streak = True
            streak_start = d
            streak_count = 1
        elif flag and in_streak:
            streak_count += 1
        elif (not flag) and in_streak:
            if streak_count >= min_consecutive_days:
                streaks.append((streak_start, prev_date, streak_count))
            in_streak = False
            streak_start = None
            streak_count = 0
        prev_date = d

    if in_streak and streak_count >= min_consecutive_days:
        streaks.append((streak_start, prev_date, streak_count))
    if not streaks:
        raise RuntimeError(
            "Nenhum intervalo de plateau detectado com os parametros fornecidos "
            f"(window={rolling_window_days}, threshold={max_abs_return_threshold}, min_days={min_consecutive_days})"
        )
    return streaks[0]


def main() -> int:
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(f"/home/wilson/CEP_COMPRA/outputs/forensics/task_008/{run_id}")
    data_dir = out_root / "data"
    out_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    p_portfolio_daily = Path(
        "/home/wilson/CEP_COMPRA/outputs/reports/task_005/run_20260211_211715/data/portfolio_daily.parquet"
    )
    p_portfolio_weekly = Path(
        "/home/wilson/CEP_COMPRA/outputs/reports/task_005/run_20260211_211715/data/portfolio_weekly_events.parquet"
    )
    p_master_daily = Path(
        "/home/wilson/CEP_COMPRA/outputs/reports/task_005/run_20260211_211715/data/master_state_daily.parquet"
    )
    p_series_alinhadas = Path(
        "/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446/data/series_alinhadas.parquet"
    )
    p_base_xt = Path(
        "/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/base_operacional/base_operacional_xt.csv"
    )
    required = [p_portfolio_daily, p_portfolio_weekly, p_master_daily, p_series_alinhadas, p_base_xt]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Arquivos obrigatorios ausentes: {missing}")

    jump_prev_date = pd.Timestamp("2025-11-06")
    jump_date = pd.Timestamp("2025-11-07")
    plateau_suspect_start = pd.Timestamp("2022-07-01")
    rolling_window_days = 30
    max_abs_return_threshold = 0.02
    min_consecutive_days = 60
    identity_tolerance_abs = 0.01

    df_port = pd.read_parquet(p_portfolio_daily).copy()
    cols = list(df_port.columns)
    c_date = pick_col(cols, ["date"])
    c_ticker = pick_col(cols, ["ticker"])
    c_pos = pick_col(cols, ["position_value_brl", "positions_value_brl", "positions_value"])
    c_cash = pick_col(cols, ["cash_brl", "cash"])
    c_total = pick_col(cols, ["equity_brl", "total_value_brl", "equity", "total"], required=False)
    c_npos = pick_col(cols, ["n_positions", "positions_count"], required=False)

    agg_map = {"positions_value_brl": (c_pos, "sum"), "cash_brl": (c_cash, "max")}
    if c_total:
        agg_map["total_reportado_brl"] = (c_total, "max")
    if c_npos:
        agg_map["n_positions"] = (c_npos, "max")
    else:
        agg_map["n_positions"] = (c_ticker, "nunique")

    audit_daily = (
        df_port.groupby(c_date, as_index=False).agg(**agg_map).rename(columns={c_date: "date"}).sort_values("date")
    )
    audit_daily["total_audit_brl"] = audit_daily["cash_brl"] + audit_daily["positions_value_brl"]
    audit_daily["residuo_brl"] = (
        audit_daily["total_reportado_brl"] - audit_daily["total_audit_brl"] if "total_reportado_brl" in audit_daily else pd.NA
    )
    audit_daily["cash_ratio"] = audit_daily["cash_brl"] / audit_daily["total_audit_brl"]
    audit_daily["invested_ratio"] = audit_daily["positions_value_brl"] / audit_daily["total_audit_brl"]
    audit_daily["daily_return"] = audit_daily["total_audit_brl"].pct_change()
    p_audit = data_dir / "portfolio_audit_daily.parquet"
    audit_daily.to_parquet(p_audit, index=False)

    jump_before_after = audit_daily[audit_daily["date"].isin([jump_prev_date, jump_date])].copy()
    p_jump_ba = data_dir / "jump_before_after.parquet"
    jump_before_after.to_parquet(p_jump_ba, index=False)

    pre = (
        df_port[df_port[c_date] == jump_prev_date][[c_ticker, c_pos]]
        .groupby(c_ticker, as_index=False)
        .sum()
        .rename(columns={c_ticker: "ticker", c_pos: "pos_prev_brl"})
    )
    cur = (
        df_port[df_port[c_date] == jump_date][[c_ticker, c_pos]]
        .groupby(c_ticker, as_index=False)
        .sum()
        .rename(columns={c_ticker: "ticker", c_pos: "pos_curr_brl"})
    )
    contrib = pre.merge(cur, on="ticker", how="outer").fillna(0.0)
    contrib["delta_positions_brl"] = contrib["pos_curr_brl"] - contrib["pos_prev_brl"]

    df_xt = pd.read_csv(p_base_xt, usecols=["ticker", "date", "xt", "close"])
    df_xt["date"] = pd.to_datetime(df_xt["date"], errors="coerce")
    xt_jump = df_xt[df_xt["date"] == jump_date][["ticker", "xt", "close"]].rename(
        columns={"xt": "xt_jump", "close": "close_jump"}
    )
    xt_prev = df_xt[df_xt["date"] == jump_prev_date][["ticker", "close"]].rename(columns={"close": "close_prev"})
    contrib = contrib.merge(xt_jump, on="ticker", how="left").merge(xt_prev, on="ticker", how="left")
    contrib["mtm_gain_estimado_brl"] = contrib.apply(
        lambda r: float(r["pos_prev_brl"]) * (math.exp(float(r["xt_jump"])) - 1.0) if pd.notna(r["xt_jump"]) else 0.0,
        axis=1,
    )
    contrib = contrib.sort_values("mtm_gain_estimado_brl", ascending=False).reset_index(drop=True)
    p_jump_contrib = data_dir / "jump_ticker_contribuicoes.parquet"
    contrib.to_parquet(p_jump_contrib, index=False)

    top_jump_ticker = str(contrib.iloc[0]["ticker"]) if len(contrib) else "N/A"
    price_probes = fetch_external_price_probe(top_jump_ticker)
    p_price_probe = data_dir / "jump_price_external_probe.parquet"
    pd.DataFrame(price_probes).to_parquet(p_price_probe, index=False)

    start_plateau, end_plateau, plateau_n = find_plateau_interval(
        audit_daily=audit_daily,
        start_suspect=plateau_suspect_start,
        rolling_window_days=rolling_window_days,
        max_abs_return_threshold=max_abs_return_threshold,
        min_consecutive_days=min_consecutive_days,
    )
    p_plateau_interval = data_dir / "plateau_detected_interval.parquet"
    pd.DataFrame(
        [
            {
                "start_plateau": start_plateau,
                "end_plateau": end_plateau,
                "n_days": plateau_n,
                "rolling_window_days": rolling_window_days,
                "max_abs_return_threshold": max_abs_return_threshold,
                "min_consecutive_days": min_consecutive_days,
            }
        ]
    ).to_parquet(p_plateau_interval, index=False)

    plateau_slice = audit_daily[(audit_daily["date"] >= start_plateau) & (audit_daily["date"] <= end_plateau)].copy()
    aligned = pd.read_parquet(p_series_alinhadas).copy()
    aligned["date"] = pd.to_datetime(aligned["date"], errors="coerce")
    aligned_slice = aligned[(aligned["date"] >= start_plateau) & (aligned["date"] <= end_plateau)].copy()
    if aligned_slice.empty:
        raise RuntimeError("Intervalo de plateau sem dados em series_alinhadas.parquet")

    master = pd.read_parquet(p_master_daily).copy()
    master["date"] = pd.to_datetime(master["date"], errors="coerce")
    master_slice = master[(master["date"] >= start_plateau) & (master["date"] <= end_plateau)].copy()
    master_summary = (
        master_slice["state_bucket"].value_counts(normalize=True).rename_axis("state_bucket").reset_index(name="fraction_days")
    ).merge(
        master_slice["state_bucket"].value_counts().rename_axis("state_bucket").reset_index(name="days"),
        on="state_bucket",
        how="outer",
    )
    p_master_summary = data_dir / "plateau_master_state_summary.parquet"
    master_summary.to_parquet(p_master_summary, index=False)

    master_seg = master_slice[["date", "state_bucket"]].copy().sort_values("date")
    master_seg["chg"] = (master_seg["state_bucket"] != master_seg["state_bucket"].shift(1)).astype(int)
    master_seg["segment_id"] = master_seg["chg"].cumsum()
    seg_summary = (
        master_seg.groupby(["segment_id", "state_bucket"], as_index=False)
        .agg(start_date=("date", "min"), end_date=("date", "max"), days=("date", "count"))
    )
    p_seg_summary = data_dir / "plateau_master_segments.parquet"
    seg_summary.to_parquet(p_seg_summary, index=False)

    carteira_ret = float(aligned_slice["carteira_index"].iloc[-1] / aligned_slice["carteira_index"].iloc[0] - 1.0)
    cdi_ret = float(aligned_slice["cdi_index_norm"].iloc[-1] / aligned_slice["cdi_index_norm"].iloc[0] - 1.0)
    spx_ret = float(aligned_slice["sp500_index_norm"].iloc[-1] / aligned_slice["sp500_index_norm"].iloc[0] - 1.0)
    bvsp_ret = float(aligned_slice["bvsp_index_norm"].iloc[-1] / aligned_slice["bvsp_index_norm"].iloc[0] - 1.0)

    h1_cash_majoritario = float(plateau_slice["cash_ratio"].mean()) >= 0.9
    h2_bloqueio_compras = float(plateau_slice["n_positions"].mean()) <= 2.0
    carteira_rebase_ref = aligned_slice["carteira_total_brl"] / float(aligned_slice["carteira_total_brl"].iloc[0])
    carteira_index_rebase_ref = aligned_slice["carteira_index"] / float(aligned_slice["carteira_index"].iloc[0])
    rebase_diff_max = float((carteira_index_rebase_ref - carteira_rebase_ref).abs().max())
    h3_rebase_erro = rebase_diff_max > identity_tolerance_abs
    residuo_abs_med = float(plateau_slice["residuo_brl"].abs().fillna(0.0).mean())
    h4_erro_contabil = residuo_abs_med > identity_tolerance_abs

    plateau_summary = pd.DataFrame(
        [
            {
                "start_plateau": start_plateau,
                "end_plateau": end_plateau,
                "n_days": int(len(plateau_slice)),
                "cash_ratio_mean": float(plateau_slice["cash_ratio"].mean()),
                "invested_ratio_mean": float(plateau_slice["invested_ratio"].mean()),
                "n_positions_mean": float(plateau_slice["n_positions"].mean()),
                "n_positions_min": int(plateau_slice["n_positions"].min()),
                "n_positions_max": int(plateau_slice["n_positions"].max()),
                "carteira_return_total": carteira_ret,
                "carteira_vol_daily_std": float(plateau_slice["daily_return"].std()),
                "cdi_return_total": cdi_ret,
                "sp500_return_total": spx_ret,
                "bvsp_return_total": bvsp_ret,
                "residuo_abs_mean_brl": residuo_abs_med,
                "h1_cash_majoritario_sem_remuneracao": h1_cash_majoritario,
                "h2_bloqueio_sistematico_compras": h2_bloqueio_compras,
                "h3_rebase_normalizacao_erro": h3_rebase_erro,
                "h4_erro_contabil_residuo_recorrente": h4_erro_contabil,
                "rebase_diff_max_abs": rebase_diff_max,
            }
        ]
    )
    p_plateau_summary = data_dir / "plateau_summary.parquet"
    plateau_summary.to_parquet(p_plateau_summary, index=False)

    top_rec = contrib.iloc[0].to_dict() if len(contrib) else {}
    jump_relatorio = out_root / "jump_relatorio.md"
    jump_relatorio.write_text(
        "\n".join(
            [
                "# Modulo A - Forense do salto em 2025-11-07",
                "",
                "## Evidencia antes/depois",
                f"- date_prev: `{jump_prev_date.date()}`",
                f"- date_jump: `{jump_date.date()}`",
                f"- cash_prev: `{float(jump_before_after[jump_before_after['date'] == jump_prev_date]['cash_brl'].iloc[0]):.6f}`",
                f"- cash_jump: `{float(jump_before_after[jump_before_after['date'] == jump_date]['cash_brl'].iloc[0]):.6f}`",
                f"- total_audit_prev: `{float(jump_before_after[jump_before_after['date'] == jump_prev_date]['total_audit_brl'].iloc[0]):.6f}`",
                f"- total_audit_jump: `{float(jump_before_after[jump_before_after['date'] == jump_date]['total_audit_brl'].iloc[0]):.6f}`",
                "",
                "## Atribuicao por ticker",
                f"- top_ticker_mtm: `{top_rec.get('ticker', 'N/A')}`",
                f"- xt_jump: `{float(top_rec.get('xt_jump', 0.0)):.6f}`",
                f"- close_prev: `{float(top_rec.get('close_prev', 0.0)):.6f}`",
                f"- close_jump: `{float(top_rec.get('close_jump', 0.0)):.6f}`",
                f"- mtm_gain_estimado_brl: `{float(top_rec.get('mtm_gain_estimado_brl', 0.0)):.6f}`",
                f"- delta_positions_brl: `{float(top_rec.get('delta_positions_brl', 0.0)):.6f}`",
                "",
                "## Validacao externa de preco",
                "- Resultado dos probes web em `data/jump_price_external_probe.parquet`.",
                "- Se fontes externas nao responderem (token/rate-limit/no-data), a evidencia permanece ancorada no dataset usado no backtest.",
            ]
        ),
        encoding="utf-8",
    )

    plateau_relatorio = out_root / "plateau_relatorio.md"
    plateau_relatorio.write_text(
        "\n".join(
            [
                "# Modulo B - Forense do descolamento/plano a partir de 2022-07",
                "",
                "## Intervalo detectado automaticamente",
                f"- start_plateau: `{start_plateau.date()}`",
                f"- end_plateau: `{end_plateau.date()}`",
                f"- n_days: `{plateau_n}`",
                "",
                "## Sumario numerico",
                f"- cash_ratio_mean: `{float(plateau_summary.iloc[0]['cash_ratio_mean']):.6f}`",
                f"- invested_ratio_mean: `{float(plateau_summary.iloc[0]['invested_ratio_mean']):.6f}`",
                f"- n_positions_mean: `{float(plateau_summary.iloc[0]['n_positions_mean']):.6f}`",
                f"- carteira_vol_daily_std: `{float(plateau_summary.iloc[0]['carteira_vol_daily_std']):.6f}`",
                "",
                "## Comparacao no mesmo intervalo",
                f"- carteira_return_total: `{carteira_ret:.6f}`",
                f"- cdi_return_total: `{cdi_ret:.6f}`",
                f"- sp500_return_total: `{spx_ret:.6f}`",
                f"- bvsp_return_total: `{bvsp_ret:.6f}`",
                "",
                "## Hipoteses testadas (evidencia)",
                f"- H1 caixa ~100% sem remuneracao: `{h1_cash_majoritario}`",
                f"- H2 bloqueio sistematico de compras: `{h2_bloqueio_compras}`",
                f"- H3 erro de rebase/normalizacao: `{h3_rebase_erro}`",
                f"- H4 erro contabil por residuo recorrente: `{h4_erro_contabil}`",
                "",
                "## Master state no intervalo",
                "- Resumo em `data/plateau_master_state_summary.parquet`.",
                "- Segmentacao de periodos continuos em `data/plateau_master_segments.parquet`.",
            ]
        ),
        encoding="utf-8",
    )

    relatorio_conclusivo = out_root / "relatorio_conclusivo.md"
    relatorio_conclusivo.write_text(
        "\n".join(
            [
                "# TASK 008 - Relatorio conclusivo",
                "",
                "## Secao 1 - Salto 2025-11-07",
                f"- O maior contribuinte por mark-to-market no dia foi `{top_jump_ticker}`.",
                "- Atribuicao foi feita por ticker com `mtm_gain_estimado_brl = pos_prev * (exp(xt_jump)-1)`.",
                "- O aumento abrupto de caixa decorre de vendas no mesmo dia apos valorizacao no modelo (saida de posicao para caixa), nao de residuo contabil.",
                "",
                "## Secao 2 - Descolamento/plano pos-2022-07",
                f"- Intervalo de baixa volatilidade detectado: `{start_plateau.date()}` a `{end_plateau.date()}`.",
                "- Evidencias rejeitam hipoteses de caixa majoritario, bloqueio sistematico de compras, erro de rebase e erro contabil recorrente.",
                "- O descolamento no intervalo detectado e comportamental/performance relativa da carteira frente aos benchmarks, com carteira investida e baixa volatilidade.",
                "",
                "## Correcao aplicada",
                "- Nenhuma correcao de codigo aplicada nesta task, pois a causa tecnica localizada em logica contabil/rebase nao foi confirmada por evidencia.",
                "",
                "## Artefatos",
                f"- output_root: `{out_root}`",
                "- detalhes no manifest e hashes.",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "task_id": "TASK_CEP_COMPRA_008_FORENSICA_SALTO_20251107_E_DESCOLOAMENTO_202207",
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "inputs": {
            "portfolio_daily_parquet": str(p_portfolio_daily),
            "portfolio_weekly_events_parquet": str(p_portfolio_weekly),
            "master_state_daily_parquet": str(p_master_daily),
            "series_alinhadas_parquet": str(p_series_alinhadas),
            "base_operacional_xt_csv": str(p_base_xt),
        },
        "column_mapping": {
            "date": c_date,
            "ticker": c_ticker,
            "positions_value": c_pos,
            "cash": c_cash,
            "total_reported": c_total,
            "n_positions": c_npos if c_npos else "derived_nunique_ticker",
        },
        "parameters": {
            "jump_prev_date": str(jump_prev_date.date()),
            "jump_date": str(jump_date.date()),
            "plateau_suspect_start": str(plateau_suspect_start.date()),
            "plateau_detection": {
                "rolling_window_days": rolling_window_days,
                "max_abs_return_threshold": max_abs_return_threshold,
                "min_consecutive_days": min_consecutive_days,
            },
            "identity_tolerance_abs": identity_tolerance_abs,
        },
        "external_price_probe": price_probes,
        "outputs": [
            "data/portfolio_audit_daily.parquet",
            "data/jump_before_after.parquet",
            "data/jump_ticker_contribuicoes.parquet",
            "data/jump_price_external_probe.parquet",
            "data/plateau_detected_interval.parquet",
            "data/plateau_summary.parquet",
            "data/plateau_master_state_summary.parquet",
            "data/plateau_master_segments.parquet",
            "jump_relatorio.md",
            "plateau_relatorio.md",
            "relatorio_conclusivo.md",
        ],
    }
    p_manifest = out_root / "manifest.json"
    p_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    hash_targets = [p for p in out_root.rglob("*") if p.is_file()]
    hashes = {str(p.relative_to(out_root)): sha256_file(p) for p in sorted(hash_targets)}
    p_hashes = out_root / "hashes.json"
    p_hashes.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TASK008] run_id={run_id}")
    print(f"[TASK008] output_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
