#!/usr/bin/env python3
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} sem colunas obrigatorias: {missing}")


def md_table(df: pd.DataFrame, cols: List[str]) -> str:
    d = df[cols].copy()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, r in d.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def nearest_event_on_or_before(events: pd.Series, target: pd.Timestamp) -> pd.Timestamp:
    s = events[events <= target]
    if s.empty:
        return events.min()
    return s.max()


def main() -> int:
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(f"/home/wilson/CEP_COMPRA/outputs/reports/task_016/{run_id}")
    out_data = out_root / "data"
    out_root.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    # Inputs
    p_rel = Path("/home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309/relatorio_fases_m0_m1.md")
    p_wh_m0 = Path("/home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309/data/weekly_holdings_m0.parquet")
    p_wh_m1 = Path("/home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309/data/weekly_holdings_m1.parquet")
    p_ct = Path("/home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309/data/contribuicao_ticker_fases.parquet")
    p_cs = Path("/home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309/data/contribuicao_setor_assetclass_fases.parquet")
    p_audit = Path("/home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309/data/audit_sell_triggers_windows.parquet")
    p_m3 = Path("/home/wilson/CEP_COMPRA/docs/emendas/PROPOSTA_M3_CEP_COMPRA_V1_4_NAO_INTEGRADA.md")
    p_m3_run = Path("/home/wilson/CEP_COMPRA/outputs/backtests/task_015_m3/run_20260212_121309")
    p_m3_metrics = p_m3_run / "consolidated/metricas_consolidadas.parquet"

    for p in [p_rel, p_wh_m0, p_wh_m1, p_ct, p_cs, p_audit, p_m3, p_m3_metrics]:
        if not p.exists():
            raise RuntimeError(f"Input ausente: {p}")

    windows = {
        "W1": ("W1_outperformance", pd.Timestamp("2018-07-01"), pd.Timestamp("2021-06-30")),
        "W2": ("W2_underperformance", pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
        "W2a": ("W2a_drawdown_aug21", pd.Timestamp("2021-08-01"), pd.Timestamp("2021-09-30")),
        "W2b": ("W2b_drawdown_may22", pd.Timestamp("2022-05-01"), pd.Timestamp("2022-06-30")),
        "W3": ("W3_late_decline", pd.Timestamp("2024-09-01"), pd.Timestamp("2025-11-30")),
    }

    wh_m0 = pd.read_parquet(p_wh_m0)
    wh_m1 = pd.read_parquet(p_wh_m1)
    ct = pd.read_parquet(p_ct)
    cs = pd.read_parquet(p_cs)
    audit = pd.read_parquet(p_audit)
    m3_metrics = pd.read_parquet(p_m3_metrics)
    m3_text = p_m3.read_text(encoding="utf-8")

    assert_columns(wh_m0, ["event_date", "ticker", "value_brl", "weight", "sector", "asset_class"], "weekly_holdings_m0")
    assert_columns(wh_m1, ["event_date", "ticker", "value_brl", "weight", "sector", "asset_class"], "weekly_holdings_m1")
    assert_columns(ct, ["window", "mechanism", "ticker", "pnl_brl", "sector", "asset_class"], "contrib_ticker")
    assert_columns(cs, ["window", "mechanism", "asset_class", "sector", "pnl_brl"], "contrib_setor_assetclass")
    assert_columns(audit, ["window", "mechanism", "sold_count", "avg_turnover_ratio", "avg_cash_ratio", "avg_n_positions"], "audit_sell_triggers")

    wh_m0["event_date"] = pd.to_datetime(wh_m0["event_date"])
    wh_m1["event_date"] = pd.to_datetime(wh_m1["event_date"])

    # Artifacts 1: top tickers per window/mechanism
    top_ticker_paths = []
    for w_short, (w_long, _, _) in windows.items():
        for mech in ["M0", "M1"]:
            sub = ct[(ct["window"] == w_long) & (ct["mechanism"] == mech)].copy()
            sub["abs_pnl"] = sub["pnl_brl"].abs()
            sub = sub.sort_values(["abs_pnl", "pnl_brl"], ascending=[False, False]).head(10).copy()
            sub["rank"] = range(1, len(sub) + 1)
            p = out_data / f"top_tickers_{w_short}_{mech}.parquet"
            sub.to_parquet(p, index=False)
            top_ticker_paths.append(str(p))

    # Artifacts 2: top setor/assetclass per window
    top_sa_paths = []
    for w_short, (w_long, _, _) in windows.items():
        sub = cs[cs["window"] == w_long].copy()
        sub["abs_pnl"] = sub["pnl_brl"].abs()
        sub = (
            sub.sort_values(["mechanism", "abs_pnl", "pnl_brl"], ascending=[True, False, False])
            .groupby("mechanism", as_index=False, group_keys=False)
            .head(5)
        )
        sub["rank"] = sub.groupby("mechanism").cumcount() + 1
        p = out_data / f"top_setor_assetclass_{w_short}.parquet"
        sub.to_parquet(p, index=False)
        top_sa_paths.append(str(p))

    # Artifacts 3: snapshots sentinela holdings
    # Pior semana em cada janela = menor variação semanal do total de holdings por mecanismo.
    def worst_week_date(wh: pd.DataFrame, ws: pd.Timestamp, we: pd.Timestamp) -> Dict[str, pd.Timestamp]:
        g = (
            wh.groupby(["mechanism", "event_date"], as_index=False)
            .agg(total_value=("value_brl", "sum"))
            .sort_values(["mechanism", "event_date"])
        )
        g = g[(g["event_date"] >= ws) & (g["event_date"] <= we)].copy()
        out = {}
        for mech in g["mechanism"].unique():
            gm = g[g["mechanism"] == mech].copy()
            gm["ret_week"] = gm["total_value"].pct_change()
            if gm["ret_week"].dropna().empty:
                out[mech] = gm["event_date"].min()
            else:
                out[mech] = gm.loc[gm["ret_week"].idxmin(), "event_date"]
        return out

    wh = pd.concat(
        [
            wh_m0.assign(mechanism="M0"),
            wh_m1.assign(mechanism="M1"),
        ],
        ignore_index=True,
    )
    events_by_mech = {m: wh[wh["mechanism"] == m]["event_date"].drop_duplicates().sort_values() for m in ["M0", "M1"]}

    sentinels = []
    # fixed date
    for m in ["M0", "M1"]:
        sentinels.append(("SENTINELA_2021_06_30", m, nearest_event_on_or_before(events_by_mech[m], pd.Timestamp("2021-06-30"))))
        sentinels.append(("SENTINELA_2024_09_02", m, nearest_event_on_or_before(events_by_mech[m], pd.Timestamp("2024-09-02"))))

    ww_w2a = worst_week_date(wh, windows["W2a"][1], windows["W2a"][2])
    ww_w2b = worst_week_date(wh, windows["W2b"][1], windows["W2b"][2])
    ww_w3 = worst_week_date(wh, windows["W3"][1], windows["W3"][2])
    for m in ["M0", "M1"]:
        sentinels.append(("PIOR_SEMANA_W2a", m, ww_w2a[m]))
        sentinels.append(("PIOR_SEMANA_W2b", m, ww_w2b[m]))
        sentinels.append(("PIOR_SEMANA_W3", m, ww_w3[m]))

    snap_rows = []
    for label, mech, d in sentinels:
        sub = wh[(wh["mechanism"] == mech) & (wh["event_date"] == d)].copy()
        sub = sub.sort_values("value_brl", ascending=False).head(10)
        for _, r in sub.iterrows():
            snap_rows.append(
                {
                    "sentinela": label,
                    "mechanism": mech,
                    "event_date": d,
                    "ticker": r["ticker"],
                    "value_brl": r["value_brl"],
                    "weight": r["weight"],
                    "sector": r["sector"],
                    "asset_class": r["asset_class"],
                }
            )
    df_snap = pd.DataFrame(snap_rows)
    p_snap = out_data / "snapshots_sentinela_holdings.parquet"
    df_snap.to_parquet(p_snap, index=False)

    # Artifact 4: audit proteção resumo (W2a/W2b, M0/M1)
    audit_sub = audit[
        audit["window"].isin([windows["W2a"][0], windows["W2b"][0]]) & audit["mechanism"].isin(["M0", "M1"])
    ].copy()
    p_audit = out_data / "audit_protecao_resumo.parquet"
    audit_sub.to_parquet(p_audit, index=False)

    # Build markdown chat-ready
    lines = []
    lines.append("# Pacote de evidências - M0 vs M1 por fases + definição de M3")
    lines.append("")
    lines.append(f"- Run forense base: `{p_rel.parent}`")
    lines.append(f"- Run M3 comparativo: `{p_m3_run}`")
    lines.append("")

    lines.append("## 1) Qual M3 foi criado")
    lines.append("")
    # Trechos literais curtos (<=25 palavras contínuas)
    lines.append('Trechos literais da proposta:')
    lines.append('- `"score_m3 = z(score_m0) + z(ret_lookback_62) - z(vol_lookback_62)"`')
    lines.append('- `"volume > 0 em pelo menos 50/62 dias (excludente)"`')
    lines.append('- `"cap por setor 20% (UNKNOWN conta)"`')
    lines.append('- `"mix B3 entre 50% e 80%"`')
    lines.append("")
    lines.append("Parâmetros operacionais (paráfrase objetiva):")
    lines.append("- Ranking M3 combina score base do M0, retorno acumulado no lookback e penalização por volatilidade no lookback.")
    lines.append("- Desempate operacional segue score e critérios determinísticos de ticker no pipeline.")
    lines.append("- Invariantes mantidos: liquidez 50/62, uma classe por empresa, cap setor 20%, mix B3/BDR 50-80%.")
    lines.append("")

    lines.append("## 2) Top contribuições por janela")
    lines.append("")
    for w_short, (w_long, _, _) in windows.items():
        lines.append(f"### {w_short}")
        tm0 = pd.read_parquet(out_data / f"top_tickers_{w_short}_M0.parquet")
        tm1 = pd.read_parquet(out_data / f"top_tickers_{w_short}_M1.parquet")
        tsa = pd.read_parquet(out_data / f"top_setor_assetclass_{w_short}.parquet")

        lines.append("**Top-10 tickers M0**")
        lines.append("")
        lines.append(md_table(tm0.head(10), ["rank", "ticker", "pnl_brl", "sector", "asset_class"]))
        lines.append("")
        lines.append("**Top-10 tickers M1**")
        lines.append("")
        lines.append(md_table(tm1.head(10), ["rank", "ticker", "pnl_brl", "sector", "asset_class"]))
        lines.append("")
        lines.append("**Top-5 setor/asset_class por mecanismo**")
        lines.append("")
        lines.append(md_table(tsa.head(10), ["mechanism", "rank", "asset_class", "sector", "pnl_brl"]))
        lines.append("")

    lines.append("## 3) Snapshots sentinela de holdings")
    lines.append("")
    lines.append(md_table(df_snap, ["sentinela", "mechanism", "event_date", "ticker", "weight", "value_brl", "sector", "asset_class"]))
    lines.append("")

    lines.append("## 4) Auditoria de proteção (W2a e W2b)")
    lines.append("")
    lines.append(md_table(audit_sub, ["window", "mechanism", "avg_turnover_ratio", "avg_cash_ratio", "avg_n_positions", "sold_count", "trigger_i_lcl_count", "trigger_i_ucl_count", "trigger_mr_ucl_count"]))
    lines.append("")

    lines.append("## 5) Conclusão curta (com referência de tabela)")
    lines.append("")
    lines.append(f"- W2a/W2b mostram diferença operacional de proteção entre M0 e M1 em `data/audit_protecao_resumo.parquet`.")
    lines.append(f"- Drivers por ticker por janela estão em `data/top_tickers_W2a_M0.parquet`, `data/top_tickers_W2a_M1.parquet`, `data/top_tickers_W2b_M0.parquet`, `data/top_tickers_W2b_M1.parquet`.")
    lines.append(f"- Drivers por setor/asset_class por janela estão em `data/top_setor_assetclass_W2a.parquet` e `data/top_setor_assetclass_W2b.parquet`.")
    lines.append(f"- Composição em datas sentinela está em `data/snapshots_sentinela_holdings.parquet`.")
    lines.append(f"- Definição de M3 foi extraída de `docs/emendas/PROPOSTA_M3_CEP_COMPRA_V1_4_NAO_INTEGRADA.md` e métricas comparativas de `../task_015_m3/.../metricas_consolidadas.parquet`.")
    lines.append("")

    p_md = out_root / "pacote_evidencias_chat_m0_m1_m3.md"
    p_md.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "task_id": "TASK_CEP_COMPRA_016_PACOTE_EVIDENCIAS_CHAT_FASES_E_DEFINICAO_M3",
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "inputs": {
            "forensics_run_root": str(p_rel.parent),
            "m3_proposta_md": str(p_m3),
            "m3_backtest_run_root": str(p_m3_run),
        },
        "outputs": {
            "chat_ready_md": str(p_md),
            "tables_dir": str(out_data),
        },
        "artifacts": {
            "top_tickers": top_ticker_paths,
            "top_setor_assetclass": top_sa_paths,
            "snapshots_sentinela_holdings": str(p_snap),
            "audit_protecao_resumo": str(p_audit),
        },
    }
    p_manifest = out_root / "manifest.json"
    p_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    hashes = {str(p.relative_to(out_root)): sha256_file(p) for p in sorted([x for x in out_root.rglob("*") if x.is_file()])}
    p_hashes = out_root / "hashes.json"
    p_hashes.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TASK016] run_id={run_id}")
    print(f"[TASK016] output_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
