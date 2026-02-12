#!/usr/bin/env python3
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class RunConfig:
    mechanism_id: str = "M0"
    initial_cash: float = 100000.0
    start_calendar: str = "2018-07-01"
    end_calendar: str = "2025-12-31"
    min_lookback: int = 62
    target_positions: int = 10
    snapshot_dates_calendar: Tuple[str, ...] = (
        "2018-07-01",
        "2018-12-31",
        "2019-06-30",
        "2019-12-31",
        "2020-06-30",
        "2020-12-31",
        "2021-06-30",
        "2021-12-31",
        "2022-06-30",
        "2022-12-31",
        "2023-06-30",
        "2023-12-31",
        "2024-06-30",
        "2024-12-31",
        "2025-06-30",
        "2025-12-31",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_lfs_pointer(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.readline().startswith("version https://git-lfs.github.com/spec/v1")


def first_trading_on_or_after(dates: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    cand = dates[dates >= target]
    if len(cand) == 0:
        raise ValueError(f"Sem pregao em/apos {target.date()}")
    return cand[0]


def last_trading_on_or_before(dates: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    cand = dates[dates <= target]
    if len(cand) == 0:
        raise ValueError(f"Sem pregao em/antes {target.date()}")
    return cand[-1]


def first_trading_day_by_week(dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
    out = []
    seen = set()
    for d in dates:
        key = (d.isocalendar().year, d.isocalendar().week)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def progress_line(done: int, total: int, t0: float) -> str:
    pct = (done / total) * 100 if total else 100.0
    elapsed = time.time() - t0
    eta = ((elapsed / done) * (total - done)) if done else 0.0
    return f"[TASK005] weekly_events {done}/{total} ({pct:5.1f}%) elapsed={elapsed:7.1f}s eta={eta:7.1f}s"


def semester_key(d: pd.Timestamp) -> str:
    return f"{d.year}-S1" if d.month <= 6 else f"{d.year}-S2"


def master_bucket(state: str) -> str:
    s = str(state).upper()
    if "PRESERV" in s:
        return "PROTECAO_TOTAL"
    if s.endswith("ON") or "RISK_ON" in s:
        return "ON"
    return "OUTROS_ESTADOS_SE_EXISTIREM_NO_BUNDLE"


def main() -> int:
    cfg = RunConfig()
    workspace_root = Path("/home/wilson/.cursor/worktrees/CEP_COMPRA/gxt")
    principal_root = Path("/home/wilson/CEP_COMPRA")
    now = datetime.now(UTC)
    run_id = now.strftime("run_%Y%m%d_%H%M%S")

    out_root = principal_root / "outputs/reports/task_005" / run_id
    data_dir = out_root / "data"
    logs_dir = out_root / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    spec_dir = workspace_root / "docs/spec_canonica_v1"
    bundle_manifest_path = workspace_root / "docs/bundle_cep_na_bolsa/manifest.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    mecan = json.loads((spec_dir / "mecanismos.json").read_text(encoding="utf-8"))
    walk = json.loads((spec_dir / "walk_forward.json").read_text(encoding="utf-8"))

    comp = {c["component"]: Path(c["path"]) for c in bundle_manifest["bundle_components"]}
    ssot_by_class = {s["asset_class"]: Path(s["manifest_path"]) for s in bundle_manifest["ssot_universo"]}

    # Paths reais
    p_base_xt = Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/base_operacional/base_operacional_xt.csv")
    p_limits = Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/merged/limits_per_ticker.csv")
    p_master_daily = Path("/home/wilson/CEP_NA_BOLSA/outputs/experimentos/fase1_calibracao/exp/20260209/dataset_sizing/master_states.csv")
    p_ssot_a = Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/acoes/b3/20260204/ssot_acoes_b3.csv")
    p_ssot_b = Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/bdr/b3/20260204/ssot_bdr_b3.csv")

    required = [p_base_xt, p_limits, p_master_daily, p_ssot_a, p_ssot_b, comp["sizing_config_v2"], ssot_by_class["acoes"], ssot_by_class["bdr"]]
    missing = [str(p) for p in required if not p.exists()]
    lfs_blocked = [str(p) for p in required if detect_lfs_pointer(p)]
    if missing or lfs_blocked:
        raise RuntimeError(f"S0_FAIL missing={missing} lfs_blocked={lfs_blocked}")

    if mecan.get("universo_mode") != "TODOS_OS_ATIVOS_DO_SSOT_CEP_NA_BOLSA":
        raise RuntimeError("universo_mode divergente")
    if int(mecan.get("portfolio_target_positions", 0)) != cfg.target_positions:
        raise RuntimeError("target_positions divergente")
    if walk.get("dp3_regra") != "SEGUNDA_FEIRA_MANHA_OU_PROXIMO_DIA_UTIL":
        raise RuntimeError("DP3 divergente")

    # Dados
    df_base = pd.read_csv(p_base_xt, parse_dates=["date"])
    df_limits = pd.read_csv(p_limits).set_index("ticker")
    df_master = pd.read_csv(p_master_daily, parse_dates=["date"]).sort_values("date")
    df_ssot_a = pd.read_csv(p_ssot_a)
    df_ssot_b = pd.read_csv(p_ssot_b)
    sizing = json.loads(comp["sizing_config_v2"].read_text(encoding="utf-8"))
    w_cap = float(sizing.get("w_cap", 0.15))

    universe = set(df_ssot_a["ticker"].astype(str)) | set(df_ssot_b["ticker"].astype(str))
    df_base = df_base[df_base["ticker"].isin(universe)].copy()
    df_base.sort_values(["date", "ticker"], inplace=True)
    df_master = df_master[df_master["date"].isin(df_base["date"].unique())].copy()

    xt = df_base.pivot(index="date", columns="ticker", values="xt").sort_index()
    master_state = dict(zip(df_master["date"], df_master["state"]))
    trading_dates = pd.DatetimeIndex([d for d in xt.index if d in master_state])
    start_trade = first_trading_on_or_after(trading_dates, pd.Timestamp(cfg.start_calendar))
    end_trade = last_trading_on_or_before(trading_dates, pd.Timestamp(cfg.end_calendar))
    trading_dates = pd.DatetimeIndex([d for d in trading_dates if start_trade <= d <= end_trade])

    weekly_events = set(first_trading_day_by_week(list(trading_dates)))
    k = int(mecan["mecanismos"][0]["parametros"]["K_subgrupos"])

    cash = cfg.initial_cash
    pos: Dict[str, float] = {}
    prev_xt: Dict[str, float] = {}

    portfolio_daily_rows = []
    weekly_event_rows = []
    master_daily_rows = []

    t0 = time.time()
    event_counter = 0
    total_events = sum(1 for d in trading_dates if d in weekly_events)
    progress_log = []

    for d in trading_dates:
        row_xt = xt.loc[d]
        # mtm diario
        for t in list(pos.keys()):
            r = row_xt.get(t)
            if pd.notna(r):
                pos[t] *= math.exp(float(r))

        state = master_state[d]
        master_daily_rows.append({"date": d, "state": state, "state_bucket": master_bucket(state)})

        # venda diaria congelada
        if "PRESERV" in str(state).upper():
            cash += sum(pos.values())
            pos.clear()
        else:
            for t in list(pos.keys()):
                if t not in df_limits.index:
                    continue
                r = row_xt.get(t)
                if pd.isna(r):
                    continue
                lim = df_limits.loc[t]
                mr = abs(float(r) - float(prev_xt.get(t, 0.0)))
                if float(r) < float(lim["i_lcl"]) or float(r) > float(lim["i_ucl"]) or mr > float(lim["mr_ucl"]):
                    cash += pos.pop(t)
                prev_xt[t] = float(r)

        # compra semanal
        if d in weekly_events:
            event_counter += 1
            if event_counter == 1 or event_counter % 25 == 0 or event_counter == total_events:
                line = progress_line(event_counter, total_events, t0)
                print(line, flush=True)
                progress_log.append(line)

        if d in weekly_events and "RISK_ON" in str(state):
            hist = xt.loc[:d].tail(max(k, cfg.min_lookback))
            mean_k = hist.mean(skipna=True)
            eligible = []
            for t, mu in mean_k.items():
                if pd.isna(mu) or t not in df_limits.index:
                    continue
                rt = row_xt.get(t)
                if pd.isna(rt):
                    continue
                lim = df_limits.loc[t]
                if float(rt) < float(lim["i_lcl"]) or float(rt) > float(lim["i_ucl"]):
                    continue
                eligible.append((t, float(mu)))
            eligible.sort(key=lambda x: x[1], reverse=True)
            ranked = [t for t, _ in eligible]

            selected_existing = [t for t in ranked if t in pos][:cfg.target_positions]
            selected_new = [t for t in ranked if t not in pos]
            target_list = (selected_existing + selected_new)[:cfg.target_positions]

            total_equity = cash + sum(pos.values())
            slot = total_equity / cfg.target_positions if cfg.target_positions else 0.0
            cap_value = total_equity * w_cap

            for t in target_list:
                current = pos.get(t, 0.0)
                desired = min(slot, cap_value)
                need = max(0.0, desired - current)
                alloc = min(need, cash)
                if alloc > 0:
                    pos[t] = current + alloc
                    cash -= alloc
                weekly_event_rows.append(
                    {
                        "event_date": d,
                        "ticker": t,
                        "selected": True,
                        "allocation_brl": alloc,
                        "post_position_value_brl": pos.get(t, 0.0),
                        "state_master": state,
                    }
                )

        equity = cash + sum(pos.values())
        if pos:
            total_pos = sum(pos.values())
            for t, v in pos.items():
                portfolio_daily_rows.append(
                    {
                        "date": d,
                        "ticker": t,
                        "position_value_brl": v,
                        "weight": (v / total_pos) if total_pos else 0.0,
                        "cash_brl": cash,
                        "equity_brl": equity,
                        "n_positions": len(pos),
                    }
                )
        else:
            portfolio_daily_rows.append(
                {
                    "date": d,
                    "ticker": None,
                    "position_value_brl": 0.0,
                    "weight": 0.0,
                    "cash_brl": cash,
                    "equity_brl": equity,
                    "n_positions": 0,
                }
            )

    df_port_daily = pd.DataFrame(portfolio_daily_rows)
    df_weekly = pd.DataFrame(weekly_event_rows)
    df_master_daily = pd.DataFrame(master_daily_rows)

    # snapshots semestrais
    snapshot_rows = []
    for i, ds in enumerate(cfg.snapshot_dates_calendar):
        target = pd.Timestamp(ds)
        if i == 0:
            snap_trade = first_trading_on_or_after(trading_dates, target)
        else:
            snap_trade = last_trading_on_or_before(trading_dates, target)
        day_rows = df_port_daily[df_port_daily["date"] == snap_trade].copy()
        day_rows = day_rows[day_rows["ticker"].notna()].sort_values("position_value_brl", ascending=False).head(cfg.target_positions)
        tickers = day_rows["ticker"].tolist()
        while len(tickers) < cfg.target_positions:
            tickers.append(None)
        equity = float(day_rows["equity_brl"].iloc[0]) if not day_rows.empty else float(df_port_daily[df_port_daily["date"] == snap_trade]["equity_brl"].iloc[0])
        cash_v = float(day_rows["cash_brl"].iloc[0]) if not day_rows.empty else float(df_port_daily[df_port_daily["date"] == snap_trade]["cash_brl"].iloc[0])
        snapshot_rows.append(
            {
                "snapshot_calendar_date": ds,
                "snapshot_trading_date": snap_trade,
                "cash_brl": cash_v,
                "equity_brl": equity,
                **{f"ticker_{j+1}": tickers[j] for j in range(cfg.target_positions)},
            }
        )
    df_snap = pd.DataFrame(snapshot_rows)

    # resumo Master por semestre
    dms = df_master_daily.sort_values("date").copy()
    dms["state_change"] = dms["state"] != dms["state"].shift(1)
    dms["segment_id"] = dms["state_change"].cumsum()
    seg = dms.groupby("segment_id").agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        state=("state", "first"),
        days=("date", "count"),
    ).reset_index(drop=True)
    seg["state_bucket"] = seg["state"].map(master_bucket)
    seg["semester"] = seg["start_date"].map(semester_key)

    summary = (
        seg.groupby(["semester", "state", "state_bucket"], as_index=False)
        .agg(period_count=("state", "count"), trading_days=("days", "sum"))
        .sort_values(["semester", "state_bucket", "state"])
    )

    # persist parquet
    p_port_daily = data_dir / "portfolio_daily.parquet"
    p_weekly = data_dir / "portfolio_weekly_events.parquet"
    p_master = data_dir / "master_state_daily.parquet"
    p_snap = data_dir / "snapshots_semestrais.parquet"
    p_master_sum = data_dir / "master_event_summary_semestral.parquet"
    df_port_daily.to_parquet(p_port_daily, index=False)
    df_weekly.to_parquet(p_weekly, index=False)
    df_master_daily.to_parquet(p_master, index=False)
    df_snap.to_parquet(p_snap, index=False)
    summary.to_parquet(p_master_sum, index=False)

    # report markdown
    report_md = out_root / "relatorio_semestral_m0_20180701_20251231.md"
    lines = []
    lines.append("# Relatorio semestral M0 (2018-07-01 a 2025-12-31)")
    lines.append("")
    lines.append(f"- Mecanismo: `{cfg.mechanism_id}`")
    lines.append(f"- Aporte inicial: R$ {cfg.initial_cash:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    lines.append("- Universo: SSOT Acoes + BDR (bundle CEP_NA_BOLSA)")
    lines.append("- Portfolio alvo: 10 posicoes")
    lines.append("- Regra semanal: segunda de manha; se sem pregao, proximo dia util")
    lines.append("")

    for _, r in df_snap.iterrows():
        sem = semester_key(pd.Timestamp(r["snapshot_trading_date"]))
        lines.append(f"## Snapshot {r['snapshot_calendar_date']} -> {pd.Timestamp(r['snapshot_trading_date']).date()} ({sem})")
        lines.append(f"- Cash: R$ {r['cash_brl']:.2f}")
        lines.append(f"- Equity: R$ {r['equity_brl']:.2f}")
        ticks = [r[f"ticker_{i}"] for i in range(1, cfg.target_positions + 1)]
        tick_str = ", ".join([t if isinstance(t, str) and t else "-" for t in ticks])
        lines.append("- Tickers da carteira (10): " + tick_str)

        sem_rows = summary[summary["semester"] == sem]
        if sem_rows.empty:
            lines.append("- Master no semestre: sem dados para o recorte.")
        else:
            lines.append("- Resumo Master no semestre:")
            for _, sr in sem_rows.iterrows():
                lines.append(
                    f"  - {sr['state_bucket']} / {sr['state']}: {int(sr['period_count'])} periodos, {int(sr['trading_days'])} dias de negociacao"
                )
        lines.append("")

    lines.append("## Evidencias e datasets")
    lines.append(f"- `portfolio_daily.parquet`: `{p_port_daily}`")
    lines.append(f"- `portfolio_weekly_events.parquet`: `{p_weekly}`")
    lines.append(f"- `master_state_daily.parquet`: `{p_master}`")
    lines.append(f"- `snapshots_semestrais.parquet`: `{p_snap}`")
    lines.append(f"- `master_event_summary_semestral.parquet`: `{p_master_sum}`")
    lines.append("")
    report_md.write_text("\n".join(lines), encoding="utf-8")

    # manifest
    manifest = {
        "task_id": "TASK_CEP_COMPRA_005_RELATORIO_SEMESTRAL_HOLDINGS_E_MASTER_M0_20180701_20251231",
        "run_id": run_id,
        "timestamp_utc": now.isoformat(),
        "command": "python tools/task_005_report_runner.py",
        "parameters": {
            "mechanism_id": cfg.mechanism_id,
            "initial_cash_brl": cfg.initial_cash,
            "start_date_calendar": cfg.start_calendar,
            "end_date_calendar": cfg.end_calendar,
            "min_lookback_trading_days": cfg.min_lookback,
            "portfolio_target_positions": cfg.target_positions,
        },
        "entrypoint_real": str(workspace_root / "tools/task_004_backtest_runner.py"),
        "outputs": {
            "portfolio_daily_parquet": str(p_port_daily),
            "portfolio_weekly_events_parquet": str(p_weekly),
            "master_state_daily_parquet": str(p_master),
            "snapshots_semestrais_parquet": str(p_snap),
            "master_event_summary_semestral_parquet": str(p_master_sum),
            "relatorio_md": str(report_md),
            "progress_log": str(logs_dir / "progress.log"),
        },
    }
    (logs_dir / "progress.log").write_text("\n".join(progress_log) + "\n", encoding="utf-8")
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    # hashes
    file_list = [p_port_daily, p_weekly, p_master, p_snap, p_master_sum, report_md, out_root / "manifest.json", logs_dir / "progress.log"]
    hashes = [{"path": str(p), "sha256": sha256_file(p)} for p in file_list]
    (out_root / "hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=True), encoding="utf-8")

    print(str(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
