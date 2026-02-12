#!/usr/bin/env python3
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class Paths:
    spec_dir: Path
    bundle_manifest: Path
    base_xt_csv: Path
    limits_csv: Path
    master_states_csv: Path
    ssot_acoes_csv: Path
    ssot_bdr_csv: Path
    sizing_json: Path


def detect_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    return first.startswith("version https://git-lfs.github.com/spec/v1")


def first_trading_day_by_week(dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
    out: List[pd.Timestamp] = []
    seen = set()
    for d in dates:
        key = (d.isocalendar().year, d.isocalendar().week)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def progress_line(prefix: str, idx: int, total: int, t0: float) -> str:
    done = idx + 1
    pct = (done / total) * 100.0 if total else 100.0
    elapsed = time.time() - t0
    eta = ((elapsed / done) * (total - done)) if done and total else 0.0
    return f"{prefix} {done}/{total} ({pct:5.1f}%) elapsed={elapsed:7.1f}s eta={eta:7.1f}s"


def main() -> int:
    repo_root = Path("/home/wilson/.cursor/worktrees/CEP_COMPRA/gxt")
    main_repo_root = Path("/home/wilson/CEP_COMPRA")
    spec_dir = repo_root / "docs/spec_canonica_v1"
    bundle_manifest = repo_root / "docs/bundle_cep_na_bolsa/manifest.json"

    run_id = datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
    out_root = main_repo_root / "outputs/backtests/task_004" / run_id
    out_raw = out_root / "raw"
    out_consolidated = out_root / "consolidated"
    out_logs = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    out_raw.mkdir(parents=True, exist_ok=True)
    out_consolidated.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    mecanismos = json.loads((spec_dir / "mecanismos.json").read_text(encoding="utf-8"))
    walk_forward = json.loads((spec_dir / "walk_forward.json").read_text(encoding="utf-8"))
    reward = json.loads((spec_dir / "reward.json").read_text(encoding="utf-8"))
    bundle = json.loads(bundle_manifest.read_text(encoding="utf-8"))

    if mecanismos.get("universo_mode") != "TODOS_OS_ATIVOS_DO_SSOT_CEP_NA_BOLSA":
        raise RuntimeError("universo_mode inesperado")
    if int(mecanismos.get("portfolio_target_positions", 0)) != 10:
        raise RuntimeError("portfolio_target_positions != 10")
    if walk_forward.get("dp3_regra") != "SEGUNDA_FEIRA_MANHA_OU_PROXIMO_DIA_UTIL":
        raise RuntimeError("dp3_regra inesperada")

    # Paths reais do bundle
    component_map = {c["component"]: Path(c["path"]) for c in bundle["bundle_components"]}
    ssot_map = {s["asset_class"]: Path(s["manifest_path"]) for s in bundle["ssot_universo"]}

    paths = Paths(
        spec_dir=spec_dir,
        bundle_manifest=bundle_manifest,
        base_xt_csv=Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/base_operacional/base_operacional_xt.csv"),
        limits_csv=Path("/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/merged/limits_per_ticker.csv"),
        master_states_csv=Path("/home/wilson/CEP_NA_BOLSA/outputs/experimentos/fase1_calibracao/exp/20260209/dataset_sizing/master_states.csv"),
        ssot_acoes_csv=Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/acoes/b3/20260204/ssot_acoes_b3.csv"),
        ssot_bdr_csv=Path("/home/wilson/CEP_NA_BOLSA/outputs/ssot/bdr/b3/20260204/ssot_bdr_b3.csv"),
        sizing_json=component_map["sizing_config_v2"],
    )

    required_files = [
        paths.base_xt_csv,
        paths.limits_csv,
        paths.master_states_csv,
        paths.ssot_acoes_csv,
        paths.ssot_bdr_csv,
        paths.sizing_json,
        ssot_map["acoes"],
        ssot_map["bdr"],
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    lfs_blocked = [str(p) for p in required_files if detect_lfs_pointer(p)]
    if missing or lfs_blocked:
        raise RuntimeError(f"preflight_failed missing={missing} lfs_blocked={lfs_blocked}")

    # Carregamento de dados
    df_base = pd.read_csv(paths.base_xt_csv, parse_dates=["date"])
    df_limits = pd.read_csv(paths.limits_csv)
    df_master = pd.read_csv(paths.master_states_csv, parse_dates=["date"])
    df_ssot_a = pd.read_csv(paths.ssot_acoes_csv)
    df_ssot_b = pd.read_csv(paths.ssot_bdr_csv)
    sizing = json.loads(paths.sizing_json.read_text(encoding="utf-8"))

    # Universo total (SSOT Acoes + BDR)
    universe = set(df_ssot_a["ticker"].astype(str)) | set(df_ssot_b["ticker"].astype(str))
    df_base = df_base[df_base["ticker"].isin(universe)].copy()
    df_base.sort_values(["date", "ticker"], inplace=True)

    # Tabelas auxiliares
    xt = df_base.pivot(index="date", columns="ticker", values="xt").sort_index()
    closes = df_base.pivot(index="date", columns="ticker", values="close").sort_index()
    master_by_date = dict(zip(df_master["date"], df_master["state"]))
    limits = df_limits.set_index("ticker")

    trading_dates = [d for d in xt.index if d in master_by_date]
    buy_dates = set(first_trading_day_by_week(trading_dates))
    k = int(mecanismos["mecanismos"][0]["parametros"]["K_subgrupos"])
    target_n = int(mecanismos["portfolio_target_positions"])
    w_cap = float(sizing.get("w_cap", 0.15))

    metrics_rows = []
    regret_rows = []

    for m_idx, m in enumerate(mecanismos["mecanismos"]):
        mech_id = m["id"]
        log_lines: List[str] = []

        cash = 1.0
        pos: Dict[str, float] = {}  # valor monetario por ticker
        prev_xt: Dict[str, float] = {}
        equity_curve: List[Tuple[pd.Timestamp, float]] = []

        # janela de eventos semanais (walk-forward completo)
        event_dates = [d for d in trading_dates if d in buy_dates]
        t0 = time.time()
        for i, d in enumerate(trading_dates):
            # mark-to-market diario
            row_xt = xt.loc[d]
            for t in list(pos.keys()):
                r = row_xt.get(t)
                if pd.notna(r):
                    pos[t] *= math.exp(float(r))

            state = master_by_date.get(d, "RISK_ON")
            # venda diaria congelada
            if state == "PRESERVAÇÃO_TOTAL" or state == "PRESERVACAO_TOTAL":
                cash += sum(pos.values())
                pos.clear()
            else:
                for t in list(pos.keys()):
                    if t not in limits.index:
                        continue
                    r = row_xt.get(t)
                    if pd.isna(r):
                        continue
                    lim = limits.loc[t]
                    mr = abs(float(r) - float(prev_xt.get(t, 0.0)))
                    n_sub = int(lim["n"]) if "n" in lim and pd.notna(lim["n"]) else 3
                    hist_t = xt.loc[:d, t].dropna().tail(max(2, n_sub))
                    xbar_t = float(hist_t.mean()) if len(hist_t) >= 2 else float("nan")
                    r_t = float(hist_t.max() - hist_t.min()) if len(hist_t) >= 2 else float("nan")
                    upside_extreme = (
                        (pd.notna(lim.get("xbar_ucl", None)) and pd.notna(xbar_t) and xbar_t > float(lim["xbar_ucl"]))
                        or float(r) > float(lim["i_ucl"])
                    )
                    stress_amp = (
                        pd.notna(lim.get("r_ucl", None))
                        and pd.notna(r_t)
                        and (r_t > float(lim["r_ucl"]))
                        and (not upside_extreme)
                    )
                    sell_i_lcl = float(r) < float(lim["i_lcl"])
                    sell_i_ucl = float(r) > float(lim["i_ucl"]) and (not upside_extreme)
                    sell_mr = mr > float(lim["mr_ucl"]) and (not upside_extreme)
                    if sell_i_lcl or sell_i_ucl or sell_mr or stress_amp:
                        cash += pos.pop(t)
                    prev_xt[t] = float(r)

            # compra semanal no primeiro pregao da semana (segunda ou proximo util)
            if d in buy_dates and state == "RISK_ON":
                hist = xt.loc[:d].tail(k)
                # elegiveis: limite por ticker disponivel + k historico + controle no dia
                mean_k = hist.mean(skipna=True)
                eligible = []
                for t, mu in mean_k.items():
                    if pd.isna(mu):
                        continue
                    if t not in limits.index:
                        continue
                    rt = row_xt.get(t)
                    if pd.isna(rt):
                        continue
                    lim = limits.loc[t]
                    if float(rt) < float(lim["i_lcl"]) or float(rt) > float(lim["i_ucl"]):
                        continue
                    eligible.append((t, float(mu)))

                eligible.sort(key=lambda x: x[1], reverse=True)
                ranked = [t for t, _ in eligible]

                # M0: prioriza ativos ja presentes no topo + prossegue com novos
                selected_existing = [t for t in ranked if t in pos][:target_n]
                selected_new = [t for t in ranked if t not in pos]
                target_list = (selected_existing + selected_new)[:target_n]

                total_equity = cash + sum(pos.values())
                target_slot = total_equity / target_n if target_n else 0.0
                cap_value = total_equity * w_cap

                for t in target_list:
                    current = pos.get(t, 0.0)
                    desired = min(target_slot, cap_value)
                    need = max(0.0, desired - current)
                    alloc = min(need, cash)
                    if alloc > 0:
                        pos[t] = current + alloc
                        cash -= alloc

                # DP2 regret semanal (U_t = log-retorno ate proximo evento semanal)
                event_idx = event_dates.index(d) if d in event_dates else -1
                if event_idx >= 0 and event_idx < len(event_dates) - 1:
                    d_next = event_dates[event_idx + 1]
                    window = xt.loc[(xt.index > d) & (xt.index <= d_next)]
                    if not window.empty:
                        u = window.sum(skipna=True)  # utilidade canônica: soma de xt (log-retorno acumulado)
                        # S*: top-N por U_t sob mesmo universo/target
                        u_items = [(t, float(v)) for t, v in u.items() if pd.notna(v)]
                        u_items.sort(key=lambda x: x[1], reverse=True)
                        s_star = [t for t, _ in u_items[:target_n]]
                        s_exec = target_list[:target_n]
                        u_star = sum(dict(u_items).get(t, 0.0) for t in s_star)
                        u_exec = sum(dict(u_items).get(t, 0.0) for t in s_exec)
                        regret = max(0.0, u_star - u_exec)
                        regret_rows.append(
                            {
                                "mecanismo": mech_id,
                                "event_date": d.strftime("%Y-%m-%d"),
                                "next_event_date": d_next.strftime("%Y-%m-%d"),
                                "u_star": u_star,
                                "u_exec": u_exec,
                                "regret_t": regret,
                                "s_star": ",".join(s_star),
                                "s_exec": ",".join(s_exec),
                            }
                        )

                # progresso com ETA (tqdm-like)
                done_events = event_dates.index(d) + 1 if d in event_dates else 0
                if d in event_dates and (done_events == 1 or done_events % 25 == 0 or done_events == len(event_dates)):
                    line = progress_line(f"[{mech_id}] events", done_events - 1, len(event_dates), t0)
                    print(line, flush=True)
                    log_lines.append(line)

            equity = cash + sum(pos.values())
            equity_curve.append((d, equity))

        # outputs por mecanismo
        df_eq = pd.DataFrame(equity_curve, columns=["date", "equity"])
        df_eq["rolling_max"] = df_eq["equity"].cummax()
        df_eq["drawdown"] = (df_eq["equity"] / df_eq["rolling_max"]) - 1.0
        df_eq.to_csv(out_raw / f"{mech_id}_equity_curve.csv", index=False)
        df_eq[["date", "drawdown"]].to_csv(out_raw / f"{mech_id}_drawdown.csv", index=False)
        (out_logs / f"{mech_id}.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        total_return = float(df_eq["equity"].iloc[-1] - 1.0)
        mdd = float(df_eq["drawdown"].min())
        metrics_rows.append(
            {
                "mecanismo": mech_id,
                "n_trading_days": len(df_eq),
                "n_weekly_events": len(event_dates),
                "total_return": total_return,
                "max_drawdown": mdd,
                "equity_final": float(df_eq["equity"].iloc[-1]),
            }
        )

    # consolidado
    df_metrics = pd.DataFrame(metrics_rows)
    df_regret = pd.DataFrame(regret_rows)
    df_metrics.to_csv(out_consolidated / "metricas_consolidadas.csv", index=False)
    df_regret.to_csv(out_consolidated / "regret_dp2_consolidado.csv", index=False)

    # ranking informativo (sem criterio oficial de vencedor)
    df_rank = df_metrics.sort_values(["equity_final", "max_drawdown"], ascending=[False, False]).copy()
    df_rank["rank_informativo"] = range(1, len(df_rank) + 1)
    df_rank.to_csv(out_consolidated / "ranking_final.csv", index=False)

    manifest = {
        "task_id": "TASK_CEP_COMPRA_004_BACKTEST_WALK_FORWARD_ALL_ASSETS",
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "mecanismos_testados": [m["id"] for m in mecanismos["mecanismos"]],
        "outputs_root": str(out_root),
        "notes": [
            "Progresso e ETA registrados em logs por mecanismo (tqdm-like).",
            "Ranking gerado em modo informativo; criterio oficial de vencedor deve existir explicitamente na especificacao."
        ],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
