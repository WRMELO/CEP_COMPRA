#!/usr/bin/env python3
from typing import Dict, List, Tuple

import pandas as pd


def select_m1_tickers(
    candidates: pd.DataFrame,
    target_positions: int = 10,
    sector_cap_positions: int = 2,
    min_b3: int = 5,
    max_b3: int = 8,
    min_bdr: int = 2,
    max_bdr: int = 5,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Seleciona tickers para M1 com regras v1.3.

    candidates deve conter colunas:
    - ticker
    - score
    - ret_lookback
    - liquidity_ok (bool)
    - company_key
    - sector
    - asset_class ("B3"|"BDR")
    """
    required = {
        "ticker",
        "score",
        "ret_lookback",
        "liquidity_ok",
        "company_key",
        "sector",
        "asset_class",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates sem colunas obrigatorias: {sorted(missing)}")

    work = candidates.copy()
    work["sector"] = work["sector"].fillna("UNKNOWN").replace("", "UNKNOWN")
    work = work[work["liquidity_ok"]].copy()

    # Uma classe por empresa: manter maior retorno acumulado no lookback.
    work = (
        work.sort_values(["ret_lookback", "score", "ticker"], ascending=[False, False, True])
        .drop_duplicates(subset=["company_key"], keep="first")
        .sort_values(["score", "ret_lookback", "ticker"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    chosen: List[str] = []
    log_rows: List[Dict] = []
    sec_count: Dict[str, int] = {}
    b3_count = 0
    bdr_count = 0

    def can_add(row: pd.Series) -> Tuple[bool, str]:
        nonlocal b3_count, bdr_count
        sec = row["sector"]
        cls = row["asset_class"]
        if sec_count.get(sec, 0) >= sector_cap_positions:
            return False, "sector_cap"
        if cls == "B3" and b3_count >= max_b3:
            return False, "b3_max"
        if cls == "BDR" and bdr_count >= max_bdr:
            return False, "bdr_max"
        return True, "ok"

    def add_row(row: pd.Series, phase: str) -> None:
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
        log_rows.append({"ticker": t, "phase": phase, "decision": "selected", "reason": "ok"})

    # Phase 1: minimo B3
    for _, row in work[work["asset_class"] == "B3"].iterrows():
        if len(chosen) >= target_positions or b3_count >= min_b3:
            break
        ok, reason = can_add(row)
        if ok:
            add_row(row, "min_b3")
        else:
            log_rows.append({"ticker": row["ticker"], "phase": "min_b3", "decision": "skipped", "reason": reason})

    # Phase 2: minimo BDR
    for _, row in work[work["asset_class"] == "BDR"].iterrows():
        if len(chosen) >= target_positions or bdr_count >= min_bdr:
            break
        if row["ticker"] in chosen:
            continue
        ok, reason = can_add(row)
        if ok:
            add_row(row, "min_bdr")
        else:
            log_rows.append({"ticker": row["ticker"], "phase": "min_bdr", "decision": "skipped", "reason": reason})

    # Phase 3: completar carteira gulosa
    for _, row in work.iterrows():
        if len(chosen) >= target_positions:
            break
        if row["ticker"] in chosen:
            continue
        ok, reason = can_add(row)
        if ok:
            add_row(row, "fill")
        else:
            log_rows.append({"ticker": row["ticker"], "phase": "fill", "decision": "skipped", "reason": reason})

    log_df = pd.DataFrame(log_rows)
    return chosen[:target_positions], log_df
