# Task 007 - Pos-task (fechamento)

## 0. Identificacao

- LLM autor: `[LLM: GPT 5.3 Codex]`
- Data/hora: 2026-02-11
- Task ID: `TASK_CEP_COMPRA_007_DIAGNOSTICO_SALTO_E_PLOTLY_COM_BVSP`
- Executor: GPT 5.3 Codex

## 1. OVERALL PASS/FAIL

- **OVERALL: PASS**

## 2. Maior salto identificado (evidencia numerica)

- `date_prev`: `2025-11-06`
- `date_jump`: `2025-11-07`
- `daily_return`: `+171.2103%`
- `delta_total`: `+176652.95`
- `delta_equity`: `+176652.95`
- `delta_cash`: `+186544.28`
- `delta_positions`: `-9891.32`
- `master_state_prev`: `RISK_ON`
- `master_state_jump`: `RISK_ON`

## 3. Causa provavel (baseada em evidencia)

- Diagnostico indica **inconsistencia contabil provavel no runner**:
  - caixa sobe muito (`+186.5k`) enquanto variacao liquida total e menor (`+176.7k`) e as posicoes ate reduzem (`-9.9k`);
  - Master permaneceu em `RISK_ON` no antes/depois;
  - isso sugere descontinuidade tecnica de contabilizacao (nao mudanca de regime do Master).

## 4. Fontes usadas para benchmarks

- CDI:
  - Banco Central SGS serie 12
  - URL: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial=02/07/2018&dataFinal=31/12/2025`
- S&P 500:
  - Stooq (`^spx`)
  - URL: `https://stooq.com/q/d/l/?s=%5Espx&i=d`
- ^BVSP:
  - BRAPI tentou e falhou por token (`MISSING_TOKEN`)
  - Stooq (`^bvsp`) sem dados
  - Fallback usado: SSOT local `CEP_NA_BOLSA`:
    - `/home/wilson/CEP_NA_BOLSA/outputs/ssot/precos_brutos/ibov/brapi/20260204/precos_brutos_ibov.csv`

## 5. Paths principais

Run oficial:

- `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446`

HTML Plotly:

- `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446/carteira_vs_cdi_vs_sp500_vs_bvsp.html`

Parquet alinhado:

- `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446/data/series_alinhadas.parquet`

Diagnostico:

- `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446/diagnostico_salto.md`

Governanca:

- `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446/manifest.json`
- `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446/hashes.json`
- `docs/task_007_evidencias_hashes.md`
