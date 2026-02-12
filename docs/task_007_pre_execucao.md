# TASK_CEP_COMPRA_007 - Pre-execucao

## S0 - Baseline da Task 006 selecionado

- Diretorio base inspecionado: `/home/wilson/CEP_COMPRA/outputs/plots/task_006/`
- Run mais recente escolhido: `run_20260211_213227`
- Arquivos encontrados no run baseline:
  - `data/carteira_cdi_sp500_alinhado.parquet`
  - `manifest.json`
  - `hashes.json`
  - `carteira_vs_cdi_vs_sp500.html`

## Mapeamento de colunas da carteira (auditado)

Input:

- `/home/wilson/CEP_COMPRA/outputs/reports/task_005/run_20260211_211715/data/portfolio_daily.parquet`

Colunas identificadas:

- `date`, `position_value_brl`, `cash_brl`, `equity_brl`, `ticker`, `weight`, `n_positions`

Regra:

- `carteira_total_calc = sum(position_value_brl) + cash_brl`
- `carteira_total_used = equity_brl` quando `equity_brl` ja inclui caixa
- checagem: `max_abs_diff(equity_brl - carteira_total_calc) = 5.820766091346741e-11` -> `equity_brl` inclui caixa

## Fontes de benchmarks planejadas

- CDI (preferencial): Banco Central SGS serie 12
- S&P 500: Stooq (`^spx`)
- ^BVSP:

  - tentativa BRAPI (`^BVSP`)
  - fallback Stooq (`^bvsp`)
  - fallback final (se ambos falharem): SSOT local IBOV do CEP_NA_BOLSA em `outputs/ssot/precos_brutos/ibov/brapi/20260204/precos_brutos_ibov.csv`

## Regra de alinhamento

- Calendario de referencia: carteira
- Benchmarks: forward-fill em dias sem atualizacao
- bfill apenas no inicio se necessario
- Normalizacao: base 1.0 no primeiro dia da carteira
