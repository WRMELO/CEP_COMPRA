# Task 012 - Pos-task (fechamento)

## 0. Identificacao

- LLM autor: `[LLM: GPT 5.3 Codex]`
- Data/hora: 2026-02-12
- Task ID: `TASK_CEP_COMPRA_012_EMENDA_V13_REGRAS_E_BACKTEST_M1_COM_SSOT_SETOR`
- Executor: GPT 5.3 Codex

## 1. OVERALL PASS/FAIL

- **OVERALL: PASS**

## 2. Entregas executadas

- Emenda v1.3 criada sem alterar Constituicao:
  - `docs/emendas/EMENDA_CEP_COMPRA_V1_3_REGRAS_COMPRA_VENDA.md`
- SSOT de setor em Parquet criado e atualizado:
  - `outputs/ssot/setores/20260212/setores_ticker.parquet`
  - `outputs/ssot/setores/ssot_latest/setores_ticker_latest.parquet`
- Patch de venda com excecao de upside aplicado:
  - `tools/task_004_backtest_runner.py`
  - `tools/task_005_report_runner.py`
- Smoke test documentado:
  - `docs/task_012_smoke_sell_patch.md`
- M1 implementado e documentado:
  - `src/cep/mecanismos/m1_compra_regras_v13.py`
  - `docs/task_012_m1_design.md`
- Backtest comparativo M0 vs M1 executado:
  - `outputs/backtests/task_012/run_20260212_114129/...`

## 3. Cobertura do SSOT de setor

- B3: `854/854` conhecidos (`100.00%`)
- BDR: `822/838` conhecidos (`98.09%`)
- UNKNOWN total: `16` (BDR)

Fontes:

- Primaria: BRAPI local raw
- Fallback Acoes: `segment` do raw SSOT B3 (listed companies)
- Fallback BDR: coluna `setor` do SSOT BDR

## 4. Resultado comparativo M0 vs M1

- M1:
  - `equity_final=4.138194`
  - `total_return=3.138194`
  - `max_drawdown=-0.781062`
- M0:
  - `equity_final=2.055620`
  - `total_return=1.055620`
  - `max_drawdown=-0.357772`

Observacao: havia um erro de janela no primeiro run (incluindo 2026-02-04, que inflou o grafico). O run oficial foi corrigido limitando o periodo ao intervalo canonico ate 2025-12-31.

## 5. Paths finais

- Run backtest: `/home/wilson/CEP_COMPRA/outputs/backtests/task_012/run_20260212_114129`
- Métricas: `/home/wilson/CEP_COMPRA/outputs/backtests/task_012/run_20260212_114129/consolidated/metricas_consolidadas.parquet`
- Ranking: `/home/wilson/CEP_COMPRA/outputs/backtests/task_012/run_20260212_114129/consolidated/ranking_final.parquet`
- Plot: `/home/wilson/CEP_COMPRA/outputs/backtests/task_012/run_20260212_114129/plots/m0_vs_m1_vs_cdi_sp500_bvsp.html`
- Evidências de hash: `docs/task_012_evidencias_hashes.md`
