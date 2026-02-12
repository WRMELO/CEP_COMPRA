# Task 015 - Pos-task (fechamento)

## 0. Identificacao

- LLM autor: `[LLM: GPT 5.3 Codex]`
- Data/hora: 2026-02-12
- Task ID: `TASK_CEP_COMPRA_015_FORENSICA_FASES_M0_M1_E_PROPOSTA_M3_COM_GATING_SOB_CONTROLE`
- Executor: GPT 5.3 Codex

## 1. OVERALL PASS/FAIL

- **OVERALL: PASS**

## 2. Janela canônica e validação

- Todas as séries de entrada e de saída foram validadas no intervalo `2018-07-01..2025-12-31`.
- O runner possui hard fail para qualquer série fora da janela.

## 3. Artefatos forenses por fases (M0 vs M1)

- Relatório: `outputs/forensics/task_015/run_20260212_121309/relatorio_fases_m0_m1.md`
- Holdings semanais:
  - `outputs/forensics/task_015/run_20260212_121309/data/weekly_holdings_m0.parquet`
  - `outputs/forensics/task_015/run_20260212_121309/data/weekly_holdings_m1.parquet`
- Atribuição:
  - `outputs/forensics/task_015/run_20260212_121309/data/contribuicao_ticker_fases.parquet`
  - `outputs/forensics/task_015/run_20260212_121309/data/contribuicao_setor_assetclass_fases.parquet`
- Auditoria de gatilhos:
  - `outputs/forensics/task_015/run_20260212_121309/data/audit_sell_triggers_windows.parquet`

## 4. Evidência por janela (síntese)

- W1 (2018-07..2021-06): M1 supera M0 em retorno e suporta drawdown menor.
- W2 (2021-07..2022-12): M1 perde mais e apresenta drawdown substancialmente pior que M0.
- W2a (Ago/21) e W2b (Mai/22): quedas de M1 mais profundas e com maior rotação/cash médio.
- W3 (2024-09..2025-11): M1 apresenta retorno positivo na janela, porém com drawdown alto.

Todas as afirmações acima estão detalhadas e rastreadas no `relatorio_fases_m0_m1.md` e nos Parquets de auditoria/contribuição.

## 5. Proposta M3 e comparativo

- Proposta escrita (não integrada): `docs/emendas/PROPOSTA_M3_CEP_COMPRA_V1_4_NAO_INTEGRADA.md`
- M3 implementado como ranking combinado apenas com medidas existentes no pipeline.
- Backtest comparativo M0 vs M1 vs M3:
  - `outputs/backtests/task_015_m3/run_20260212_121309/consolidated/metricas_consolidadas.parquet`
  - `outputs/backtests/task_015_m3/run_20260212_121309/plots/m0_vs_m1_vs_m3_vs_cdi_sp500_bvsp.html`

Métricas consolidadas:

- M1: `equity_final=4.138194`, `max_drawdown=-0.781062`
- M3: `equity_final=2.385440`, `max_drawdown=-0.680335`
- M0: `equity_final=2.055620`, `max_drawdown=-0.357772`
