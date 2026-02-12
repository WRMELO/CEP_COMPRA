# TASK_CEP_COMPRA_005 - Pre-execucao (S0/S1)

## Entrypoint real localizado no repositorio

- Base de simulacao usada: `tools/task_004_backtest_runner.py` (task anterior, walk-forward completo)
- Entrypoint desta task de relatorio semestral (instrumentado): `tools/task_005_report_runner.py`

## Parametros efetivos aplicados

- `mechanism_id`: `M0`
- `initial_cash_brl`: `100000.0`
- `start_date_calendar`: `2018-07-01`
- `end_date_calendar`: `2025-12-31`
- `min_lookback_trading_days`: `62`
- `portfolio_target_positions`: `10`
- Frequencia de snapshot: semestral
- Conversao calendario -> pregao:
  - inicio: `FIRST_TRADING_DAY_ON_OR_AFTER`
  - snapshots: `LAST_TRADING_DAY_ON_OR_BEFORE` (com excecao do primeiro marco inicial, alinhado ao inicio efetivo)

## Congelamentos de governanca confirmados

- Universo: `TODOS_OS_ATIVOS_DO_SSOT_CEP_NA_BOLSA` (ACOES + BDR)
- Venda diaria: bundle CEP_NA_BOLSA congelado
- Compra semanal DP3: segunda de manha ou proximo pregao
- Sem alteracao de Constituicao, SSOTs ou bundle

## Paths reais de dados usados

- Base operacional XT:
  - `/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/base_operacional/base_operacional_xt.csv`
- Limites por ticker (Burners):
  - `/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos_v5/merged/limits_per_ticker.csv`
- Estados do Master por dia:
  - `/home/wilson/CEP_NA_BOLSA/outputs/experimentos/fase1_calibracao/exp/20260209/dataset_sizing/master_states.csv`
- SSOT ACOES:
  - `/home/wilson/CEP_NA_BOLSA/outputs/ssot/acoes/b3/20260204/ssot_acoes_b3.csv`
- SSOT BDR:
  - `/home/wilson/CEP_NA_BOLSA/outputs/ssot/bdr/b3/20260204/ssot_bdr_b3.csv`

## Saida principal (caminho principal)

- `outputs/reports/task_005/run_<id>/...` em `/home/wilson/CEP_COMPRA`
