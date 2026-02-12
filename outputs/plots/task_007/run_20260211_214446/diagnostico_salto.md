# Diagnostico de descontinuidade da Carteira

- Run task_006 baseline selecionado: `/home/wilson/CEP_COMPRA/outputs/plots/task_006/run_20260211_213227`
- Maior salto por abs_daily_return: 2025-11-07
- date_prev: 2025-11-06
- daily_return: 1.712103
- delta_total: 176652.95
- delta_equity: 176652.95
- delta_cash: 186544.28
- delta_positions: -9891.32
- master_state_prev: RISK_ON
- master_state_jump: RISK_ON
- Causa provavel: inconsistencia contabil provavel no runner: caixa sobe acima da variacao liquida com Master em RISK_ON; indicativo de descontinuidade tecnica, nao mudanca de regime.

## Evidencias
- `jumps_topk.parquet`: `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446/data/jumps_topk.parquet`
- `jump_window_60d.parquet`: `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446/data/jump_window_60d.parquet`
- `carteira_total_daily.parquet`: `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446/data/carteira_total_daily.parquet`