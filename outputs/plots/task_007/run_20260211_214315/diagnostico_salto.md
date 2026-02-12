# Diagnostico de descontinuidade da Carteira

- Run task_006 baseline selecionado: `/home/wilson/CEP_COMPRA/outputs/plots/task_006/run_20260211_213227`
- Maior salto por abs_daily_return: 2025-11-07
- date_prev: 2025-11-06
- daily_return: 1.712103
- delta_total: 176652.95
- delta_equity: 176652.95
- delta_cash: 186544.28
- Causa provavel: movimento relevante de caixa/realocacao contribuiu para o salto.

## Evidencias
- `jumps_topk.parquet`: `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214315/data/jumps_topk.parquet`
- `jump_window_60d.parquet`: `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214315/data/jump_window_60d.parquet`
- `carteira_total_daily.parquet`: `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214315/data/carteira_total_daily.parquet`