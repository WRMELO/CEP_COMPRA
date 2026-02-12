# Modulo B - Forense do descolamento/plano a partir de 2022-07

## Intervalo detectado automaticamente
- start_plateau: `2022-10-26`
- end_plateau: `2023-03-31`
- n_days: `108`

## Sumario numerico
- cash_ratio_mean: `0.007976`
- invested_ratio_mean: `0.992024`
- n_positions_mean: `8.907407`
- carteira_vol_daily_std: `0.004316`

## Comparacao no mesmo intervalo
- carteira_return_total: `-0.053615`
- cdi_return_total: `0.056369`
- sp500_return_total: `0.072759`
- bvsp_return_total: `-0.096502`

## Hipoteses testadas (evidencia)
- H1 caixa ~100% sem remuneracao: `False`
- H2 bloqueio sistematico de compras: `False`
- H3 erro de rebase/normalizacao: `False`
- H4 erro contabil por residuo recorrente: `False`

## Master state no intervalo
- Resumo em `data/plateau_master_state_summary.parquet`.
- Segmentacao de periodos continuos em `data/plateau_master_segments.parquet`.