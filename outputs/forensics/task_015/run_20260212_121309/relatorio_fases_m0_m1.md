# Task 015 - Relatorio forense por fases (M0 vs M1)

## Janela canônica
- Todas as séries usadas passaram no assert: 2018-07-01..2025-12-31

## Evidências principais
- `weekly_holdings_m0.parquet`: /home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309/data/weekly_holdings_m0.parquet
- `weekly_holdings_m1.parquet`: /home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309/data/weekly_holdings_m1.parquet
- `contribuicao_ticker_fases.parquet`: /home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309/data/contribuicao_ticker_fases.parquet
- `contribuicao_setor_assetclass_fases.parquet`: /home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309/data/contribuicao_setor_assetclass_fases.parquet
- `audit_sell_triggers_windows.parquet`: /home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309/data/audit_sell_triggers_windows.parquet

## Síntese por janela
### W1_outperformance
- M0: retorno=1.6810, max_drawdown=-0.2739
- M1: retorno=4.0802, max_drawdown=-0.2085
- M0 audit: sold=200, i_ucl=0, mr_ucl=132, cash_ratio=0.1161, turnover=0.0584
- M1 audit: sold=334, i_ucl=0, mr_ucl=220, cash_ratio=0.1703, turnover=0.0947

### W2_underperformance
- M0: retorno=-0.0808, max_drawdown=-0.2536
- M1: retorno=-0.6074, max_drawdown=-0.6930
- M0 audit: sold=53, i_ucl=0, mr_ucl=39, cash_ratio=0.0339, turnover=0.0213
- M1 audit: sold=152, i_ucl=0, mr_ucl=112, cash_ratio=0.1267, turnover=0.0689

### W2a_drawdown_aug21
- M0: retorno=-0.0263, max_drawdown=-0.0824
- M1: retorno=-0.1380, max_drawdown=-0.2931
- M0 audit: sold=14, i_ucl=0, mr_ucl=7, cash_ratio=0.0683, turnover=0.0543
- M1 audit: sold=29, i_ucl=0, mr_ucl=19, cash_ratio=0.1617, turnover=0.1124

### W2b_drawdown_may22
- M0: retorno=-0.0314, max_drawdown=-0.2347
- M1: retorno=-0.1947, max_drawdown=-0.5864
- M0 audit: sold=3, i_ucl=0, mr_ucl=2, cash_ratio=0.0413, turnover=0.0087
- M1 audit: sold=16, i_ucl=0, mr_ucl=11, cash_ratio=0.2235, turnover=0.0670

### W3_late_decline
- M0: retorno=-0.0288, max_drawdown=-0.3559
- M1: retorno=0.8013, max_drawdown=-0.7811
- M0 audit: sold=40, i_ucl=0, mr_ucl=29, cash_ratio=0.0228, turnover=0.0163
- M1 audit: sold=119, i_ucl=0, mr_ucl=76, cash_ratio=0.1231, turnover=0.0675

