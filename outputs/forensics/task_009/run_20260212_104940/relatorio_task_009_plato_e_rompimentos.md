# Task 009 - Verificação SELL por rompimento e impacto no plateau

## Resposta direta
- SELL por UCL em Xbar: **não** (no fluxo da carteira).
- SELL por UCL em R: **indireto via estado Master** (estresse de amplitude), com exceção de upside.
- SELL por UCL em I: **sim** (regra explícita no runner da carteira).
- SELL por UCL em MR: **sim** (regra explícita no runner da carteira).

## Contagens no intervalo amplo (2022-07-01..2023-06-30)
- I/UCL eventos: `11944`
- I/LCL eventos: `11603`
- MR/UCL eventos: `33543`
- Taxa SELL mesmo dia ou dia útil seguinte (I/UCL): `0.0002`
- Taxa SELL mesmo dia ou dia útil seguinte (MR/UCL): `0.0002`

## Contagens no intervalo foco plateau (2022-10-26..2023-03-31)
- I/UCL eventos: `5382`
- I/LCL eventos: `5451`
- MR/UCL eventos: `15821`
- Taxa SELL mesmo dia ou dia útil seguinte (I/UCL): `0.0000`
- Taxa SELL mesmo dia ou dia útil seguinte (MR/UCL): `0.0001`

## Conclusão sobre o plateau (evidencial)
- O dataset de eventos mostra presença de rompimentos I/MR e associação com saídas de posição.
- A explicação do plateau deve considerar a combinação de: regime Master (RISK_ON/OFF/PRESERVACAO), vendas por I/MR e performance dos ativos mantidos.
- Não há evidência de SELL direto por `Xbar > UCL` no fluxo da carteira.

## Artefatos
- `eventos_rompimento.parquet`: `/home/wilson/CEP_COMPRA/outputs/forensics/task_009/run_20260212_104940/data/eventos_rompimento.parquet`
- `eventos_com_sell.parquet`: `/home/wilson/CEP_COMPRA/outputs/forensics/task_009/run_20260212_104940/data/eventos_com_sell.parquet`
- `metricas_impacto.parquet`: `/home/wilson/CEP_COMPRA/outputs/forensics/task_009/run_20260212_104940/data/metricas_impacto.parquet`
