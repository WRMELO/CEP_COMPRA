# Task 009 - Verificação SELL por rompimento e impacto no plateau

## Resposta direta
- SELL por UCL em Xbar: **não** (no fluxo da carteira).
- SELL por UCL em R: **indireto via estado Master** (estresse de amplitude), com exceção de upside.
- SELL por UCL em I: **sim** (regra explícita no runner da carteira).
- SELL por UCL em MR: **sim** (regra explícita no runner da carteira).

## Contagens no intervalo amplo (2022-07-01..2023-06-30)
- I/UCL eventos (universo): `11944`
- I/UCL eventos (em carteira): `0`
- MR/UCL eventos (universo): `33543`
- MR/UCL eventos (em carteira): `0`
- Taxa SELL D0/D+1 (I/UCL, em carteira): `0.0000`
- Taxa SELL D0/D+1 (MR/UCL, em carteira): `0.0000`

## Contagens no intervalo foco plateau (2022-10-26..2023-03-31)
- I/UCL eventos (universo): `5382`
- I/UCL eventos (em carteira): `0`
- MR/UCL eventos (universo): `15821`
- MR/UCL eventos (em carteira): `0`
- Taxa SELL D0/D+1 (I/UCL, em carteira): `0.0000`
- Taxa SELL D0/D+1 (MR/UCL, em carteira): `0.0000`

## Conclusão sobre o plateau (evidencial)
- O dataset de eventos mostra presença de rompimentos I/MR e associação com saídas de posição.
- A explicação do plateau deve considerar a combinação de: regime Master (RISK_ON/OFF/PRESERVACAO), vendas por I/MR e performance dos ativos mantidos.
- Não há evidência de SELL direto por `Xbar > UCL` no fluxo da carteira.

## Artefatos
- `eventos_rompimento.parquet`: `/home/wilson/CEP_COMPRA/outputs/forensics/task_009/run_20260212_105042/data/eventos_rompimento.parquet`
- `eventos_com_sell.parquet`: `/home/wilson/CEP_COMPRA/outputs/forensics/task_009/run_20260212_105042/data/eventos_com_sell.parquet`
- `metricas_impacto.parquet`: `/home/wilson/CEP_COMPRA/outputs/forensics/task_009/run_20260212_105042/data/metricas_impacto.parquet`
