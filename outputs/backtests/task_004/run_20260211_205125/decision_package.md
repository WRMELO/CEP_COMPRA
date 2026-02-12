# Decision Package - TASK_CEP_COMPRA_004

## Escopo executado

- Universo: todos os ativos SSOT (ACOES+B3 + BDR+B3)
- Portfolio alvo: 10 posicoes
- Compra semanal: segunda-feira de manha ou proximo pregao
- Venda diaria: bundle congelado CEP_NA_BOLSA (referenciado, sem alteracao)
- Mecanismos executados: `M0` (todos os mecanismos presentes em `mecanismos.json`)

## Ranking (informativo)

Fonte: `consolidated/ranking_final.csv`

| rank_informativo | mecanismo | equity_final | total_return | max_drawdown |
| --- | --- | --- | --- | --- |
| 1 | M0 | 151.82129619775708 | 150.82129619775708 | -0.4748832975623565 |

## Criterio de selecao do vencedor

**Bloqueio de governanca:** a especificacao canônica v1.2 nao define explicitamente uma regra operacional unica para escolher o vencedor entre mecanismos (funcao agregadora/ordem de desempate formal).  
Conforme a task, nao foi inventada regra.

Resultado:

- Ranking comparativo foi produzido.
- **Vencedor oficial nao foi declarado** por ausencia de criterio explicito na especificacao.

## Evidencias

- `manifest.json`
- `consolidated/metricas_consolidadas.csv`
- `consolidated/regret_dp2_consolidado.csv`
- `consolidated/ranking_final.csv`
- `raw/M0_equity_curve.csv`
- `raw/M0_drawdown.csv`
- `logs/M0.log`
