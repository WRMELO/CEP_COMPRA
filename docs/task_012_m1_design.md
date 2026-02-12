# Task 012 - Design do M1 (v1.3)

## Objetivo

Definir o mecanismo `M1` de compra semanal como extensao do ranking do `M0`, com filtros e restricoes adicionais.

## Regras implementadas

1. Liquidez excludente: `count(volume>0) >= 50` em lookback de `62` dias.
2. Uma classe por empresa: manter a classe de maior retorno acumulado no lookback.
3. Cap por setor: no maximo 2 ativos por setor (20% da carteira com 10 posicoes).
4. Mix B3/BDR: B3 entre 5 e 8; BDR entre 2 e 5.

## Fontes de dados

- Universo B3: `ssot_acoes_b3.csv`
- Universo BDR: `ssot_bdr_b3.csv` (ticker operacional via `ticker_bdr`)
- Setor: `outputs/ssot/setores/ssot_latest/setores_ticker_latest.parquet`
- Volume diario: cache local BRAPI em `data/raw/market/*/brapi/20260204/*.json`

## Seleção e desempate

- Ranking base por score (`media xt lookback`).
- Desempate: score, retorno acumulado no lookback, ticker.
- Selecao gulosa em fases: minimo B3, minimo BDR, preenchimento final.

## Logs de decisão

Cada evento semanal persiste:
- tickers selecionados;
- quantidade selecionada;
- mecanismo (`M0`/`M1`);
- estado do Master no dia.
