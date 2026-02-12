# EMENDA CEP_COMPRA v1.3 - Regras de Compra/Venda (sem alterar Constituicao)

## Escopo da emenda

Esta emenda altera regras operacionais do `CEP_COMPRA` sem modificar a Constituicao do `CEP_NA_BOLSA`.

- Venda (queimadores): excecao de upside em rompimentos positivos.
- Compra (M1): quatro regras novas com parametros fixados pelo Owner.
- Setor: criacao de SSOT oficial em Parquet para suportar cap por setor.

## Decisoes confirmadas do Owner

- `upside_extreme = (xbar > ucl_xbar) OR (xt > ucl_i)`
- `movimentacao_dia := volume > 0`
- Regra de liquidez excludente: `count(movimentacao_dia) >= 50` em `62` dias.
- Mais rentavel por empresa: retorno acumulado no lookback de `62` dias.

## Regra de venda v1.3 (queimadores)

Aplicar excecao de upside nos rompimentos positivos:

- Se `xt > ucl_i` e isso caracteriza `upside_extreme`, nao vender por I/UCL.
- Se `mr > mr_ucl`, vender apenas quando `NOT(upside_extreme)`.
- Compatibilidade com Master:
  - `stress_amp = (r > ucl_r) AND NOT(upside_extreme)`.

## Regra de compra M1

### R1 - Liquidez excludente

- Elegivel somente se `count(volume>0) >= 50` no lookback de `62` dias.

### R2 - Uma classe por empresa

- Se houver mais de uma classe para a mesma empresa, manter somente a classe mais rentavel no lookback.
- Chave de agrupamento:
  - Preferencia por identificador de provedor (`code_cvm`/`isin`/id equivalente).
  - Fallback deterministico: prefixo de 4 letras do ticker, com log.

### R3 - Cap por setor

- Maximo de `20%` por setor.
- Com 10 posicoes, operacionalmente equivale a no maximo 2 tickers por setor.
- Setor ausente deve ser `UNKNOWN` e conta como setor para o cap.

### R4 - Mix B3/BDR

- B3 entre `50%` e `80%`.
- Com 10 posicoes: B3 entre `5` e `8`; BDR entre `2` e `5`.

### Metodo de selecao

- Selecao gulosa por score de ranking, respeitando filtros e restricoes.
- Desempate deterministico: score, retorno acumulado, ticker.

## SSOT de setor

- Referencia oficial: `outputs/ssot/setores/ssot_latest/setores_ticker_latest.parquet`
- Formato principal: Parquet.
- Manifest e hashes obrigatorios por snapshot.
