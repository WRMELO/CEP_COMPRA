# Análise Completa: Por que M0 e M1 se comportam diferente nas 3 fases

`[LLM: Claude Opus 4.6]`

---

## 0. Contexto rápido dos mecanismos

| Mecanismo | Ranking de compra | Filtros adicionais |
|-----------|------------------|--------------------|
| **M0** | score_m0 = média de X_t nos últimos 62 dias (simples) | Nenhum filtro extra: compra os top-10 do ranking direto |
| **M1** | Mesmo score_m0, mas com filtros | Liquidez (>=50 dias com volume em 62), 1 classe/empresa, cap 20%/setor, mix 50-80% B3 |
| **M3** | score_m3 = z(score_m0) + z(ret_62) - z(vol_62) | Mesmos filtros de M1 |

Todos usam a **mesma regra de venda** (sell engine CEP_NA_BOLSA v1.3 com upside_extreme).

---

## 1. PERGUNTA 1: Diferença M0 vs M1 nas 3 fases

### 1.1 Retornos por janela (dados reais do backtest)

| Janela | M0 | M1 | M3 |
|--------|----|----|-----|
| **W1** Jul/18 - Jun/21 | **+168%** | **+408%** | **+437%** |
| **W2** Jul/21 - Dez/22 | **-8%** | **-61%** | **-53%** |
| Ago/21 (queda) | +2% | -8% | -8% |
| Mai/22 (queda) | -4% | -12% | -11% |
| **W3** Set/24 - Nov/25 | **-3%** | **+80%** | **-20%** |
| **Total** Jul/18 - Dez/25 | **+106%** | **+314%** | **+139%** |
| **Max drawdown** | -36% | -78% | -68% |

---

### 1.2 W1 (Jul/18 - Jun/21): Por que M1 dispara na frente

**Fato**: M1 retornou +408% vs +168% de M0 (2.4x mais).

**Causa principal: concentração em small/mid caps de alta performance que passaram nos filtros**

| Ticker | Setor | M0 PnL | M1 PnL | Diferença |
|--------|-------|--------|--------|-----------|
| INEP3 | Máq. Industriais | +0.17 | **+0.56** | M1 alocou mais peso |
| RCSL3 | Material Rodoviário | ~0 | **+0.51** | M0 não selecionou; M1 sim |
| FHER3 | Fertilizantes | +0.20 | **+0.48** | M1 alocou mais peso |
| LWSA3 | Programas/Serviços | ~0 | **+0.43** | M0 não selecionou; M1 sim |
| CRPG6 | Químicos | ~0 | **+0.42** | M0 não selecionou; M1 sim |

**Mecanismo causal**:
- M1 filtra por **liquidez mínima**, o que paradoxalmente **exclui os piores ativos ilíquidos** que M0 carrega.
- M1 aplica **1 classe por empresa** (mantém a mais rentável), concentrando em winners.
- O filtro de liquidez + retorno acumulado no lookback como desempate faz M1 naturalmente convergir para **momentum winners** no bull market de 2019-2021.
- M0, sem filtro, dilui capital em ativos ruins: MAPT4(-0.13), GPAR3(-0.13), NUTR3(-0.11) são detratores que M1 evitou.

**Métricas operacionais em W1**:

| Métrica | M0 | M1 |
|---------|----|----|
| Buys | 295 | 493 |
| Sells | 253 | 383 |
| Avg cash ratio | 11.6% | 17.0% |
| Avg posições | 10.0 | 8.9 |

M1 gira **muito mais** (493 buys vs 295). O giro alto em bull market = re-seleção frequente de winners.

---

### 1.3 W2 (Jul/21 - Dez/22): Por que M1 perde sistematicamente

**Fato**: M1 caiu -61% vs -8% de M0 (7.6x pior).

**Causa principal: o mesmo momentum que premiou M1 em W1 agora PENALIZA em bear market**

#### Ago/21 (evento de queda):
| Ticker | Setor | M0 PnL | M1 PnL |
|--------|-------|--------|--------|
| CASH3 | Programas/Serviços | 0 | **-0.14** |
| MGEL4 | Artefatos Ferro/Aço | 0 | **-0.12** |
| PLAS3 | Automóveis | 0 | **-0.10** |
| RSUL4 | Material Rodoviário | **+0.18** | +0.05 |

M1 estava carregando CASH3, MGEL4, PLAS3 (winners de W1 que reverteram). M0 tinha RSUL4 em peso maior (+0.18) que amorteceu.

#### Mai/22 (evento de queda):
| Ticker | Setor | M0 PnL | M1 PnL |
|--------|-------|--------|--------|
| KEPL3 | Máq. Industriais | 0 | **-0.22** |
| CBAV3 | Minerais Metálicos | 0 | **-0.08** |
| PSVM11 | Serv. Apoio | 0 | **-0.05** |
| SOND5 | Engenharia | -0.04 | 0 |

M1 tinha KEPL3 (-0.22 de PnL!) que sozinho explica quase toda a diferença. M0 não o selecionou.

#### Por que M0 protegeu mais que M1?

1. **M0 não gira**: Em W2, M0 fez apenas 65 buys vs 184 de M1. Menos giro = menos chance de entrar em posições ruins.
2. **M0 ficou com posições "travadas" de W1 que eram defensivas**: RSUL4 (+0.41), ENMT4 (+0.11) e CRPG3 (+0.07) foram os maiores contribuintes de M0 em W2. Posições antigas mantidas por inércia.
3. **M1 re-seleciona semanalmente**: O filtro de momentum (ret_lookback como desempate) faz M1 comprar ativos que subiram recentemente. Em bear market, esses são exatamente os que mais caem depois (mean reversion).
4. **Cash ratio conta a história**: Em Mai/22, M1 tinha cash ratio de **35.8%** (vendeu muito por MR_UCL/I_LCL) mas o capital reinvestido foi para KEPL3 e outros que caíram forte.

**Resumo causal W2**: M1 sofre de **"turnover trap"** - o giro alto força re-alocação constante em mercado adverso, comprando ativos recém-estressados que depois caem mais.

**Motivos de venda dominantes em W2**:

| Motivo | M0 | M1 |
|--------|----|----|
| MR_UCL | 29 | 78 |
| I_LCL | 18 | 58 |
| STRESS_AMP | 6 | 16 |

M1 gera **3x mais vendas forçadas** que M0 em W2.

---

### 1.4 W3 (Set/24 - Nov/25): M1 sobe por causa de um único ticker

**Fato**: M1 retornou +80% vs -3% de M0 e -20% de M3.

**Causa ÚNICA: IFCM3**

| Ticker | M0 PnL | M1 PnL | M3 PnL |
|--------|--------|--------|--------|
| IFCM3 | 0 | **+2.36** | 0 |
| AMBP3 | ~0 | +0.27 | +0.36 |
| Resto | -0.06 | -0.68 | -0.58 |

IFCM3 (Programas/Serviços) teve um **salto de preço extremo em 07/Nov/2025** que sozinho contribuiu +2.35 no PnL de M1. Sem IFCM3, M1 em W3 teria retorno **negativo** (~-32%).

**Por que M1 tinha IFCM3 e M0/M3 não?**
- IFCM3 passou nos filtros de liquidez e sector cap de M1.
- O ranking por score_m0 posicionou IFCM3 entre os top-10 de M1 na semana em que foi comprado.
- M0 provavelmente também teria IFCM3 no ranking, mas como M0 não aplica filtro de liquidez, tickers ilíquidos competem com IFCM3 pelo top-10 e o deslocam.
- M3 usa score_m3 = z(score_m0) + z(ret) - z(vol): a penalização por volatilidade de IFCM3 pode ter reduzido sua posição no ranking M3.

**IMPORTANTE**: O resultado de W3 para M1 é dominado por um **evento idiossincrático** (um único ticker), não por superioridade sistemática do mecanismo.

---

## 2. PERGUNTA 2: O fenômeno e o M3 criado

### 2.1 O fenômeno explicado

O fenômeno central é o **dilema momentum vs diversificação**:

| Regime | M0 (simples, sem filtro) | M1 (filtros + momentum implícito) |
|--------|--------------------------|-----------------------------------|
| **Bull market** (W1) | Dilui em ativos ruins, underperforms | Concentra em winners, outperforms |
| **Bear market** (W2) | Inércia protege (não gira), outperforms | Giro alto penaliza (compra losers recentes), underperforms |
| **Lateral/queda** (W3) | Neutro | Depende de eventos idiossincráticos |

**Os filtros de M1 funcionam como um amplificador**:
- Em alta: amplificam ganhos (seleção de winners)
- Em queda: amplificam perdas (re-seleção de ativos estressados)
- O giro de M1 é 2-3x maior que M0 em todas as janelas

**Números-chave do giro**:

| Janela | M0 sells | M1 sells | M0 avg_cash | M1 avg_cash |
|--------|----------|----------|-------------|-------------|
| W1 | 253 | 383 | 11.6% | 17.0% |
| W2 | 53 | 152 | 3.4% | 12.7% |
| W3 | 40 | 119 | 2.3% | 12.3% |

### 2.2 O M3 criado

**Objetivo**: Combinar o melhor de M0 (estabilidade) com o melhor de M1 (seleção de qualidade), sem inventar novas features.

**Fórmula**:

    score_m3 = z(score_m0) + z(ret_lookback_62) - z(vol_lookback_62)

Onde:
- z() = z-score (normalização pelo desvio padrão)
- score_m0: média de X_t nos últimos 62 dias (premia processos centrados/estáveis)
- ret_lookback_62: soma de X_t nos últimos 62 dias (premia retorno acumulado/momentum)
- vol_lookback_62: desvio-padrão de X_t nos últimos 62 dias (PENALIZA volatilidade)

**A diferença crucial vs M1**: M3 penaliza volatilidade (-z(vol)), o que em teoria:
- Evita ativos que subiram muito mas com alta dispersão (potenciais reversões)
- Prefere ativos com retorno consistente e baixa volatilidade

**Mantém os mesmos invariantes de M1**:
- Liquidez >= 50/62 dias
- 1 classe por empresa
- Cap 20% por setor
- Mix 50-80% B3

### 2.3 Resultado real do M3

| Métrica | M0 | M1 | M3 |
|---------|----|----|-----|
| **Retorno total** | +106% | +314% | +139% |
| **Max drawdown** | -36% | -78% | **-68%** |
| **W1 (bull)** | +168% | +408% | **+437%** |
| **W2 (bear)** | -8% | -61% | **-53%** |
| **W3** | -3% | +80%* | -20% |
| **Avg cash ratio** | 6.5% | 13.7% | 12.9% |
| **Avg posições** | 10.8 | 9.0 | 9.5 |

*W3 de M1 inflado por IFCM3 (evento idiossincrático)

**Avaliação honesta do M3**:

1. Em W1 (bull), M3 é **o melhor** (+437%), porque -z(vol) seleciona winners de baixa vol.
2. Em W2 (bear), M3 é **melhor que M1** (-53% vs -61%), mas **muito pior que M0** (-8%).
3. Em W3, M3 é **pior que M0** (-20% vs -3%) e pior que M1 (mas M1 foi salvo por IFCM3).
4. O drawdown máximo de M3 (-68%) é inaceitável comparado a M0 (-36%).

### 2.4 Conclusão: nenhum dos 3 é ideal isoladamente

| Mecanismo | Força | Fraqueza |
|-----------|-------|----------|
| M0 | Proteção em bear market (baixo giro) | Retorno inferior em bull market |
| M1 | Retorno superior em bull market | Drawdown catastrófico em bear (-78%) |
| M3 | Melhor bull que M0/M1, bear melhor que M1 | Ainda sofre drawdown alto (-68%) |

**O problema fundamental**: os filtros de M1/M3 aumentam o giro, e giro alto em mercado de queda é destrutivo. Uma possível solução futura seria **adaptar o giro ao regime do Master** (reduzir re-seleção em RISK_OFF/PRESERVACAO), mas isso seria uma regra nova que está fora do escopo atual.

---

## 3. Referência dos artefatos gerados

Todos em /home/wilson/CEP_COMPRA/outputs/reports/task_017/run_20260212_125255/data/:

| Arquivo | Conteúdo |
|---------|----------|
| ledger_trades_m0.parquet | Todas as ordens BUY/SELL de M0 com data, ticker, notional, motivo |
| ledger_trades_m1.parquet | Idem para M1 |
| ledger_trades_m3.parquet | Idem para M3 |
| mtm_daily_by_ticker_m0.parquet | MTM diário por ticker para M0 |
| mtm_daily_by_ticker_m1.parquet | Idem para M1 |
| mtm_daily_by_ticker_m3.parquet | Idem para M3 |
| decomp_pnl_windows.parquet | PnL decomposto por ticker/setor nas janelas W1/W2/W3 |
| decomp_pnl_event_months.parquet | PnL decomposto nos eventos Ago/21, Mai/22, Nov/25 |
| holdings_weekly_m0.parquet | Holdings semanais M0 |
| holdings_weekly_m1.parquet | Holdings semanais M1 |
| holdings_weekly_m3.parquet | Holdings semanais M3 |
| master_state_daily.parquet | Estado do Master (RISK_ON/OFF/PRESERVACAO) por dia |
| daily_portfolio_m0.parquet | Equity, cash, n_positions diários M0 |
| daily_portfolio_m1.parquet | Idem M1 |
| daily_portfolio_m3.parquet | Idem M3 |
