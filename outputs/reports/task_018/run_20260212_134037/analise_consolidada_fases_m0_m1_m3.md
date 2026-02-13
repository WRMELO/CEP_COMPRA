# TASK 018 — Análise Consolidada: Diferenças M0 vs M1 vs M3 por Fase

`[LLM: Claude Opus 4.6]` | run_id: run_20260212_134037
Gerado em: 2026-02-12T13:40:37.111069+00:00

---

## 0. Mecanismos comparados

| Mecanismo | Ranking | Filtros | Gating de venda |
| --- | --- | --- | --- |
| M0 | score_m0 = mean(X_t, 62d) | Nenhum | CEP v1.3 (upside_extreme + stress_amp) |
| M1 | score_m0 (desempate: ret_lookback) | Liquidez>=50/62, 1 classe/empresa, cap 20%/setor, B3 50-80% | Idem |
| M3 | z(score_m0)+z(ret_62)-z(vol_62) | Mesmos de M1 | Idem |

## 1. Equity em datas-chave

| date | M0_equity | M0_cash% | M0_n_pos | M0_dd | M1_equity | M1_cash% | M1_n_pos | M1_dd | M3_equity | M3_cash% | M3_n_pos | M3_dd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-07-02 | 1.0000 | 0.0% | 10 | 0.0% | 1.0000 | 0.0% | 10 | 0.0% | 1.0000 | 0.0% | 10 | 0.0% |
| 2021-06-30 | 2.6810 | 14.4% | 9 | -10.6% | 5.0802 | 46.0% | 7 | -10.9% | 5.3696 | 49.2% | 5 | -12.3% |
| 2021-08-31 | 3.0190 | 0.0% | 11 | -3.1% | 4.4032 | 10.4% | 11 | -22.8% | 5.0586 | 10.2% | 11 | -17.4% |
| 2022-05-31 | 2.4586 | 0.0% | 11 | -21.1% | 2.6006 | 0.0% | 10 | -54.4% | 3.1442 | 9.8% | 9 | -48.7% |
| 2022-12-30 | 2.4385 | 0.0% | 11 | -21.7% | 2.0044 | 0.0% | 9 | -64.8% | 2.5075 | 19.6% | 10 | -59.1% |
| 2024-09-02 | 2.2481 | 0.0% | 14 | -27.8% | 2.2977 | 0.0% | 10 | -59.7% | 2.7199 | 0.0% | 12 | -55.6% |
| 2025-11-07 | 2.1363 | 4.3% | 10 | -31.4% | 3.6237 | 0.0% | 8 | -36.4% | 2.0187 | 0.0% | 9 | -67.0% |
| 2025-11-28 | 2.1833 | 0.0% | 11 | -29.9% | 4.1389 | 22.9% | 10 | -27.4% | 2.1764 | 9.3% | 9 | -64.5% |
| 2025-12-30 | 2.0556 | 10.2% | 10 | -34.0% | 4.1382 | 10.4% | 9 | -27.4% | 2.3854 | 0.0% | 10 | -61.1% |

## 2. Retornos por janela

| Janela | Período | M0 | M1 | M3 |
| --- | --- | --- | --- | --- |
| W1 | 2018-07-02..2021-06-30 | +168.10% | +408.02% | +436.96% |
| W2 | 2021-07-01..2022-12-30 | -8.08% | -60.74% | -53.27% |
| W3 | 2024-09-02..2025-11-28 | -2.88% | +80.13% | -19.98% |
| queda_agosto_2021 | 2021-08-01..2021-08-31 | +2.07% | -8.48% | -7.83% |
| queda_maio_2022 | 2022-05-01..2022-05-31 | -3.74% | -12.02% | -11.24% |
| salto_2025_11_07 | 2025-11-07..2025-11-07 | +0.00% | +0.00% | +0.00% |

## 3. Métricas consolidadas (período completo)

| Métrica | M0 | M1 | M3 |
| --- | --- | --- | --- |
| Equity final | 2.0556 | 4.1382 | 2.3854 |
| Retorno total | 105.56% | 313.82% | 138.54% |
| Equity pico | 3.1145 | 5.7016 | 6.1244 |
| Max drawdown | -35.78% | -78.11% | -68.03% |
| Avg cash ratio | 6.50% | 13.69% | 12.91% |
| Avg n_positions | 10.8 | 9.0 | 9.5 |

## 4. W1 (Jul/18 — Jun/21): M1 dispara — por quê?

### 4.1 Top 10 contribuintes por mecanismo

**M0**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| A1LG34 | Tecnologia | BDR | 0.2840 |
| FHER3 | Fertilizantes e Defensivos | B3 | 0.1952 |
| INEP3 | Máq. e Equip. Industriais | B3 | 0.1660 |
| NORD3 | Exploração de Imóveis | B3 | 0.1541 |
| TXRX3 | Fios e Tecidos | B3 | 0.1374 |
| INEP4 | Máq. e Equip. Industriais | B3 | 0.1226 |
| ENMT4 | Energia Elétrica | B3 | 0.1219 |
| BSLI3 | Bancos | B3 | 0.1186 |
| PMAM3 | Artefatos de Cobre | B3 | 0.1124 |
| U1RI34 | Locadora | BDR | 0.0965 |

**M1**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| INEP3 | Máq. e Equip. Industriais | B3 | 0.5584 |
| RCSL3 | Material Rodoviário | B3 | 0.5069 |
| FHER3 | Fertilizantes e Defensivos | B3 | 0.4804 |
| LWSA3 | Programas e Serviços | B3 | 0.4304 |
| CRPG6 | Químicos Diversos | B3 | 0.4220 |
| LOGN3 | Transporte Hidroviário | B3 | 0.2189 |
| TASA3 | Armas e Munições | B3 | 0.1867 |
| PPLA11 | Financeiro e Outros/Serviços Financeiros/Gestão de Recursos e Investimentos | BDR | 0.1610 |
| ETER3 | Produtos para Construção | B3 | 0.1525 |
| OSXB3 | Equipamentos e Serviços | B3 | 0.1330 |

**M3**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| FHER3 | Fertilizantes e Defensivos | B3 | 0.5870 |
| INEP3 | Máq. e Equip. Industriais | B3 | 0.5567 |
| LWSA3 | Programas e Serviços | B3 | 0.4811 |
| CRPG6 | Químicos Diversos | B3 | 0.3811 |
| RCSL3 | Material Rodoviário | B3 | 0.3026 |
| TASA3 | Armas e Munições | B3 | 0.2569 |
| BRKM5 | Petroquímicos | B3 | 0.2385 |
| PPLA11 | Financeiro e Outros/Serviços Financeiros/Gestão de Recursos e Investimentos | BDR | 0.1567 |
| ETER3 | Produtos para Construção | B3 | 0.1537 |
| CSNA3 | Siderurgia | B3 | 0.1507 |

### 4.2 Top 10 detratores por mecanismo

**M0**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| MAPT4 | Outros | B3 | -0.1335 |
| GPAR3 | Energia Elétrica | B3 | -0.1252 |
| NUTR3 | Fertilizantes e Defensivos | B3 | -0.1108 |
| DASA3 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 | -0.1029 |
| CTSA4 | Fios e Tecidos | B3 | -0.0889 |
| HETA4 | Utensílios Domésticos | B3 | -0.0631 |
| JOPA4 | Alimentos Diversos | B3 | -0.0610 |
| MERC4 | Soc. Crédito e Financiamento | B3 | -0.0575 |
| BRKM6 | Petroquímicos | B3 | -0.0557 |
| MWET3 | Material Rodoviário | B3 | -0.0525 |

**M1**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| DASA3 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 | -0.2692 |
| NGRD3 | Programas e Serviços | B3 | -0.1029 |
| VIVR3 | Incorporações | B3 | -0.0906 |
| MGLU3 | Eletrodomésticos | B3 | -0.0891 |
| GSHP3 | Exploração de Imóveis | B3 | -0.0705 |
| BPAC3 | Bancos | B3 | -0.0552 |
| HBOR3 | Incorporações | B3 | -0.0550 |
| NORD3 | Exploração de Imóveis | B3 | -0.0471 |
| F1AN34 | Energia | BDR | -0.0467 |
| AFLT3 | Energia Elétrica | B3 | -0.0463 |

**M3**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| DASA3 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 | -0.2697 |
| MGLU3 | Eletrodomésticos | B3 | -0.1215 |
| PSVM11 | Serviços de Apoio e Armazenagem | B3 | -0.1080 |
| GUAR3 | Tecidos. Vestuário e Calçados | B3 | -0.0972 |
| HBOR3 | Incorporações | B3 | -0.0928 |
| VIVR3 | Incorporações | B3 | -0.0639 |
| TELB3 | Telecomunicações | B3 | -0.0601 |
| COPH34 | Petróleo | BDR | -0.0557 |
| F1AN34 | Energia | BDR | -0.0515 |
| OSXB3 | Equipamentos e Serviços | B3 | -0.0500 |

### 4.3 Operacional W1: compras/vendas/motivos

| Mecanismo | Buys | Notional BUY | Sells | Notional SELL | Motivos SELL (top) |
| --- | --- | --- | --- | --- | --- |
| M0 | 295 | 32.39 | 253 | 31.78 | MR_UCL:85, I_LCL:80, MASTER_PRESERVACAO_TOTAL:53 |
| M1 | 493 | 65.66 | 383 | 67.00 | MR_UCL:169, I_LCL:92, STRESS_AMP:73 |
| M3 | 489 | 77.02 | 392 | 78.67 | MR_UCL:175, I_LCL:89, STRESS_AMP:79 |

### 4.4 Setores mais relevantes em W1

**M0** — Top 5 setores:

| sector | pnl_brl |
| --- | --- |
| Tecnologia | 0.2899 |
| Máq. e Equip. Industriais | 0.2893 |
| Energia Elétrica | 0.1961 |
| Programas e Serviços | 0.1714 |
| Exploração de Imóveis | 0.1323 |

**M0** — Piores 5 setores:

| sector | pnl_brl |
| --- | --- |
| Petroquímicos | -0.0557 |
| Utensílios Domésticos | -0.0631 |
| Alimentos Diversos | -0.0665 |
| Serv.Méd.Hospit..Análises e Diagnósticos | -0.1018 |
| Outros | -0.1335 |

**M1** — Top 5 setores:

| sector | pnl_brl |
| --- | --- |
| Máq. e Equip. Industriais | 0.5645 |
| Material Rodoviário | 0.5206 |
| Fertilizantes e Defensivos | 0.4804 |
| Programas e Serviços | 0.4435 |
| Químicos Diversos | 0.4220 |

**M1** — Piores 5 setores:

| sector | pnl_brl |
| --- | --- |
| Siderurgia | -0.0506 |
| Incorporações | -0.0574 |
| Eletrodomésticos | -0.0892 |
| Exploração de Imóveis | -0.1176 |
| Serv.Méd.Hospit..Análises e Diagnósticos | -0.2423 |

**M3** — Top 5 setores:

| sector | pnl_brl |
| --- | --- |
| Programas e Serviços | 0.5922 |
| Fertilizantes e Defensivos | 0.5870 |
| Máq. e Equip. Industriais | 0.5731 |
| Químicos Diversos | 0.3794 |
| Material Rodoviário | 0.3230 |

**M3** — Piores 5 setores:

| sector | pnl_brl |
| --- | --- |
| Energia | -0.0704 |
| Petróleo | -0.0831 |
| Tecidos. Vestuário e Calçados | -0.0972 |
| Serviços de Apoio e Armazenagem | -0.1080 |
| Serv.Méd.Hospit..Análises e Diagnósticos | -0.2136 |

### 4.5 Asset class em W1

**M0**: B3=1.1320, BDR=0.5490
**M1**: B3=3.4787, BDR=0.6015
**M3**: B3=3.8476, BDR=0.5220

### 4.6 Explicação causal W1

1. **M1 concentra em momentum winners**: Os filtros de liquidez eliminam ativos ilíquidos/ruins que M0 carrega (MAPT4, GPAR3, NUTR3 = -0.37 combinados em M0).
2. **1-classe-por-empresa retém a classe mais rentável**: Isso concentra peso nos winners de cada grupo empresarial.
3. **Giro mais alto = re-seleção de winners**: M1 fez 493 buys vs 295 de M0. Em bull market, re-selecionar semanalmente os top-ranked é vantajoso.
4. **M0 dilui**: Sem filtro, M0 distribui capital em ativos que passam no gating mas têm retorno ruim.
5. **M3 supera M1 em W1 (+437% vs +408%)**: A penalização de volatilidade (-z(vol)) prefere winners de baixa dispersão, que são os que sobem de forma mais sustentável.

## 5. W2 (Jul/21 — Dez/22): M1 perde sistematicamente — por quê?

### 5.1 Top 10 contribuintes por mecanismo

**M0**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| RSUL4 | Material Rodoviário | B3 | 0.4122 |
| ENMT4 | Energia Elétrica | B3 | 0.1065 |
| CRPG3 | Químicos Diversos | B3 | 0.0726 |
| B2HI34 | Tecnologia | BDR | 0.0616 |
| ESTR4 | Brinquedos e Jogos | B3 | 0.0587 |
| EQPA5 | Energia Elétrica | B3 | 0.0415 |
| A2SO34 | Entretenimento | BDR | 0.0411 |
| HAGA3 | Artefatos de Ferro e Aço | B3 | 0.0321 |
| PTNT3 | Fios e Tecidos | B3 | 0.0278 |
| EQPA7 | Energia Elétrica | B3 | 0.0218 |

**M1**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| HAGA3 | Artefatos de Ferro e Aço | B3 | 0.0816 |
| GFSA3 | Incorporações | B3 | 0.0727 |
| RSUL4 | Material Rodoviário | B3 | 0.0459 |
| EMAE4 | Energia Elétrica | B3 | 0.0412 |
| TUPY3 | Material Rodoviário | B3 | 0.0402 |
| PLPL3 | Incorporações | B3 | 0.0328 |
| B1NT34 | Farmacêutico | BDR | 0.0317 |
| HETA4 | Utensílios Domésticos | B3 | 0.0314 |
| NORD3 | Exploração de Imóveis | B3 | 0.0280 |
| AGXY3 | Agricultura | B3 | 0.0249 |

**M3**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| M1RN34 | Farmacêutico | BDR | 0.4154 |
| HAGA3 | Artefatos de Ferro e Aço | B3 | 0.2424 |
| MOSC34 | Materiais básicos | BDR | 0.0853 |
| LILY34 | Farmacêutico | BDR | 0.0555 |
| EMAE4 | Energia Elétrica | B3 | 0.0551 |
| RSUL4 | Material Rodoviário | B3 | 0.0422 |
| B1NT34 | Farmacêutico | BDR | 0.0362 |
| BEEF3 | Carnes e Derivados | B3 | 0.0260 |
| CMIG3 | Energia Elétrica | B3 | 0.0250 |
| ENMT3 | Energia Elétrica | B3 | 0.0236 |

### 5.2 Top 10 detratores por mecanismo

**M0**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| GPAR3 | Energia Elétrica | B3 | -0.1339 |
| MWET4 | Material Rodoviário | B3 | -0.1099 |
| COCE3 | Energia Elétrica | B3 | -0.1079 |
| PATI4 | Artefatos de Ferro e Aço | B3 | -0.0969 |
| SOND5 | Engenharia Consultiva | B3 | -0.0963 |
| SNSY5 | Materiais Diversos | B3 | -0.0856 |
| HBTS5 | Exploração de Imóveis | B3 | -0.0687 |
| USIM6 | Siderurgia | B3 | -0.0683 |
| SOND6 | Engenharia Consultiva | B3 | -0.0528 |
| CLOV34 | UNKNOWN | BDR | -0.0472 |

**M1**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| FHER3 | Fertilizantes e Defensivos | B3 | -0.2126 |
| CASH3 | Programas e Serviços | B3 | -0.2110 |
| MWET4 | Material Rodoviário | B3 | -0.2035 |
| D1OC34 | Tecnologia | BDR | -0.2008 |
| KEPL3 | Máq. e Equip. Industriais | B3 | -0.1990 |
| IFCM3 | Programas e Serviços | B3 | -0.1518 |
| ALLD3 | Eletrodomésticos | B3 | -0.1373 |
| BRAV3 | Exploração. Refino e Distribuição | B3 | -0.1358 |
| SYNE3 | Exploração de Imóveis | B3 | -0.1267 |
| CBAV3 | Minerais Metálicos | B3 | -0.1249 |

**M3**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| VAMO3 | Aluguel de carros | B3 | -0.4191 |
| KEPL3 | Máq. e Equip. Industriais | B3 | -0.2405 |
| MWET4 | Material Rodoviário | B3 | -0.2122 |
| SYNE3 | Exploração de Imóveis | B3 | -0.2096 |
| CASH3 | Programas e Serviços | B3 | -0.1678 |
| FHER3 | Fertilizantes e Defensivos | B3 | -0.1598 |
| BRAV3 | Exploração. Refino e Distribuição | B3 | -0.1522 |
| CBAV3 | Minerais Metálicos | B3 | -0.1481 |
| ALLD3 | Eletrodomésticos | B3 | -0.1480 |
| D1OC34 | Tecnologia | BDR | -0.1293 |

### 5.3 Operacional W2

| Mecanismo | Buys | Not. BUY | Sells | Not. SELL | Motivos SELL |
| --- | --- | --- | --- | --- | --- |
| M0 | 65 | 11.37 | 53 | 10.98 | MR_UCL:29, I_LCL:18, STRESS_AMP:6 |
| M1 | 184 | 43.25 | 152 | 40.91 | MR_UCL:78, I_LCL:58, STRESS_AMP:16 |
| M3 | 191 | 52.24 | 158 | 50.09 | MR_UCL:80, I_LCL:58, STRESS_AMP:20 |

### 5.4 Meses críticos em W2

**M0**:

| month | avg_cash_ratio | avg_n_pos | min_equity | max_equity | end_equity | max_dd |
| --- | --- | --- | --- | --- | --- | --- |
| 2021-07 | 0.0741 | 9.8095 | 2.6377 | 2.9576 | 2.9576 | -0.1201 |
| 2021-08 | 0.0897 | 9.8182 | 2.9478 | 3.1145 | 3.0190 | -0.0348 |
| 2021-09 | 0.0459 | 10.4286 | 2.8580 | 3.0665 | 2.8702 | -0.0824 |
| 2022-04 | 0.0329 | 10.6842 | 2.5542 | 2.7661 | 2.5542 | -0.1799 |
| 2022-05 | 0.0751 | 10.0000 | 2.4579 | 2.5446 | 2.4586 | -0.2108 |
| 2022-06 | 0.0058 | 10.9048 | 2.3834 | 2.4648 | 2.4648 | -0.2347 |
| 2022-12 | 0.0000 | 11.0000 | 2.3956 | 2.4392 | 2.4385 | -0.2308 |

**M1**:

| month | avg_cash_ratio | avg_n_pos | min_equity | max_equity | end_equity | max_dd |
| --- | --- | --- | --- | --- | --- | --- |
| 2021-07 | 0.1890 | 9.9524 | 4.8114 | 5.3160 | 4.8114 | -0.1561 |
| 2021-08 | 0.1480 | 10.4545 | 4.3558 | 4.9468 | 4.4032 | -0.2360 |
| 2021-09 | 0.1760 | 9.6667 | 4.0307 | 4.3498 | 4.2142 | -0.2931 |
| 2022-04 | 0.0868 | 9.0526 | 2.9560 | 3.2435 | 2.9560 | -0.4816 |
| 2022-05 | 0.3580 | 6.2273 | 2.5610 | 2.9284 | 2.6006 | -0.5508 |
| 2022-06 | 0.0826 | 8.6190 | 2.3582 | 2.6577 | 2.3582 | -0.5864 |
| 2022-12 | 0.0592 | 8.2381 | 1.7502 | 2.0044 | 2.0044 | -0.6930 |

**M3**:

| month | avg_cash_ratio | avg_n_pos | min_equity | max_equity | end_equity | max_dd |
| --- | --- | --- | --- | --- | --- | --- |
| 2021-07 | 0.1654 | 8.4286 | 5.3273 | 5.6934 | 5.4883 | -0.1302 |
| 2021-08 | 0.1515 | 9.0455 | 4.9399 | 5.8982 | 5.0586 | -0.1934 |
| 2021-09 | 0.1711 | 9.6190 | 4.5340 | 5.0101 | 4.6722 | -0.2597 |
| 2022-04 | 0.0419 | 10.3684 | 3.5423 | 3.8726 | 3.5423 | -0.4216 |
| 2022-05 | 0.3503 | 6.2273 | 3.0309 | 3.4922 | 3.1442 | -0.5051 |
| 2022-06 | 0.1011 | 8.6667 | 2.8643 | 3.1512 | 2.8654 | -0.5323 |
| 2022-12 | 0.1002 | 10.0952 | 2.3705 | 2.5075 | 2.5075 | -0.6129 |

### 5.5 Evento: Queda Agosto/2021

**M0 top contribuintes Ago/21:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| RSUL4 | Material Rodoviário | B3 | 0.1771 |
| ENMT4 | Energia Elétrica | B3 | 0.0583 |
| PATI3 | Artefatos de Ferro e Aço | B3 | 0.0268 |
| ESTR4 | Brinquedos e Jogos | B3 | 0.0156 |
| PATI4 | Artefatos de Ferro e Aço | B3 | 0.0053 |

**M0 top detratores Ago/21:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| CRPG3 | Químicos Diversos | B3 | -0.0629 |
| MWET4 | Material Rodoviário | B3 | -0.0420 |
| COCE3 | Energia Elétrica | B3 | -0.0346 |
| JOPA4 | Alimentos Diversos | B3 | -0.0243 |
| EQPA5 | Energia Elétrica | B3 | -0.0198 |

**M1 top contribuintes Ago/21:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| RSUL4 | Material Rodoviário | B3 | 0.0459 |
| MNDL3 | Acessórios | B3 | 0.0359 |
| B1NT34 | Farmacêutico | BDR | 0.0317 |
| ALLD3 | Eletrodomésticos | B3 | 0.0299 |
| PSVM11 | Serviços de Apoio e Armazenagem | B3 | 0.0169 |

**M1 top detratores Ago/21:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| CASH3 | Programas e Serviços | B3 | -0.1394 |
| MGEL4 | Artefatos de Ferro e Aço | B3 | -0.1209 |
| PLAS3 | Automóveis e Motocicletas | B3 | -0.1019 |
| MWET4 | Material Rodoviário | B3 | -0.0773 |
| CRPG6 | Químicos Diversos | B3 | -0.0630 |

**M3 top contribuintes Ago/21:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| M1RN34 | Farmacêutico | BDR | 0.2056 |
| BRKM5 | Petroquímicos | B3 | 0.0870 |
| RSUL4 | Material Rodoviário | B3 | 0.0422 |
| B1NT34 | Farmacêutico | BDR | 0.0362 |
| SGPS3 | Fios e Tecidos | B3 | 0.0330 |

**M3 top detratores Ago/21:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| VAMO3 | Aluguel de carros | B3 | -0.5048 |
| MGEL4 | Artefatos de Ferro e Aço | B3 | -0.1191 |
| CASH3 | Programas e Serviços | B3 | -0.0860 |
| MWET4 | Material Rodoviário | B3 | -0.0846 |
| CRPG6 | Químicos Diversos | B3 | -0.0603 |

### 5.6 Evento: Queda Maio/2022

**M0 top contribuintes Mai/22:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| ENMT4 | Energia Elétrica | B3 | 0.0005 |
| I2NG34 | Produtos de consumo não duráveis | BDR | -0.0001 |
| OXYP34 | Energia | BDR | -0.0010 |
| GPAR3 | Energia Elétrica | B3 | -0.0022 |
| LAND3 | Agricultura | B3 | -0.0024 |

**M0 top detratores Mai/22:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| SOND5 | Engenharia Consultiva | B3 | -0.0388 |
| SOND6 | Engenharia Consultiva | B3 | -0.0242 |
| LPSB3 | Intermediação Imobiliária | B3 | -0.0207 |
| EQPA5 | Energia Elétrica | B3 | -0.0068 |
| LAND3 | Agricultura | B3 | -0.0024 |

**M1 top contribuintes Mai/22:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| BEEF3 | Carnes e Derivados | B3 | 0.0308 |
| DOTZ3 | Programas de Fidelização | B3 | 0.0140 |
| OXYP34 | Energia | BDR | 0.0097 |
| RECV3 | Exploração. Refino e Distribuição | B3 | 0.0095 |
| VULC3 | Calçados | B3 | 0.0074 |

**M1 top detratores Mai/22:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| KEPL3 | Máq. e Equip. Industriais | B3 | -0.2222 |
| CBAV3 | Minerais Metálicos | B3 | -0.0755 |
| PSVM11 | Serviços de Apoio e Armazenagem | B3 | -0.0506 |
| LPSB3 | Intermediação Imobiliária | B3 | -0.0421 |
| MOSC34 | Materiais básicos | BDR | -0.0284 |

**M3 top contribuintes Mai/22:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| BEEF3 | Carnes e Derivados | B3 | 0.0369 |
| OXYP34 | Energia | BDR | 0.0150 |
| RECV3 | Exploração. Refino e Distribuição | B3 | 0.0113 |
| VULC3 | Calçados | B3 | 0.0088 |
| SBSP3 | Água e Saneamento | B3 | 0.0081 |

**M3 top detratores Mai/22:**

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| KEPL3 | Máq. e Equip. Industriais | B3 | -0.2735 |
| CBAV3 | Minerais Metálicos | B3 | -0.0894 |
| LPSB3 | Intermediação Imobiliária | B3 | -0.0529 |
| KHCB34 | Consumo não Cíclico/Alimentos/Alimentos Diversos | BDR | -0.0389 |
| VIVT3 | Telecomunicações | B3 | -0.0162 |

### 5.7 Holdings snapshot pré-queda Ago/21 (último rebalanceamento antes)

**M0 holdings em ~01/Ago/21:**

| ticker | value_brl | weight | sector | asset_class |
| --- | --- | --- | --- | --- |
| CRPG3 | 0.4194 | 0.1450 | Químicos Diversos | B3 |
| RSUL4 | 0.3718 | 0.1286 | Material Rodoviário | B3 |
| ENMT4 | 0.3552 | 0.1228 | Energia Elétrica | B3 |
| PATI4 | 0.2891 | 0.1000 | Artefatos de Ferro e Aço | B3 |
| PATI3 | 0.2883 | 0.0997 | Artefatos de Ferro e Aço | B3 |
| COCE3 | 0.2773 | 0.0959 | Energia Elétrica | B3 |
| USIM6 | 0.2706 | 0.0936 | Siderurgia | B3 |
| MWET4 | 0.2608 | 0.0902 | Material Rodoviário | B3 |
| EQPA5 | 0.1947 | 0.0674 | Energia Elétrica | B3 |
| JOPA4 | 0.1641 | 0.0568 | Alimentos Diversos | B3 |

**M1 holdings em ~01/Ago/21:**

| ticker | value_brl | weight | sector | asset_class |
| --- | --- | --- | --- | --- |
| CRPG6 | 0.8949 | 0.1724 | Químicos Diversos | B3 |
| PLAS3 | 0.5499 | 0.1059 | Automóveis e Motocicletas | B3 |
| MNDL3 | 0.5249 | 0.1011 | Acessórios | B3 |
| MWET4 | 0.5191 | 0.1000 | Material Rodoviário | B3 |
| MGEL4 | 0.5191 | 0.1000 | Artefatos de Ferro e Aço | B3 |
| ALLD3 | 0.4254 | 0.0820 | Eletrodomésticos | B3 |
| BRAV3 | 0.3849 | 0.0742 | Exploração. Refino e Distribuição | B3 |
| CASH3 | 0.3791 | 0.0730 | Programas e Serviços | B3 |
| NGRD3 | 0.3446 | 0.0664 | Programas e Serviços | B3 |
| PSVM11 | 0.3284 | 0.0633 | Serviços de Apoio e Armazenagem | B3 |

**M3 holdings em ~01/Ago/21:**

| ticker | value_brl | weight | sector | asset_class |
| --- | --- | --- | --- | --- |
| CRPG6 | 0.8831 | 0.1585 | Químicos Diversos | B3 |
| M1RN34 | 0.6807 | 0.1222 | Farmacêutico | BDR |
| BRKM5 | 0.6360 | 0.1141 | Petroquímicos | B3 |
| VAMO3 | 0.6010 | 0.1079 | Aluguel de carros | B3 |
| MGEL4 | 0.5282 | 0.0948 | Artefatos de Ferro e Aço | B3 |
| AMBP3 | 0.5195 | 0.0932 | Água e Saneamento | B3 |
| MWET4 | 0.4826 | 0.0866 | Material Rodoviário | B3 |
| ALLD3 | 0.4484 | 0.0805 | Eletrodomésticos | B3 |
| SGPS3 | 0.4128 | 0.0741 | Fios e Tecidos | B3 |
| BRAV3 | 0.3799 | 0.0682 | Exploração. Refino e Distribuição | B3 |

### 5.8 Holdings snapshot pré-queda Mai/22

**M0 holdings em ~01/Mai/22:**

| ticker | value_brl | weight | sector | asset_class |
| --- | --- | --- | --- | --- |
| ENMT4 | 0.4400 | 0.1635 | Energia Elétrica | B3 |
| CEBR5 | 0.3503 | 0.1302 | Energia Elétrica | B3 |
| EQPA5 | 0.3154 | 0.1172 | Energia Elétrica | B3 |
| SOND6 | 0.2842 | 0.1056 | Engenharia Consultiva | B3 |
| SOND5 | 0.2691 | 0.1000 | Engenharia Consultiva | B3 |
| PATI4 | 0.2674 | 0.0994 | Artefatos de Ferro e Aço | B3 |
| COCE3 | 0.2169 | 0.0806 | Energia Elétrica | B3 |
| LPSB3 | 0.1706 | 0.0634 | Intermediação Imobiliária | B3 |
| GPAR3 | 0.1358 | 0.0505 | Energia Elétrica | B3 |
| EQPA7 | 0.1315 | 0.0489 | Energia Elétrica | B3 |

**M1 holdings em ~01/Mai/22:**

| ticker | value_brl | weight | sector | asset_class |
| --- | --- | --- | --- | --- |
| ENMT3 | 0.4537 | 0.1453 | Energia Elétrica | B3 |
| CEBR5 | 0.3853 | 0.1234 | Energia Elétrica | B3 |
| LPSB3 | 0.3469 | 0.1111 | Intermediação Imobiliária | B3 |
| UNIP6 | 0.3185 | 0.1020 | Químicos Diversos | B3 |
| BMEB4 | 0.3009 | 0.0964 | Bancos | B3 |
| KEPL3 | 0.2981 | 0.0955 | Máq. e Equip. Industriais | B3 |
| CBAV3 | 0.2924 | 0.0936 | Minerais Metálicos | B3 |
| PSVM11 | 0.2906 | 0.0931 | Serviços de Apoio e Armazenagem | B3 |
| MDNE3 | 0.2841 | 0.0910 | Incorporações | B3 |
| TECN3 | 0.1517 | 0.0486 | Acessórios | B3 |

**M3 holdings em ~01/Mai/22:**

| ticker | value_brl | weight | sector | asset_class |
| --- | --- | --- | --- | --- |
| ENMT3 | 0.5233 | 0.1387 | Energia Elétrica | B3 |
| CEBR5 | 0.4997 | 0.1324 | Energia Elétrica | B3 |
| LPSB3 | 0.4363 | 0.1157 | Intermediação Imobiliária | B3 |
| HYPE3 | 0.3773 | 0.1000 | Medicamentos e Outros Produtos | B3 |
| KEPL3 | 0.3670 | 0.0973 | Máq. e Equip. Industriais | B3 |
| BMEB4 | 0.3579 | 0.0949 | Bancos | B3 |
| CBAV3 | 0.3462 | 0.0918 | Minerais Metálicos | B3 |
| SBSP3 | 0.3185 | 0.0844 | Água e Saneamento | B3 |
| VIVT3 | 0.2669 | 0.0708 | Telecomunicações | B3 |
| UNIP6 | 0.1445 | 0.0383 | Químicos Diversos | B3 |

### 5.9 Master state em W2

| Estado | Dias |
| --- | --- |
| RISK_ON | 306 |
| RISK_OFF | 69 |

### 5.10 Explicação causal W2

1. **Turnover trap**: M1 fez 184 buys / 152 sells vs M0 com 65/53. Giro 3x maior em bear market = recompra constante de ativos que acabaram de cair.
2. **M0 protegeu por inércia**: RSUL4 (+0.41), ENMT4 (+0.11) são posições que M0 **manteve** de W1 e que continuaram rendendo em W2. M0 quase não rebalanceou.
3. **M1 carregou detratores concentrados**: FHER3(-0.21), CASH3(-0.21), MWET4(-0.20), KEPL3(-0.20) são 4 tickers que sozinhos custaram -0.83 a M1. M0 não tinha nenhum desses em peso relevante.
4. **Cash ratio de M1 em Mai/22 = 35.8%**: Muitas vendas forçadas (I_LCL/MR_UCL) geraram caixa, mas o capital foi reinvestido em KEPL3 e outros que caíram forte.
5. **M3 sofre menos que M1 (-53% vs -61%)**: A penalização de vol evitou parcialmente os ativos mais voláteis, mas o giro alto (158 sells) ainda foi destrutivo.
6. **Ponto-chave: M0 com 65 buys em 18 meses = ~3.6 buys/mês. M1 com 184 buys = ~10.2 buys/mês.** O mecanismo simples trava o portfólio e evita rotação pró-cíclica.

## 6. W3 (Set/24 — Nov/25): M1 sobe por evento idiossincrático

### 6.1 Top 10 contribuintes

**M0**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| CCTY3 | Incorporações | B3 | 0.2076 |
| D2OC34 | Tecnologia | BDR | 0.0739 |
| E2XE34 | Pacífico Asiático, Ex Japão | BDR | 0.0716 |
| G2WR34 | Tecnologia | BDR | 0.0707 |
| ITUB3 | Bancos | B3 | 0.0620 |
| ITUB4 | Bancos | B3 | 0.0478 |
| TKNO4 | Artefatos de Ferro e Aço | B3 | 0.0452 |
| CBEE3 | Energia Elétrica | B3 | 0.0392 |
| AMBP3 | Água e Saneamento | B3 | 0.0264 |
| BKCH39 | Não Classificados | B3 | 0.0191 |

**M1**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| IFCM3 | Programas e Serviços | B3 | 2.3551 |
| AMBP3 | Água e Saneamento | B3 | 0.2736 |
| A2RR34 | Pacífico Asiático, Ex Japão | BDR | 0.1437 |
| RCSL4 | Material Rodoviário | B3 | 0.1393 |
| DASA3 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 | 0.1282 |
| CASH3 | Programas e Serviços | B3 | 0.0635 |
| ITUB3 | Bancos | B3 | 0.0607 |
| AURA33 | Materiais Básicos/Mineração/Minerais Metálicos | BDR | 0.0378 |
| K2CG34 | Tecnologia | BDR | 0.0371 |
| ITUB4 | Bancos | B3 | 0.0308 |

**M3**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| AMBP3 | Água e Saneamento | B3 | 0.3625 |
| ITUB3 | Bancos | B3 | 0.0871 |
| CASH3 | Programas e Serviços | B3 | 0.0603 |
| DASA3 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 | 0.0541 |
| ITUB4 | Bancos | B3 | 0.0514 |
| RCSL4 | Material Rodoviário | B3 | 0.0390 |
| TTEN3 | Agricultura | B3 | 0.0320 |
| AGXY3 | Agricultura | B3 | 0.0275 |
| PSSA3 | Seguradoras | B3 | 0.0240 |
| K2CG34 | Tecnologia | BDR | 0.0223 |

### 6.2 Top 10 detratores

**M0**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| B2RK34 | Equipamentos Médicos | BDR | -0.0905 |
| M2KS34 | Tecnologia | BDR | -0.0889 |
| GFSA3 | Incorporações | B3 | -0.0827 |
| AZTE3 | Exploração. Refino e Distribuição | B3 | -0.0740 |
| CALI3 | Incorporações | B3 | -0.0677 |
| P2CY34 | Consultoria | BDR | -0.0656 |
| P2ST34 | Tecnologia | BDR | -0.0594 |
| JOPA4 | Alimentos Diversos | B3 | -0.0361 |
| CEEB5 | Energia Elétrica | B3 | -0.0347 |
| C2RN34 | Tecnologia | BDR | -0.0304 |

**M1**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| FICT3 | Carnes e Derivados | B3 | -0.1647 |
| ENMT3 | Energia Elétrica | B3 | -0.1408 |
| GFSA3 | Incorporações | B3 | -0.1170 |
| SBFG3 | Produtos Diversos | B3 | -0.0791 |
| CTSA4 | Fios e Tecidos | B3 | -0.0783 |
| MNPR3 | Carnes e Derivados | B3 | -0.0762 |
| RCSL3 | Material Rodoviário | B3 | -0.0727 |
| FHER3 | Fertilizantes e Defensivos | B3 | -0.0658 |
| BHIA3 | Eletrodomésticos | B3 | -0.0647 |
| BRAV3 | Exploração. Refino e Distribuição | B3 | -0.0541 |

**M3**:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| MNPR3 | Carnes e Derivados | B3 | -0.2103 |
| ENMT3 | Energia Elétrica | B3 | -0.1624 |
| SBFG3 | Produtos Diversos | B3 | -0.0924 |
| BRAV3 | Exploração. Refino e Distribuição | B3 | -0.0792 |
| FICT3 | Carnes e Derivados | B3 | -0.0668 |
| SUZB3 | Papel e Celulose | B3 | -0.0652 |
| MYPK3 | Automóveis e Motocicletas | B3 | -0.0610 |
| IRBR3 | Resseguradoras | B3 | -0.0560 |
| KEPL3 | Máq. e Equip. Industriais | B3 | -0.0532 |
| DESK3 | Telecomunicações | B3 | -0.0523 |

### 6.3 Operacional W3

| Mecanismo | Buys | Not. BUY | Sells | Not. SELL | Motivos SELL |
| --- | --- | --- | --- | --- | --- |
| M0 | 44 | 5.88 | 40 | 5.88 | MR_UCL:20, I_LCL:17, STRESS_AMP:3 |
| M1 | 150 | 21.24 | 119 | 21.99 | I_LCL:55, MR_UCL:48, STRESS_AMP:16 |
| M3 | 122 | 21.77 | 106 | 21.73 | MR_UCL:48, I_LCL:45, STRESS_AMP:13 |

### 6.4 Evento: Salto 2025-11-07

**M0:**

Contribuintes:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| ITUB4 | Bancos | B3 | 0.0002 |
| ITUB3 | Bancos | B3 | -0.0003 |
| P2ST34 | Tecnologia | BDR | -0.0099 |
| TKNO4 | Artefatos de Ferro e Aço | B3 | -0.0131 |
| CEEB5 | Energia Elétrica | B3 | -0.0209 |

Detratores:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| CEEB5 | Energia Elétrica | B3 | -0.0209 |
| TKNO4 | Artefatos de Ferro e Aço | B3 | -0.0131 |
| P2ST34 | Tecnologia | BDR | -0.0099 |
| ITUB3 | Bancos | B3 | -0.0003 |
| ITUB4 | Bancos | B3 | 0.0002 |

**M1:**

Contribuintes:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| IFCM3 | Programas e Serviços | B3 | 2.3501 |
| SEER3 | Serviços Educacionais | B3 | 0.0057 |
| DOTZ3 | Programas de Fidelização | B3 | 0.0002 |
| ITUB4 | Bancos | B3 | 0.0002 |
| ITUB3 | Bancos | B3 | -0.0003 |

Detratores:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| MYPK3 | Automóveis e Motocicletas | B3 | -0.0032 |
| BKCH39 | Não Classificados | B3 | -0.0020 |
| ITUB3 | Bancos | B3 | -0.0003 |
| ITUB4 | Bancos | B3 | 0.0002 |
| DOTZ3 | Programas de Fidelização | B3 | 0.0002 |

**M3:**

Contribuintes:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| SEER3 | Serviços Educacionais | B3 | 0.0084 |
| CEAB3 | Tecidos. Vestuário e Calçados | B3 | 0.0022 |
| ITUB4 | Bancos | B3 | 0.0003 |
| ITUB3 | Bancos | B3 | -0.0004 |
| CEBR5 | Energia Elétrica | B3 | -0.0012 |

Detratores:

| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| MYPK3 | Automóveis e Motocicletas | B3 | -0.0048 |
| SUZB3 | Papel e Celulose | B3 | -0.0043 |
| QUAL3 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 | -0.0038 |
| CEBR5 | Energia Elétrica | B3 | -0.0012 |
| ITUB3 | Bancos | B3 | -0.0004 |

### 6.5 Explicação causal W3

1. **IFCM3 domina M1 em W3**: PnL de +2.36, representando ~100% do retorno positivo de M1 na janela.
2. **Sem IFCM3, M1 em W3 seria ~-32%**: O resto da carteira M1 perdeu -0.68 combinado.
3. **M0 não tinha IFCM3**: Ativos ilíquidos competiram pelo top-10 no ranking M0 e deslocaram IFCM3.
4. **M3 não tinha IFCM3**: A penalização -z(vol) reduziu o ranking de IFCM3 (alta volatilidade).
5. **O resultado de W3 para M1 é idiossincrático, não sistemático**: Depender de um único ticker para +80% é risco de concentração, não virtude do mecanismo.

## 7. O fenômeno explicado

### 7.1 Dilema central: momentum vs inércia

| Regime | M0 (inércia) | M1/M3 (giro alto) |
| --- | --- | --- |
| Bull market (W1) | Dilui em ativos ruins → underperforms | Concentra em winners → outperforms |
| Bear market (W2) | Não gira → protege | Giro alto → compra losers → amplifica perdas |
| Lateral (W3) | Neutro | Depende de eventos idiossincráticos |

### 7.2 Evidência quantitativa do giro

| Janela | M0 buys | M0 sells | M1 buys | M1 sells | M3 buys | M3 sells |
| --- | --- | --- | --- | --- | --- | --- |
| W1 | 295 | 253 | 493 | 383 | 489 | 392 |
| W2 | 65 | 53 | 184 | 152 | 191 | 158 |
| W3 | 44 | 40 | 150 | 119 | 122 | 106 |

## 8. M3: definição, resultado e avaliação

### 8.1 Fórmula

```
score_m3 = z(score_m0) + z(ret_lookback_62) - z(vol_lookback_62)
```

Onde z() = z-score; score_m0 = mean(X_t,62d); ret = sum(X_t,62d); vol = std(X_t,62d).

### 8.2 Invariantes mantidos (idênticos a M1)
- Liquidez: volume > 0 em >= 50/62 dias
- 1 classe por empresa (mantém mais rentável)
- Cap 20% por setor (UNKNOWN conta)
- Mix B3: 50-80%; BDR: 20-50%
- Gating de venda com upside_extreme e stress_amp

### 8.3 Resultado comparativo

| Métrica | M0 | M1 | M3 |
| --- | --- | --- | --- |
| Retorno total | 105.56% | 313.82% | 138.54% |
| Max drawdown | -35.78% | -78.11% | -68.03% |
| W1 return | +168.10% | +408.02% | +436.96% |
| W2 return | -8.08% | -60.74% | -53.27% |
| W3 return | -2.88% | +80.13% | -19.98% |
| Avg cash ratio | 6.50% | 13.69% | 12.91% |
| Avg posições | 10.8 | 9.0 | 9.5 |

### 8.4 Avaliação de M3

1. **W1 (bull)**: M3 é o melhor (+437%), -z(vol) seleciona winners sustentáveis.
2. **W2 (bear)**: M3 melhora vs M1 (-53% vs -61%), mas ainda muito pior que M0 (-8%). Giro alto persiste.
3. **W3**: M3 perde -20% (sem IFCM3 que salvou M1). Pior que M0 (-3%).
4. **Drawdown**: -68% é inaceitável vs M0 (-36%).
5. **Conclusão**: M3 melhora marginalmente M1 mas NÃO resolve o problema fundamental (giro alto em bear).

## 9. Conclusão e próximos passos possíveis

| Mecanismo | Força | Fraqueza |
| --- | --- | --- |
| M0 | Proteção em bear (-8% em W2) | Retorno fraco em bull (+168% vs +408%) |
| M1 | Retorno em bull (+408%) | Drawdown catastrófico em bear (-78%) |
| M3 | Melhor bull (+437%), bear melhor que M1 | Drawdown alto (-68%), giro não resolvido |

**Problema fundamental**: filtros de M1/M3 aumentam giro 2-3x, e giro alto em bear é destrutivo.

**Possível M4 (fora do escopo atual, requer decisão Owner)**:
- Adaptar frequência de re-seleção ao regime Master (ex: re-selecionar apenas em RISK_ON; em RISK_OFF, congelar carteira como M0)
- Isso combinaria: seleção inteligente de M1/M3 em bull + inércia protetora de M0 em bear
- Requer nova regra → fora do escopo v1.3

---

## 10. Artefatos fonte (Parquet)

Todos extraídos de: `outputs/reports/task_017/run_20260212_125255/data/`

| Arquivo | Conteúdo |
| --- | --- |
| daily_portfolio_m0.parquet | 1864 linhas, 8 colunas: date, mechanism, equity, cash, n_positions... |
| daily_portfolio_m1.parquet | 1864 linhas, 8 colunas: date, mechanism, equity, cash, n_positions... |
| daily_portfolio_m3.parquet | 1864 linhas, 8 colunas: date, mechanism, equity, cash, n_positions... |
| decomp_pnl_event_months.parquet | 125 linhas, 6 colunas: mechanism, ticker, sector, asset_class, pnl_brl... |
| decomp_pnl_windows.parquet | 823 linhas, 6 colunas: mechanism, ticker, sector, asset_class, pnl_brl... |
| holdings_weekly_m0.parquet | 3339 linhas, 7 colunas: event_date, mechanism, ticker, value_brl, weight... |
| holdings_weekly_m1.parquet | 3294 linhas, 7 colunas: event_date, mechanism, ticker, value_brl, weight... |
| holdings_weekly_m3.parquet | 3330 linhas, 7 colunas: event_date, mechanism, ticker, value_brl, weight... |
| ledger_trades_m0.parquet | 876 linhas, 9 colunas: date, mechanism, action, ticker, qty... |
| ledger_trades_m1.parquet | 1818 linhas, 9 colunas: date, mechanism, action, ticker, qty... |
| ledger_trades_m3.parquet | 1705 linhas, 9 colunas: date, mechanism, action, ticker, qty... |
| master_state_daily.parquet | 1864 linhas, 6 colunas: date, xt, stress_i, stress_amp, trend_run7... |
| mtm_daily_by_ticker_m0.parquet | 7191 linhas, 9 colunas: date, mechanism, ticker, position_value_prev, position_value_curr... |
| mtm_daily_by_ticker_m1.parquet | 15820 linhas, 9 colunas: date, mechanism, ticker, position_value_prev, position_value_curr... |
| mtm_daily_by_ticker_m3.parquet | 16844 linhas, 9 colunas: date, mechanism, ticker, position_value_prev, position_value_curr... |
