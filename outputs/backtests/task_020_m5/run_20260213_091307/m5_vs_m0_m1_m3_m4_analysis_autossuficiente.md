# TASK 020 - M5 (M4 corrigido)

## Resumo executivo (M0/M1/M3/M4/M5)
| mechanism | equity_final | total_return | max_drawdown |
| --- | --- | --- | --- |
| M0 | 2.0556 | 1.0556 | -0.3578 |
| M1 | 4.1382 | 3.1382 | -0.7811 |
| M3 | 2.3854 | 1.3854 | -0.6803 |
| M4 | 1.2327 | 0.2327 | -0.1000 |
| M5 | 2.3021 | 1.3021 | -0.1000 |

## Métricas chave por mecanismo
| mechanism | equity_final | equity_peak | max_drawdown | max_dd_from_hwm | turnover_count | avg_cash_ratio | avg_n_positions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 2.0556 | 3.1145 | -0.3578 | nan | 876 | 0.0650 | 10.8433 |
| M1 | 4.1382 | 5.7016 | -0.7811 | nan | 1818 | 0.1369 | 9.0392 |
| M3 | 2.3854 | 6.1244 | -0.6803 | nan | 1705 | 0.1291 | 9.4871 |
| M4 | 1.2327 | 1.3696 | -0.1000 | 0.1000 | 227 | 0.9045 | 1.0322 |
| M5 | 2.3021 | 2.5485 | -0.1000 | 0.1000 | 1229 | 0.6385 | 3.9812 |

## Estado da máquina (dias em RISK_ON/RISK_OFF/HARD_PROTECTION)
| mechanism | days_risk_on | days_risk_off | days_hard_protection |
| --- | --- | --- | --- |
| M0 | nan | nan | nan |
| M1 | nan | nan | nan |
| M3 | nan | nan | nan |
| M4 | 170.0000 | 33.0000 | 1661.0000 |
| M5 | 672.0000 | 1021.0000 | 171.0000 |

## Validação de caixa rendendo CDI (M5)
- Dias com cash_ratio>0.999: **1071**
- mean_abs_error(ret_t - cdi_ret_t): **0.00000000**
| date | daily_return | cdi_ret_t | cash_ratio |
| --- | --- | --- | --- |
| 2018-10-10 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2018-10-11 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-02 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-03 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-06 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-07 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-08 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-09 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-10 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-13 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-14 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-15 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-16 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-17 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-20 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-21 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-22 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-23 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-24 00:00:00 | 0.0002 | 0.0002 | 1.0000 |
| 2019-05-27 00:00:00 | 0.0002 | 0.0002 | 1.0000 |

## Validação downside-only para volatilidade
- count(state_transition_to_risk_off_due_to_MR_above_ucl_with_ret_gt_0): **0**
- Gate S7: **PASS**

## Eventos de HWM-10%
| date | mechanism | hwm | hwm_floor | total_before_action | total_after_action | dd_from_hwm | action | clamp_delta | state_before | state_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019-05-02 00:00:00 | M5 | 1.3422 | 1.2080 | 1.1505 | 1.2080 | -0.1428 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0575 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2020-01-27 00:00:00 | M5 | 1.7306 | 1.5575 | 1.5440 | 1.5575 | -0.1078 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0135 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2021-08-16 00:00:00 | M5 | 1.9781 | 1.7803 | 1.6123 | 1.7803 | -0.1849 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.1680 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2021-09-17 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7753 | 1.7803 | -0.1025 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0050 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2021-10-26 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7720 | 1.7803 | -0.1042 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0083 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2021-11-17 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7571 | 1.7803 | -0.1117 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0232 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2021-12-20 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7447 | 1.7803 | -0.1180 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0356 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2022-01-14 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7782 | 1.7803 | -0.1011 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0021 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2022-02-17 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7624 | 1.7803 | -0.1090 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0179 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2022-03-08 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7353 | 1.7803 | -0.1227 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0450 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2022-04-06 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7800 | 1.7803 | -0.1001 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0003 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2022-05-18 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7690 | 1.7803 | -0.1057 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0113 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2022-06-07 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7737 | 1.7803 | -0.1033 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0066 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2022-06-28 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7733 | 1.7803 | -0.1035 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0070 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2022-08-26 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7766 | 1.7803 | -0.1019 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0037 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2022-09-13 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7668 | 1.7803 | -0.1068 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0135 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2022-10-04 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7753 | 1.7803 | -0.1025 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0050 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2022-10-25 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7782 | 1.7803 | -0.1010 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0020 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2022-11-16 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7230 | 1.7803 | -0.1290 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0573 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2022-12-06 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7663 | 1.7803 | -0.1071 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0140 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2023-01-17 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7794 | 1.7803 | -0.1005 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0009 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2023-02-23 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7559 | 1.7803 | -0.1123 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0244 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2023-03-17 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7595 | 1.7803 | -0.1105 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0208 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2023-04-04 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7560 | 1.7803 | -0.1123 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0243 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2023-04-25 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7223 | 1.7803 | -0.1293 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0580 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2023-05-16 00:00:00 | M5 | 1.9781 | 1.7803 | 1.7800 | 1.7803 | -0.1002 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0003 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2024-06-10 00:00:00 | M5 | 2.4409 | 2.1968 | 2.1805 | 2.1968 | -0.1067 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0164 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2025-07-28 00:00:00 | M5 | 2.5485 | 2.2937 | 2.2921 | 2.2937 | -0.1006 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0015 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2025-08-14 00:00:00 | M5 | 2.5485 | 2.2937 | 2.2671 | 2.2937 | -0.1104 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0266 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2025-09-02 00:00:00 | M5 | 2.5485 | 2.2937 | 2.2781 | 2.2937 | -0.1061 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0155 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2025-09-23 00:00:00 | M5 | 2.5485 | 2.2937 | 2.2597 | 2.2937 | -0.1133 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0340 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2025-10-21 00:00:00 | M5 | 2.5485 | 2.2937 | 2.1709 | 2.2937 | -0.1482 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.1228 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |
| 2025-11-11 00:00:00 | M5 | 2.5485 | 2.2937 | 2.2732 | 2.2937 | -0.1080 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0205 | PORTFOLIO_RISK_ON | HARD_PROTECTION |
| 2025-12-11 00:00:00 | M5 | 2.5485 | 2.2937 | 2.2449 | 2.2937 | -0.1192 | FORCE_TO_CASH_AND_BLOCK_BUYS_WITH_FLOOR_CLAMP | 0.0488 | PORTFOLIO_RISK_OFF | HARD_PROTECTION |

## Comparação por fases W1/W2/W3
| mechanism | W1_return | W1_max_drawdown | W2_return | W2_max_drawdown | W3_return | W3_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| M0 | 1.6810 | -0.2739 | -0.0808 | -0.2536 | -0.0288 | -0.3559 |
| M1 | 4.0802 | -0.2085 | -0.6074 | -0.6930 | 0.8013 | -0.7811 |
| M3 | 4.3696 | -0.1874 | -0.5327 | -0.6129 | -0.1998 | -0.6803 |
| M4 | 0.2327 | -0.1000 | 0.0000 | -0.1000 | 0.0000 | -0.1000 |
| M5 | 0.7006 | -0.1000 | 0.0621 | -0.1000 | 0.0265 | -0.1000 |

## Top contribuintes/detratores por fase (M5 vs M3/M4)
### W1
**M5 - Top 10**
| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| PRIO3 | Exploração. Refino e Distribuição | B3 | 0.0835 |
| LOGN3 | Transporte Hidroviário | B3 | 0.0752 |
| JFEN3 | Incorporações | B3 | 0.0633 |
| JHSF3 | Incorporações | B3 | 0.0578 |
| WHRL4 | Eletrodomésticos | B3 | 0.0553 |
| RSID3 | Incorporações | B3 | 0.0521 |
| CSUD3 | Serviços Financeiros Diversos | B3 | 0.0436 |
| WLMM4 | Material de Transporte | B3 | 0.0428 |
| POSI3 | Computadores e Equipamentos | B3 | 0.0418 |
| KEPL3 | Máq. e Equip. Industriais | B3 | 0.0400 |

**M5 - Bottom 10**
| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| PSVM11 | Serviços de Apoio e Armazenagem | B3 | -0.0528 |
| LPSB3 | Intermediação Imobiliária | B3 | -0.0355 |
| GFSA3 | Incorporações | B3 | -0.0272 |
| GSHP3 | Exploração de Imóveis | B3 | -0.0270 |
| CSNA3 | Siderurgia | B3 | -0.0172 |
| DASA3 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 | -0.0166 |
| ENGI3 | Energia Elétrica | B3 | -0.0163 |
| RANI3 | Embalagens | B3 | -0.0137 |
| M1TA34 | Redes Sociais | BDR | -0.0136 |
| MBRF3 | Carnes e Derivados | B3 | -0.0124 |

**M4 - Top 10**
| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| LOGN3 | Transporte Hidroviário | B3 | 0.0754 |
| KEPL3 | Máq. e Equip. Industriais | B3 | 0.0505 |
| PPLA11 | Financeiro e Outros/Serviços Financeiros/Gestão de Recursos e Investimentos | BDR | 0.0351 |
| FHER3 | Fertilizantes e Defensivos | B3 | 0.0327 |
| BPAC11 | Bancos | B3 | 0.0284 |
| TRIS3 | Incorporações | B3 | 0.0265 |
| IRBR3 | Resseguradoras | B3 | 0.0236 |
| PRIO3 | Exploração. Refino e Distribuição | B3 | 0.0224 |
| BAUH4 | Carnes e Derivados | B3 | 0.0200 |
| BEEF3 | Carnes e Derivados | B3 | 0.0185 |

**M4 - Bottom 10**
| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| LPSB3 | Intermediação Imobiliária | B3 | -0.0354 |
| GFSA3 | Incorporações | B3 | -0.0272 |
| CSNA3 | Siderurgia | B3 | -0.0247 |
| INEP3 | Máq. e Equip. Industriais | B3 | -0.0171 |
| ENGI3 | Energia Elétrica | B3 | -0.0166 |
| ETER3 | Produtos para Construção | B3 | -0.0161 |
| M1TA34 | Redes Sociais | BDR | -0.0136 |
| MBRF3 | Carnes e Derivados | B3 | -0.0124 |
| INEP4 | Máq. e Equip. Industriais | B3 | -0.0105 |
| BRKM5 | Petroquímicos | B3 | -0.0093 |

**M3 - Top 10**
| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| FHER3 | Fertilizantes e Defensivos | B3 | 0.5870 |
| INEP3 | Máq. e Equip. Industriais | B3 | 0.5694 |
| LWSA3 | Programas e Serviços | B3 | 0.4811 |
| CRPG6 | Químicos Diversos | B3 | 0.3866 |
| RCSL3 | Material Rodoviário | B3 | 0.3155 |
| TASA3 | Armas e Munições | B3 | 0.2569 |
| BRKM5 | Petroquímicos | B3 | 0.2385 |
| ETER3 | Produtos para Construção | B3 | 0.1945 |
| PPLA11 | Financeiro e Outros/Serviços Financeiros/Gestão de Recursos e Investimentos | BDR | 0.1567 |
| CSNA3 | Siderurgia | B3 | 0.1507 |

**M3 - Bottom 10**
| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| DASA3 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 | -0.2980 |
| MGLU3 | Eletrodomésticos | B3 | -0.1176 |
| PSVM11 | Serviços de Apoio e Armazenagem | B3 | -0.1080 |
| GUAR3 | Tecidos. Vestuário e Calçados | B3 | -0.0988 |
| HBOR3 | Incorporações | B3 | -0.0928 |
| VIVR3 | Incorporações | B3 | -0.0639 |
| TELB3 | Telecomunicações | B3 | -0.0601 |
| OSXB3 | Equipamentos e Serviços | B3 | -0.0565 |
| COPH34 | Petróleo | BDR | -0.0557 |
| F1AN34 | Energia | BDR | -0.0515 |

### W2
**M5 - Top 10**
| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| B1NT34 | Farmacêutico | BDR | 0.1216 |
| HAGA3 | Artefatos de Ferro e Aço | B3 | 0.1034 |
| M1RN34 | Farmacêutico | BDR | 0.0656 |
| CEBR5 | Energia Elétrica | B3 | 0.0491 |
| AMBP3 | Água e Saneamento | B3 | 0.0472 |
| LWSA3 | Programas e Serviços | B3 | 0.0340 |
| CMIG3 | Energia Elétrica | B3 | 0.0273 |
| SYNE3 | Exploração de Imóveis | B3 | 0.0237 |
| FDMO34 | Automóvel | BDR | 0.0224 |
| E1DU34 | Prestador de Serviços | BDR | 0.0202 |

**M5 - Bottom 10**
| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| SIMH3 | Holdings Diversificadas | B3 | -0.1273 |
| IFCM3 | Programas e Serviços | B3 | -0.0546 |
| FHER3 | Fertilizantes e Defensivos | B3 | -0.0275 |
| RNEW3 | Energia Elétrica | B3 | -0.0206 |
| BCHI39 | Não Classificados | B3 | -0.0193 |
| BRBI11 | Bancos | B3 | -0.0192 |
| MWET4 | Material Rodoviário | B3 | -0.0188 |
| M1NS34 | Bebidas | BDR | -0.0181 |
| HOME34 | Lojas de departamento | BDR | -0.0181 |
| D1VN34 | Energia | BDR | -0.0180 |

**M3 - Top 10**
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

**M3 - Bottom 10**
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

### W3
**M5 - Top 10**
| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| K2CG34 | Tecnologia | BDR | 0.0997 |
| AGXY3 | Agricultura | B3 | 0.0561 |
| T1WL34 | Tecnologia | BDR | 0.0555 |
| ASAI3 | Alimentos | B3 | 0.0444 |
| DBAG34 | Financeiro | BDR | 0.0362 |
| S2ED34 | Energia | BDR | 0.0320 |
| ITUB4 | Bancos | B3 | 0.0302 |
| TTEN3 | Agricultura | B3 | 0.0288 |
| ITUB3 | Bancos | B3 | 0.0272 |
| FICT3 | Carnes e Derivados | B3 | 0.0250 |

**M5 - Bottom 10**
| ticker | sector | asset_class | pnl_brl |
| --- | --- | --- | --- |
| U1AL34 | Aviação | BDR | -0.1080 |
| M2ST34 | Tecnologia | BDR | -0.0836 |
| SEER3 | Serviços Educacionais | B3 | -0.0457 |
| CEAB3 | Tecidos. Vestuário e Calçados | B3 | -0.0456 |
| HAPV3 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 | -0.0361 |
| COGN3 | Serviços Educacionais | B3 | -0.0346 |
| AMAR3 | Tecidos. Vestuário e Calçados | B3 | -0.0255 |
| GFSA3 | Incorporações | B3 | -0.0245 |
| SHOW3 | Produção de Eventos e Shows | B3 | -0.0220 |
| RCSL4 | Material Rodoviário | B3 | -0.0162 |

**M3 - Top 10**
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

**M3 - Bottom 10**
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

## Compliance BUY blocking
- BUY_count_during_PORTFOLIO_RISK_OFF_M5 = **0**
- BUY_count_during_HARD_PROTECTION_M5 = **0**
- Gate S6 = **PASS**

## Gates
- S5 HWM guardrail M5: **PASS** (max_dd_from_hwm=0.100000)
- S6 Buy blocking: **PASS**
- S7 Downside-only volatility: **PASS**
- S8 Cash rende CDI: **PASS**
- S9 HTML TOTAL_EM_PERCENTUAL_DO_CDI: **PASS**