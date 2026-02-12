# Pacote de evidências - M0 vs M1 por fases + definição de M3

- Run forense base: `/home/wilson/CEP_COMPRA/outputs/forensics/task_015/run_20260212_121309`
- Run M3 comparativo: `/home/wilson/CEP_COMPRA/outputs/backtests/task_015_m3/run_20260212_121309`

## 1) Qual M3 foi criado

Trechos literais da proposta:
- `"score_m3 = z(score_m0) + z(ret_lookback_62) - z(vol_lookback_62)"`
- `"volume > 0 em pelo menos 50/62 dias (excludente)"`
- `"cap por setor 20% (UNKNOWN conta)"`
- `"mix B3 entre 50% e 80%"`

Parâmetros operacionais (paráfrase objetiva):
- Ranking M3 combina score base do M0, retorno acumulado no lookback e penalização por volatilidade no lookback.
- Desempate operacional segue score e critérios determinísticos de ticker no pipeline.
- Invariantes mantidos: liquidez 50/62, uma classe por empresa, cap setor 20%, mix B3/BDR 50-80%.

## 2) Top contribuições por janela

### W1
**Top-10 tickers M0**

| rank | ticker | pnl_brl | sector | asset_class |
| --- | --- | --- | --- | --- |
| 1 | A1LG34 | 0.284005 | Tecnologia | BDR |
| 2 | FHER3 | 0.195178 | Fertilizantes e Defensivos | B3 |
| 3 | INEP3 | 0.165960 | Máq. e Equip. Industriais | B3 |
| 4 | NORD3 | 0.154136 | Exploração de Imóveis | B3 |
| 5 | TXRX3 | 0.137424 | Fios e Tecidos | B3 |
| 6 | MAPT4 | -0.133479 | Outros | B3 |
| 7 | GPAR3 | -0.125230 | Energia Elétrica | B3 |
| 8 | INEP4 | 0.122580 | Máq. e Equip. Industriais | B3 |
| 9 | ENMT4 | 0.121911 | Energia Elétrica | B3 |
| 10 | BSLI3 | 0.118570 | Bancos | B3 |

**Top-10 tickers M1**

| rank | ticker | pnl_brl | sector | asset_class |
| --- | --- | --- | --- | --- |
| 1 | INEP3 | 0.558399 | Máq. e Equip. Industriais | B3 |
| 2 | RCSL3 | 0.506865 | Material Rodoviário | B3 |
| 3 | FHER3 | 0.480358 | Fertilizantes e Defensivos | B3 |
| 4 | LWSA3 | 0.430436 | Programas e Serviços | B3 |
| 5 | CRPG6 | 0.422041 | Químicos Diversos | B3 |
| 6 | DASA3 | -0.269158 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 |
| 7 | LOGN3 | 0.218867 | Transporte Hidroviário | B3 |
| 8 | TASA3 | 0.186710 | Armas e Munições | B3 |
| 9 | PPLA11 | 0.160969 | Financeiro e Outros/Serviços Financeiros/Gestão de Recursos e Investimentos | BDR |
| 10 | ETER3 | 0.152497 | Produtos para Construção | B3 |

**Top-5 setor/asset_class por mecanismo**

| mechanism | rank | asset_class | sector | pnl_brl |
| --- | --- | --- | --- | --- |
| M0 | 1 | BDR | Tecnologia | 0.289851 |
| M0 | 2 | B3 | Máq. e Equip. Industriais | 0.289252 |
| M0 | 3 | B3 | Energia Elétrica | 0.196124 |
| M0 | 4 | B3 | Programas e Serviços | 0.171373 |
| M0 | 5 | B3 | Outros | -0.133479 |
| M1 | 1 | B3 | Máq. e Equip. Industriais | 0.564535 |
| M1 | 2 | B3 | Material Rodoviário | 0.520564 |
| M1 | 3 | B3 | Fertilizantes e Defensivos | 0.480358 |
| M1 | 4 | B3 | Programas e Serviços | 0.443531 |
| M1 | 5 | B3 | Químicos Diversos | 0.422002 |

### W2
**Top-10 tickers M0**

| rank | ticker | pnl_brl | sector | asset_class |
| --- | --- | --- | --- | --- |
| 1 | RSUL4 | 0.412247 | Material Rodoviário | B3 |
| 2 | GPAR3 | -0.133874 | Energia Elétrica | B3 |
| 3 | MWET4 | -0.109908 | Material Rodoviário | B3 |
| 4 | COCE3 | -0.107884 | Energia Elétrica | B3 |
| 5 | ENMT4 | 0.106548 | Energia Elétrica | B3 |
| 6 | PATI4 | -0.096860 | Artefatos de Ferro e Aço | B3 |
| 7 | SOND5 | -0.096250 | Engenharia Consultiva | B3 |
| 8 | SNSY5 | -0.085633 | Materiais Diversos | B3 |
| 9 | CRPG3 | 0.072584 | Químicos Diversos | B3 |
| 10 | HBTS5 | -0.068667 | Exploração de Imóveis | B3 |

**Top-10 tickers M1**

| rank | ticker | pnl_brl | sector | asset_class |
| --- | --- | --- | --- | --- |
| 1 | FHER3 | -0.212580 | Fertilizantes e Defensivos | B3 |
| 2 | CASH3 | -0.210964 | Programas e Serviços | B3 |
| 3 | MWET4 | -0.203492 | Material Rodoviário | B3 |
| 4 | D1OC34 | -0.200817 | Tecnologia | BDR |
| 5 | KEPL3 | -0.198968 | Máq. e Equip. Industriais | B3 |
| 6 | IFCM3 | -0.151780 | Programas e Serviços | B3 |
| 7 | ALLD3 | -0.137342 | Eletrodomésticos | B3 |
| 8 | BRAV3 | -0.135818 | Exploração. Refino e Distribuição | B3 |
| 9 | SYNE3 | -0.126681 | Exploração de Imóveis | B3 |
| 10 | CBAV3 | -0.124887 | Minerais Metálicos | B3 |

**Top-5 setor/asset_class por mecanismo**

| mechanism | rank | asset_class | sector | pnl_brl |
| --- | --- | --- | --- | --- |
| M0 | 1 | B3 | Material Rodoviário | 0.302339 |
| M0 | 2 | B3 | Energia Elétrica | -0.164641 |
| M0 | 3 | B3 | Engenharia Consultiva | -0.149099 |
| M0 | 4 | B3 | Materiais Diversos | -0.085633 |
| M0 | 5 | B3 | Químicos Diversos | 0.076107 |
| M1 | 1 | B3 | Programas e Serviços | -0.444967 |
| M1 | 2 | B3 | Exploração. Refino e Distribuição | -0.250066 |
| M1 | 3 | B3 | Máq. e Equip. Industriais | -0.241703 |
| M1 | 4 | BDR | Tecnologia | -0.217919 |
| M1 | 5 | B3 | Fertilizantes e Defensivos | -0.208618 |

### W2a
**Top-10 tickers M0**

| rank | ticker | pnl_brl | sector | asset_class |
| --- | --- | --- | --- | --- |
| 1 | RSUL4 | 0.177094 | Material Rodoviário | B3 |
| 2 | ENMT4 | 0.068859 | Energia Elétrica | B3 |
| 3 | CRPG3 | -0.062906 | Químicos Diversos | B3 |
| 4 | JOPA4 | -0.051645 | Alimentos Diversos | B3 |
| 5 | PATI4 | -0.048991 | Artefatos de Ferro e Aço | B3 |
| 6 | CLSC3 | -0.045292 | Energia Elétrica | B3 |
| 7 | MWET4 | -0.043196 | Material Rodoviário | B3 |
| 8 | COCE3 | -0.034611 | Energia Elétrica | B3 |
| 9 | PATI3 | 0.026816 | Artefatos de Ferro e Aço | B3 |
| 10 | HBTS5 | -0.024573 | Exploração de Imóveis | B3 |

**Top-10 tickers M1**

| rank | ticker | pnl_brl | sector | asset_class |
| --- | --- | --- | --- | --- |
| 1 | CASH3 | -0.179714 | Programas e Serviços | B3 |
| 2 | MGEL4 | -0.120931 | Artefatos de Ferro e Aço | B3 |
| 3 | CRPG6 | -0.110140 | Químicos Diversos | B3 |
| 4 | PLAS3 | -0.101897 | Automóveis e Motocicletas | B3 |
| 5 | MWET4 | -0.079491 | Material Rodoviário | B3 |
| 6 | RSUL4 | 0.045927 | Material Rodoviário | B3 |
| 7 | D1OC34 | -0.039802 | Tecnologia | BDR |
| 8 | AMBP3 | -0.038609 | Água e Saneamento | B3 |
| 9 | BRBI11 | -0.036751 | Bancos | B3 |
| 10 | MNDL3 | 0.035908 | Acessórios | B3 |

**Top-5 setor/asset_class por mecanismo**

| mechanism | rank | asset_class | sector | pnl_brl |
| --- | --- | --- | --- | --- |
| M0 | 1 | B3 | Material Rodoviário | 0.133897 |
| M0 | 2 | B3 | Químicos Diversos | -0.062906 |
| M0 | 3 | B3 | Alimentos Diversos | -0.051645 |
| M0 | 4 | B3 | Energia Elétrica | -0.030864 |
| M0 | 5 | B3 | Exploração de Imóveis | -0.024573 |
| M1 | 1 | B3 | Programas e Serviços | -0.194563 |
| M1 | 2 | B3 | Artefatos de Ferro e Aço | -0.120931 |
| M1 | 3 | B3 | Químicos Diversos | -0.110140 |
| M1 | 4 | B3 | Automóveis e Motocicletas | -0.101897 |
| M1 | 5 | BDR | Farmacêutico | 0.065757 |

### W2b
**Top-10 tickers M0**

| rank | ticker | pnl_brl | sector | asset_class |
| --- | --- | --- | --- | --- |
| 1 | PATI4 | -0.025503 | Artefatos de Ferro e Aço | B3 |
| 2 | SOND6 | -0.024187 | Engenharia Consultiva | B3 |
| 3 | LPSB3 | -0.020676 | Intermediação Imobiliária | B3 |
| 4 | GPAR3 | 0.018937 | Energia Elétrica | B3 |
| 5 | SOND5 | -0.016643 | Engenharia Consultiva | B3 |
| 6 | EQPA5 | -0.013515 | Energia Elétrica | B3 |
| 7 | ENMT4 | -0.010072 | Energia Elétrica | B3 |
| 8 | I2NG34 | 0.006461 | Produtos de consumo não duráveis | BDR |
| 9 | JOPA4 | -0.002766 | Alimentos Diversos | B3 |
| 10 | OXYP34 | -0.000998 | Energia | BDR |

**Top-10 tickers M1**

| rank | ticker | pnl_brl | sector | asset_class |
| --- | --- | --- | --- | --- |
| 1 | KEPL3 | -0.222161 | Máq. e Equip. Industriais | B3 |
| 2 | TRAD3 | -0.085690 | Programas e Serviços | B3 |
| 3 | CBAV3 | -0.075525 | Minerais Metálicos | B3 |
| 4 | BRAV3 | -0.063552 | Exploração. Refino e Distribuição | B3 |
| 5 | LPSB3 | -0.042052 | Intermediação Imobiliária | B3 |
| 6 | PSVM11 | -0.039525 | Serviços de Apoio e Armazenagem | B3 |
| 7 | MOSC34 | -0.028391 | Materiais básicos | BDR |
| 8 | VULC3 | -0.022700 | Calçados | B3 |
| 9 | SBSP3 | -0.017925 | Água e Saneamento | B3 |
| 10 | DOTZ3 | -0.013260 | Programas de Fidelização | B3 |

**Top-5 setor/asset_class por mecanismo**

| mechanism | rank | asset_class | sector | pnl_brl |
| --- | --- | --- | --- | --- |
| M0 | 1 | B3 | Engenharia Consultiva | -0.040830 |
| M0 | 2 | B3 | Artefatos de Ferro e Aço | -0.025503 |
| M0 | 3 | B3 | Intermediação Imobiliária | -0.020676 |
| M0 | 4 | BDR | Produtos de consumo não duráveis | 0.006461 |
| M0 | 5 | B3 | Energia Elétrica | -0.004650 |
| M1 | 1 | B3 | Máq. e Equip. Industriais | -0.222161 |
| M1 | 2 | B3 | Programas e Serviços | -0.085690 |
| M1 | 3 | B3 | Minerais Metálicos | -0.075525 |
| M1 | 4 | B3 | Exploração. Refino e Distribuição | -0.054090 |
| M1 | 5 | B3 | Intermediação Imobiliária | -0.042052 |

### W3
**Top-10 tickers M0**

| rank | ticker | pnl_brl | sector | asset_class |
| --- | --- | --- | --- | --- |
| 1 | CCTY3 | 0.207608 | Incorporações | B3 |
| 2 | B2RK34 | -0.090527 | Equipamentos Médicos | BDR |
| 3 | M2KS34 | -0.088860 | Tecnologia | BDR |
| 4 | GFSA3 | -0.082729 | Incorporações | B3 |
| 5 | AZTE3 | -0.073971 | Exploração. Refino e Distribuição | B3 |
| 6 | D2OC34 | 0.073902 | Tecnologia | BDR |
| 7 | E2XE34 | 0.071579 | Pacífico Asiático, Ex Japão | BDR |
| 8 | G2WR34 | 0.070651 | Tecnologia | BDR |
| 9 | CALI3 | -0.067699 | Incorporações | B3 |
| 10 | P2CY34 | -0.065592 | Consultoria | BDR |

**Top-10 tickers M1**

| rank | ticker | pnl_brl | sector | asset_class |
| --- | --- | --- | --- | --- |
| 1 | IFCM3 | 2.355068 | Programas e Serviços | B3 |
| 2 | AMBP3 | 0.273585 | Água e Saneamento | B3 |
| 3 | FICT3 | -0.164727 | Carnes e Derivados | B3 |
| 4 | A2RR34 | 0.143712 | Pacífico Asiático, Ex Japão | BDR |
| 5 | ENMT3 | -0.140770 | Energia Elétrica | B3 |
| 6 | RCSL4 | 0.139322 | Material Rodoviário | B3 |
| 7 | DASA3 | 0.128151 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 |
| 8 | GFSA3 | -0.117006 | Incorporações | B3 |
| 9 | SBFG3 | -0.079110 | Produtos Diversos | B3 |
| 10 | CTSA4 | -0.078312 | Fios e Tecidos | B3 |

**Top-5 setor/asset_class por mecanismo**

| mechanism | rank | asset_class | sector | pnl_brl |
| --- | --- | --- | --- | --- |
| M0 | 1 | B3 | Bancos | 0.109814 |
| M0 | 2 | B3 | Exploração. Refino e Distribuição | -0.073971 |
| M0 | 3 | BDR | Pacífico Asiático, Ex Japão | 0.072036 |
| M0 | 4 | BDR | Equipamentos Médicos | -0.070811 |
| M0 | 5 | BDR | Consultoria | -0.065592 |
| M1 | 1 | B3 | Programas e Serviços | 2.409685 |
| M1 | 2 | B3 | Água e Saneamento | 0.273585 |
| M1 | 3 | B3 | Carnes e Derivados | -0.259168 |
| M1 | 4 | B3 | Energia Elétrica | -0.160941 |
| M1 | 5 | BDR | Pacífico Asiático, Ex Japão | 0.143712 |

## 3) Snapshots sentinela de holdings

| sentinela | mechanism | event_date | ticker | weight | value_brl | sector | asset_class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SENTINELA_2021_06_30 | M0 | 2021-06-14 00:00:00 | INEP3 | 0.165698 | 0.451682 | Máq. e Equip. Industriais | B3 |
| SENTINELA_2021_06_30 | M0 | 2021-06-14 00:00:00 | PATI3 | 0.114744 | 0.312785 | Artefatos de Ferro e Aço | B3 |
| SENTINELA_2021_06_30 | M0 | 2021-06-14 00:00:00 | ENMT4 | 0.114733 | 0.312754 | Energia Elétrica | B3 |
| SENTINELA_2021_06_30 | M0 | 2021-06-14 00:00:00 | USIM6 | 0.108730 | 0.296392 | Siderurgia | B3 |
| SENTINELA_2021_06_30 | M0 | 2021-06-14 00:00:00 | CRPG3 | 0.100000 | 0.272593 | Químicos Diversos | B3 |
| SENTINELA_2021_06_30 | M0 | 2021-06-14 00:00:00 | RSUL4 | 0.100000 | 0.272593 | Material Rodoviário | B3 |
| SENTINELA_2021_06_30 | M0 | 2021-06-14 00:00:00 | COCE3 | 0.097821 | 0.266655 | Energia Elétrica | B3 |
| SENTINELA_2021_06_30 | M0 | 2021-06-14 00:00:00 | JOPA4 | 0.083499 | 0.227613 | Alimentos Diversos | B3 |
| SENTINELA_2021_06_30 | M0 | 2021-06-14 00:00:00 | NORD3 | 0.077970 | 0.212541 | Exploração de Imóveis | B3 |
| SENTINELA_2021_06_30 | M0 | 2021-06-14 00:00:00 | POSI3 | 0.018860 | 0.051411 | Computadores e Equipamentos | B3 |
| SENTINELA_2024_09_02 | M0 | 2024-09-02 00:00:00 | SOND6 | 0.147939 | 0.332574 | Engenharia Consultiva | B3 |
| SENTINELA_2024_09_02 | M0 | 2024-09-02 00:00:00 | B2RK34 | 0.100165 | 0.225176 | Equipamentos Médicos | BDR |
| SENTINELA_2024_09_02 | M0 | 2024-09-02 00:00:00 | G2WR34 | 0.098310 | 0.221006 | Tecnologia | BDR |
| SENTINELA_2024_09_02 | M0 | 2024-09-02 00:00:00 | CALI3 | 0.097872 | 0.220020 | Incorporações | B3 |
| SENTINELA_2024_09_02 | M0 | 2024-09-02 00:00:00 | M2KS34 | 0.090001 | 0.202327 | Tecnologia | BDR |
| SENTINELA_2024_09_02 | M0 | 2024-09-02 00:00:00 | E2XE34 | 0.089299 | 0.200748 | Pacífico Asiático, Ex Japão | BDR |
| SENTINELA_2024_09_02 | M0 | 2024-09-02 00:00:00 | EKTR3 | 0.084724 | 0.190465 | Energia Elétrica | B3 |
| SENTINELA_2024_09_02 | M0 | 2024-09-02 00:00:00 | B2LN34 | 0.079377 | 0.178443 | Tecnologia | BDR |
| SENTINELA_2024_09_02 | M0 | 2024-09-02 00:00:00 | P2ST34 | 0.073061 | 0.164244 | Tecnologia | BDR |
| SENTINELA_2024_09_02 | M0 | 2024-09-02 00:00:00 | JOPA4 | 0.047298 | 0.106329 | Alimentos Diversos | B3 |
| SENTINELA_2021_06_30 | M1 | 2021-06-14 00:00:00 | INEP3 | 0.154950 | 0.809659 | Máq. e Equip. Industriais | B3 |
| SENTINELA_2021_06_30 | M1 | 2021-06-14 00:00:00 | CRPG6 | 0.135128 | 0.706085 | Químicos Diversos | B3 |
| SENTINELA_2021_06_30 | M1 | 2021-06-14 00:00:00 | LUPA3 | 0.100000 | 0.522529 | Equipamentos e Serviços | B3 |
| SENTINELA_2021_06_30 | M1 | 2021-06-14 00:00:00 | PLAS3 | 0.100000 | 0.522529 | Automóveis e Motocicletas | B3 |
| SENTINELA_2021_06_30 | M1 | 2021-06-14 00:00:00 | MWET4 | 0.100000 | 0.522529 | Material Rodoviário | B3 |
| SENTINELA_2021_06_30 | M1 | 2021-06-14 00:00:00 | POSI3 | 0.096798 | 0.505799 | Computadores e Equipamentos | B3 |
| SENTINELA_2021_06_30 | M1 | 2021-06-14 00:00:00 | NORD3 | 0.073376 | 0.383411 | Exploração de Imóveis | B3 |
| SENTINELA_2021_06_30 | M1 | 2021-06-14 00:00:00 | BRAV3 | 0.072625 | 0.379486 | Exploração. Refino e Distribuição | B3 |
| SENTINELA_2021_06_30 | M1 | 2021-06-14 00:00:00 | PSVM11 | 0.066075 | 0.345260 | Serviços de Apoio e Armazenagem | B3 |
| SENTINELA_2021_06_30 | M1 | 2021-06-14 00:00:00 | NGRD3 | 0.060661 | 0.316970 | Programas e Serviços | B3 |
| SENTINELA_2024_09_02 | M1 | 2024-09-02 00:00:00 | ENMT3 | 0.200294 | 0.460210 | Energia Elétrica | B3 |
| SENTINELA_2024_09_02 | M1 | 2024-09-02 00:00:00 | RCSL3 | 0.192023 | 0.441206 | Material Rodoviário | B3 |
| SENTINELA_2024_09_02 | M1 | 2024-09-02 00:00:00 | SNSY3 | 0.112746 | 0.259053 | Materiais Diversos | B3 |
| SENTINELA_2024_09_02 | M1 | 2024-09-02 00:00:00 | IRBR3 | 0.100000 | 0.229768 | Resseguradoras | B3 |
| SENTINELA_2024_09_02 | M1 | 2024-09-02 00:00:00 | SBFG3 | 0.099053 | 0.227592 | Produtos Diversos | B3 |
| SENTINELA_2024_09_02 | M1 | 2024-09-02 00:00:00 | AERI3 | 0.073551 | 0.168996 | Máq. e Equip. Industriais | B3 |
| SENTINELA_2024_09_02 | M1 | 2024-09-02 00:00:00 | CMIN3 | 0.073069 | 0.167888 | Minerais Metálicos | B3 |
| SENTINELA_2024_09_02 | M1 | 2024-09-02 00:00:00 | CRPG6 | 0.067192 | 0.154385 | Químicos Diversos | B3 |
| SENTINELA_2024_09_02 | M1 | 2024-09-02 00:00:00 | BCPX39 | 0.057660 | 0.132484 | Não Classificados | B3 |
| SENTINELA_2024_09_02 | M1 | 2024-09-02 00:00:00 | RCSL4 | 0.024413 | 0.056094 | Material Rodoviário | B3 |
| PIOR_SEMANA_W2a | M0 | 2021-09-20 00:00:00 | ENMT4 | 0.141855 | 0.413471 | Energia Elétrica | B3 |
| PIOR_SEMANA_W2a | M0 | 2021-09-20 00:00:00 | M1RN34 | 0.122710 | 0.357669 | Farmacêutico | BDR |
| PIOR_SEMANA_W2a | M0 | 2021-09-20 00:00:00 | EQPA5 | 0.111286 | 0.324370 | Energia Elétrica | B3 |
| PIOR_SEMANA_W2a | M0 | 2021-09-20 00:00:00 | HBTS5 | 0.090868 | 0.264858 | Exploração de Imóveis | B3 |
| PIOR_SEMANA_W2a | M0 | 2021-09-20 00:00:00 | D2AS34 | 0.088867 | 0.259025 | Comunicação | BDR |
| PIOR_SEMANA_W2a | M0 | 2021-09-20 00:00:00 | ESTR4 | 0.086773 | 0.252921 | Brinquedos e Jogos | B3 |
| PIOR_SEMANA_W2a | M0 | 2021-09-20 00:00:00 | PATI4 | 0.086100 | 0.250961 | Artefatos de Ferro e Aço | B3 |
| PIOR_SEMANA_W2a | M0 | 2021-09-20 00:00:00 | USIM6 | 0.084002 | 0.244845 | Siderurgia | B3 |
| PIOR_SEMANA_W2a | M0 | 2021-09-20 00:00:00 | COCE3 | 0.083248 | 0.242648 | Energia Elétrica | B3 |
| PIOR_SEMANA_W2a | M0 | 2021-09-20 00:00:00 | MWET4 | 0.058393 | 0.170200 | Material Rodoviário | B3 |
| PIOR_SEMANA_W2b | M0 | 2022-05-30 00:00:00 | ENMT4 | 0.183296 | 0.450524 | Energia Elétrica | B3 |
| PIOR_SEMANA_W2b | M0 | 2022-05-30 00:00:00 | EQPA5 | 0.125555 | 0.308602 | Energia Elétrica | B3 |
| PIOR_SEMANA_W2b | M0 | 2022-05-30 00:00:00 | SOND6 | 0.105786 | 0.260013 | Engenharia Consultiva | B3 |
| PIOR_SEMANA_W2b | M0 | 2022-05-30 00:00:00 | PATI4 | 0.100748 | 0.247629 | Artefatos de Ferro e Aço | B3 |
| PIOR_SEMANA_W2b | M0 | 2022-05-30 00:00:00 | LAND3 | 0.100527 | 0.247086 | Agricultura | B3 |
| PIOR_SEMANA_W2b | M0 | 2022-05-30 00:00:00 | SOND5 | 0.090281 | 0.221903 | Engenharia Consultiva | B3 |
| PIOR_SEMANA_W2b | M0 | 2022-05-30 00:00:00 | COCE3 | 0.088265 | 0.216947 | Energia Elétrica | B3 |
| PIOR_SEMANA_W2b | M0 | 2022-05-30 00:00:00 | I2NG34 | 0.055031 | 0.135260 | Produtos de consumo não duráveis | BDR |
| PIOR_SEMANA_W2b | M0 | 2022-05-30 00:00:00 | EQPA7 | 0.053497 | 0.131491 | Energia Elétrica | B3 |
| PIOR_SEMANA_W2b | M0 | 2022-05-30 00:00:00 | GPAR3 | 0.052492 | 0.129019 | Energia Elétrica | B3 |
| PIOR_SEMANA_W3 | M0 | 2025-04-14 00:00:00 | SOND6 | 0.122118 | 0.260013 | Engenharia Consultiva | B3 |
| PIOR_SEMANA_W3 | M0 | 2025-04-14 00:00:00 | P2NB34 | 0.120009 | 0.255521 | Equipamentos Médicos | BDR |
| PIOR_SEMANA_W3 | M0 | 2025-04-14 00:00:00 | ITUB4 | 0.100000 | 0.212918 | Bancos | B3 |
| PIOR_SEMANA_W3 | M0 | 2025-04-14 00:00:00 | CALI3 | 0.099361 | 0.211558 | Incorporações | B3 |
| PIOR_SEMANA_W3 | M0 | 2025-04-14 00:00:00 | P2CY34 | 0.095581 | 0.203509 | Consultoria | BDR |
| PIOR_SEMANA_W3 | M0 | 2025-04-14 00:00:00 | ITUB3 | 0.090911 | 0.193566 | Bancos | B3 |
| PIOR_SEMANA_W3 | M0 | 2025-04-14 00:00:00 | J2AZ34 | 0.090524 | 0.192741 | Farmacêutico | BDR |
| PIOR_SEMANA_W3 | M0 | 2025-04-14 00:00:00 | EKTR3 | 0.089454 | 0.190465 | Energia Elétrica | B3 |
| PIOR_SEMANA_W3 | M0 | 2025-04-14 00:00:00 | AZTE3 | 0.075453 | 0.160653 | Exploração. Refino e Distribuição | B3 |
| PIOR_SEMANA_W3 | M0 | 2025-04-14 00:00:00 | P2OD34 | 0.036819 | 0.078394 | Equipamentos Médicos | BDR |
| PIOR_SEMANA_W2a | M1 | 2021-09-20 00:00:00 | M1RN34 | 0.128186 | 0.516675 | Farmacêutico | BDR |
| PIOR_SEMANA_W2a | M1 | 2021-09-20 00:00:00 | ENMT3 | 0.102074 | 0.411428 | Energia Elétrica | B3 |
| PIOR_SEMANA_W2a | M1 | 2021-09-20 00:00:00 | MOAR3 | 0.100000 | 0.403068 | Holdings Diversificadas | B3 |
| PIOR_SEMANA_W2a | M1 | 2021-09-20 00:00:00 | HETA4 | 0.100000 | 0.403068 | Utensílios Domésticos | B3 |
| PIOR_SEMANA_W2a | M1 | 2021-09-20 00:00:00 | HBTS5 | 0.095786 | 0.386081 | Exploração de Imóveis | B3 |
| PIOR_SEMANA_W2a | M1 | 2021-09-20 00:00:00 | D1OC34 | 0.092001 | 0.370827 | Tecnologia | BDR |
| PIOR_SEMANA_W2a | M1 | 2021-09-20 00:00:00 | BRBI11 | 0.091413 | 0.368457 | Bancos | B3 |
| PIOR_SEMANA_W2a | M1 | 2021-09-20 00:00:00 | MWET4 | 0.077991 | 0.314357 | Material Rodoviário | B3 |
| PIOR_SEMANA_W2a | M1 | 2021-09-20 00:00:00 | PSVM11 | 0.072103 | 0.290623 | Serviços de Apoio e Armazenagem | B3 |
| PIOR_SEMANA_W2a | M1 | 2021-09-20 00:00:00 | BRAV3 | 0.071828 | 0.289517 | Exploração. Refino e Distribuição | B3 |
| PIOR_SEMANA_W2b | M1 | 2022-06-27 00:00:00 | ENMT3 | 0.180317 | 0.449382 | Energia Elétrica | B3 |
| PIOR_SEMANA_W2b | M1 | 2022-06-27 00:00:00 | BEEF3 | 0.114625 | 0.285666 | Carnes e Derivados | B3 |
| PIOR_SEMANA_W2b | M1 | 2022-06-27 00:00:00 | DOTZ3 | 0.111170 | 0.277056 | Programas de Fidelização | B3 |
| PIOR_SEMANA_W2b | M1 | 2022-06-27 00:00:00 | PSVM11 | 0.110083 | 0.274348 | Serviços de Apoio e Armazenagem | B3 |
| PIOR_SEMANA_W2b | M1 | 2022-06-27 00:00:00 | LAND3 | 0.102696 | 0.255936 | Agricultura | B3 |
| PIOR_SEMANA_W2b | M1 | 2022-06-27 00:00:00 | VULC3 | 0.100938 | 0.251555 | Calçados | B3 |
| PIOR_SEMANA_W2b | M1 | 2022-06-27 00:00:00 | TRAD3 | 0.100000 | 0.249218 | Programas e Serviços | B3 |
| PIOR_SEMANA_W2b | M1 | 2022-06-27 00:00:00 | SBSP3 | 0.097112 | 0.242020 | Água e Saneamento | B3 |
| PIOR_SEMANA_W2b | M1 | 2022-06-27 00:00:00 | AXIA6 | 0.083060 | 0.207001 | Energia Elétrica | B3 |
| PIOR_SEMANA_W3 | M1 | 2025-11-17 00:00:00 | IFCM3 | 0.100000 | 0.369332 | Programas e Serviços | B3 |
| PIOR_SEMANA_W3 | M1 | 2025-11-17 00:00:00 | RCSL4 | 0.100000 | 0.369332 | Material Rodoviário | B3 |
| PIOR_SEMANA_W3 | M1 | 2025-11-17 00:00:00 | DESK3 | 0.100000 | 0.369332 | Telecomunicações | B3 |
| PIOR_SEMANA_W3 | M1 | 2025-11-17 00:00:00 | DASA3 | 0.100000 | 0.369332 | Serv.Méd.Hospit..Análises e Diagnósticos | B3 |
| PIOR_SEMANA_W3 | M1 | 2025-11-17 00:00:00 | VVEO3 | 0.100000 | 0.369332 | Medicamentos e Outros Produtos | B3 |
| PIOR_SEMANA_W3 | M1 | 2025-11-17 00:00:00 | A2RR34 | 0.100000 | 0.369332 | Pacífico Asiático, Ex Japão | BDR |
| PIOR_SEMANA_W3 | M1 | 2025-11-17 00:00:00 | MUTC34 | 0.100000 | 0.369332 | Tecnologia | BDR |
| PIOR_SEMANA_W3 | M1 | 2025-11-17 00:00:00 | W1BD34 | 0.100000 | 0.369332 | Comunicação | BDR |
| PIOR_SEMANA_W3 | M1 | 2025-11-17 00:00:00 | ITUB3 | 0.060077 | 0.221883 | Bancos | B3 |
| PIOR_SEMANA_W3 | M1 | 2025-11-17 00:00:00 | ITUB4 | 0.045896 | 0.169510 | Bancos | B3 |

## 4) Auditoria de proteção (W2a e W2b)

| window | mechanism | avg_turnover_ratio | avg_cash_ratio | avg_n_positions | sold_count | trigger_i_lcl_count | trigger_i_ucl_count | trigger_mr_ucl_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W2a_drawdown_aug21 | M0 | 0.054311 | 0.068334 | 10.116279 | 14 | 4 | 0 | 7 |
| W2a_drawdown_aug21 | M1 | 0.112448 | 0.161678 | 10.069767 | 29 | 8 | 0 | 19 |
| W2b_drawdown_may22 | M0 | 0.008738 | 0.041258 | 10.441860 | 3 | 1 | 0 | 2 |
| W2b_drawdown_may22 | M1 | 0.067033 | 0.223530 | 7.395349 | 16 | 7 | 0 | 11 |

## 5) Conclusão curta (com referência de tabela)

- W2a/W2b mostram diferença operacional de proteção entre M0 e M1 em `data/audit_protecao_resumo.parquet`.
- Drivers por ticker por janela estão em `data/top_tickers_W2a_M0.parquet`, `data/top_tickers_W2a_M1.parquet`, `data/top_tickers_W2b_M0.parquet`, `data/top_tickers_W2b_M1.parquet`.
- Drivers por setor/asset_class por janela estão em `data/top_setor_assetclass_W2a.parquet` e `data/top_setor_assetclass_W2b.parquet`.
- Composição em datas sentinela está em `data/snapshots_sentinela_holdings.parquet`.
- Definição de M3 foi extraída de `docs/emendas/PROPOSTA_M3_CEP_COMPRA_V1_4_NAO_INTEGRADA.md` e métricas comparativas de `../task_015_m3/.../metricas_consolidadas.parquet`.
