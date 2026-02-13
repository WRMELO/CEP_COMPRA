# Pacote forense ampliado M0/M1/M3

## Escopo e causalidade
- Todas as afirmações abaixo referenciam tabelas Parquet deste pacote.
- Janela canônica validada: 2018-07-01..2025-12-31.

## W1 (2018-07-01..2021-06-30)
- M0 top contribuintes: A1LG34(0.2840), FHER3(0.1952), INEP3(0.1660)
- M0 top detratores: MAPT4(-0.1335), GPAR3(-0.1252), NUTR3(-0.1108)
- M0 risco/execução: turnover=0.0584, cash_ratio=0.1161, n_positions=9.99, buys=295, sells=253
- M0 decisões de venda dominantes: MR_UCL:85, I_LCL:80
- M1 top contribuintes: INEP3(0.5584), RCSL3(0.5069), FHER3(0.4804)
- M1 top detratores: DASA3(-0.2692), NGRD3(-0.1029), VIVR3(-0.0906)
- M1 risco/execução: turnover=0.0902, cash_ratio=0.1703, n_positions=8.93, buys=493, sells=383
- M1 decisões de venda dominantes: MR_UCL:169, I_LCL:92
- M3 top contribuintes: FHER3(0.5870), INEP3(0.5567), LWSA3(0.4811)
- M3 top detratores: DASA3(-0.2697), MGLU3(-0.1215), PSVM11(-0.1080)
- M3 risco/execução: turnover=0.0971, cash_ratio=0.1741, n_positions=8.89, buys=489, sells=392
- M3 decisões de venda dominantes: MR_UCL:175, I_LCL:89
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.
- Evidência operacional: `data/ledger_trades_*.parquet`, `data/daily_portfolio_*.parquet`.

## W2 (2021-07-01..2022-12-31)
- M0 top contribuintes: RSUL4(0.4122), ENMT4(0.1065), CRPG3(0.0726)
- M0 top detratores: GPAR3(-0.1339), MWET4(-0.1099), COCE3(-0.1079)
- M0 risco/execução: turnover=0.0227, cash_ratio=0.0339, n_positions=10.91, buys=65, sells=53
- M0 decisões de venda dominantes: MR_UCL:29, I_LCL:18
- M1 top contribuintes: HAGA3(0.0816), GFSA3(0.0727), RSUL4(0.0459)
- M1 top detratores: FHER3(-0.2126), CASH3(-0.2110), MWET4(-0.2035)
- M1 risco/execução: turnover=0.0722, cash_ratio=0.1267, n_positions=9.08, buys=184, sells=152
- M1 decisões de venda dominantes: MR_UCL:78, I_LCL:58
- M3 top contribuintes: M1RN34(0.4154), HAGA3(0.2424), MOSC34(0.0853)
- M3 top detratores: VAMO3(-0.4191), KEPL3(-0.2405), MWET4(-0.2122)
- M3 risco/execução: turnover=0.0754, cash_ratio=0.1299, n_positions=9.33, buys=191, sells=158
- M3 decisões de venda dominantes: MR_UCL:80, I_LCL:58
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.
- Evidência operacional: `data/ledger_trades_*.parquet`, `data/daily_portfolio_*.parquet`.

## W3 (2024-09-01..2025-11-30)
- M0 top contribuintes: CCTY3(0.2076), D2OC34(0.0739), E2XE34(0.0716)
- M0 top detratores: B2RK34(-0.0905), M2KS34(-0.0889), GFSA3(-0.0827)
- M0 risco/execução: turnover=0.0164, cash_ratio=0.0228, n_positions=12.85, buys=44, sells=40
- M0 decisões de venda dominantes: MR_UCL:20, I_LCL:17
- M1 top contribuintes: IFCM3(2.3551), AMBP3(0.2736), A2RR34(0.1437)
- M1 top detratores: FICT3(-0.1647), ENMT3(-0.1408), GFSA3(-0.1170)
- M1 risco/execução: turnover=0.0761, cash_ratio=0.1231, n_positions=8.11, buys=150, sells=119
- M1 decisões de venda dominantes: I_LCL:55, MR_UCL:48
- M3 top contribuintes: AMBP3(0.3625), ITUB3(0.0871), CASH3(0.0603)
- M3 top detratores: MNPR3(-0.2103), ENMT3(-0.1624), SBFG3(-0.0924)
- M3 risco/execução: turnover=0.0568, cash_ratio=0.1051, n_positions=9.99, buys=122, sells=106
- M3 decisões de venda dominantes: MR_UCL:48, I_LCL:45
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.
- Evidência operacional: `data/ledger_trades_*.parquet`, `data/daily_portfolio_*.parquet`.

## queda_agosto_2021 (2021-08-01..2021-08-31)
- M0 top contribuintes: RSUL4(0.1771), ENMT4(0.0583), PATI3(0.0268)
- M0 top detratores: CRPG3(-0.0629), MWET4(-0.0420), COCE3(-0.0346)
- M0 risco/execução: turnover=0.0712, cash_ratio=0.0897, n_positions=9.82, buys=10, sells=8
- M0 decisões de venda dominantes: STRESS_AMP:4, MR_UCL:2
- M1 top contribuintes: RSUL4(0.0459), MNDL3(0.0359), B1NT34(0.0317)
- M1 top detratores: CASH3(-0.1394), MGEL4(-0.1209), PLAS3(-0.1019)
- M1 risco/execução: turnover=0.1320, cash_ratio=0.1480, n_positions=10.45, buys=19, sells=16
- M1 decisões de venda dominantes: STRESS_AMP:6, MR_UCL:5
- M3 top contribuintes: M1RN34(0.2056), BRKM5(0.0870), RSUL4(0.0422)
- M3 top detratores: VAMO3(-0.5048), MGEL4(-0.1191), CASH3(-0.0860)
- M3 risco/execução: turnover=0.1152, cash_ratio=0.1515, n_positions=9.05, buys=16, sells=14
- M3 decisões de venda dominantes: STRESS_AMP:6, MR_UCL:4
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.
- Evidência operacional: `data/ledger_trades_*.parquet`, `data/daily_portfolio_*.parquet`.

## queda_maio_2022 (2022-05-01..2022-05-31)
- M0 top contribuintes: ENMT4(0.0005), I2NG34(-0.0001), OXYP34(-0.0010)
- M0 top detratores: SOND5(-0.0388), SOND6(-0.0242), LPSB3(-0.0207)
- M0 risco/execução: turnover=0.0143, cash_ratio=0.0751, n_positions=10.00, buys=3, sells=2
- M0 decisões de venda dominantes: I_LCL:1, MR_UCL:1
- M1 top contribuintes: BEEF3(0.0308), DOTZ3(0.0140), OXYP34(0.0097)
- M1 top detratores: KEPL3(-0.2222), CBAV3(-0.0755), PSVM11(-0.0506)
- M1 risco/execução: turnover=0.1050, cash_ratio=0.3580, n_positions=6.23, buys=15, sells=11
- M1 decisões de venda dominantes: MR_UCL:6, I_LCL:5
- M3 top contribuintes: BEEF3(0.0369), OXYP34(0.0150), RECV3(0.0113)
- M3 top detratores: KEPL3(-0.2735), CBAV3(-0.0894), LPSB3(-0.0529)
- M3 risco/execução: turnover=0.1216, cash_ratio=0.3503, n_positions=6.23, buys=19, sells=15
- M3 decisões de venda dominantes: MR_UCL:8, I_LCL:6
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.
- Evidência operacional: `data/ledger_trades_*.parquet`, `data/daily_portfolio_*.parquet`.

## salto_2025_11_07 (2025-11-07..2025-11-07)
- M0 top contribuintes: ITUB4(0.0002), ITUB3(-0.0003), P2ST34(-0.0099)
- M0 top detratores: CEEB5(-0.0209), TKNO4(-0.0131), P2ST34(-0.0099)
- M0 risco/execução: turnover=0.0000, cash_ratio=0.0432, n_positions=10.00, buys=0, sells=0
- M1 top contribuintes: IFCM3(2.3501), SEER3(0.0057), DOTZ3(0.0002)
- M1 top detratores: MYPK3(-0.0032), BKCH39(-0.0020), ITUB3(-0.0003)
- M1 risco/execução: turnover=0.0000, cash_ratio=0.0000, n_positions=8.00, buys=0, sells=0
- M3 top contribuintes: SEER3(0.0084), CEAB3(0.0022), ITUB4(0.0003)
- M3 top detratores: MYPK3(-0.0048), SUZB3(-0.0043), QUAL3(-0.0038)
- M3 risco/execução: turnover=0.0000, cash_ratio=0.0000, n_positions=9.00, buys=0, sells=0
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.
- Evidência operacional: `data/ledger_trades_*.parquet`, `data/daily_portfolio_*.parquet`.

## Diferenças operacionais (giro/proteção)
- M0: avg_cash=0.0650, avg_n_positions=10.84, max_drawdown=-0.3578
- M1: avg_cash=0.1369, avg_n_positions=9.04, max_drawdown=-0.7811
- M3: avg_cash=0.1291, avg_n_positions=9.49, max_drawdown=-0.6803
- Evidência: `data/master_state_daily.parquet`, `data/ledger_trades_*.parquet`, `data/holdings_weekly_*.parquet`.
