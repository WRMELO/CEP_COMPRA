# Pacote forense ampliado M0/M1/M3

## Escopo e causalidade
- Todas as afirmações abaixo referenciam tabelas Parquet deste pacote.
- Janela canônica validada: 2018-07-01..2025-12-31.

## W1 (2018-07-01..2021-06-30)
- M0 top contribuintes: A1LG34(0.2840), FHER3(0.1952), INEP3(0.1660)
- M0 top detratores: MAPT4(-0.1335), GPAR3(-0.1252), NUTR3(-0.1108)
- M1 top contribuintes: INEP3(0.5584), RCSL3(0.5069), FHER3(0.4804)
- M1 top detratores: DASA3(-0.2692), NGRD3(-0.1029), VIVR3(-0.0906)
- M3 top contribuintes: FHER3(0.5870), INEP3(0.5567), LWSA3(0.4811)
- M3 top detratores: DASA3(-0.2697), MGLU3(-0.1215), PSVM11(-0.1080)
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.

## W2 (2021-07-01..2022-12-31)
- M0 top contribuintes: RSUL4(0.4122), ENMT4(0.1065), CRPG3(0.0726)
- M0 top detratores: GPAR3(-0.1339), MWET4(-0.1099), COCE3(-0.1079)
- M1 top contribuintes: HAGA3(0.0816), GFSA3(0.0727), RSUL4(0.0459)
- M1 top detratores: FHER3(-0.2126), CASH3(-0.2110), MWET4(-0.2035)
- M3 top contribuintes: M1RN34(0.4154), HAGA3(0.2424), MOSC34(0.0853)
- M3 top detratores: VAMO3(-0.4191), KEPL3(-0.2405), MWET4(-0.2122)
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.

## W3 (2024-09-01..2025-11-30)
- M0 top contribuintes: CCTY3(0.2076), D2OC34(0.0739), E2XE34(0.0716)
- M0 top detratores: B2RK34(-0.0905), M2KS34(-0.0889), GFSA3(-0.0827)
- M1 top contribuintes: IFCM3(2.3551), AMBP3(0.2736), A2RR34(0.1437)
- M1 top detratores: FICT3(-0.1647), ENMT3(-0.1408), GFSA3(-0.1170)
- M3 top contribuintes: AMBP3(0.3625), ITUB3(0.0871), CASH3(0.0603)
- M3 top detratores: MNPR3(-0.2103), ENMT3(-0.1624), SBFG3(-0.0924)
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.

## queda_agosto_2021 (2021-08-01..2021-08-31)
- M0 top contribuintes: RSUL4(0.1771), ENMT4(0.0583), PATI3(0.0268)
- M0 top detratores: CRPG3(-0.0629), MWET4(-0.0420), COCE3(-0.0346)
- M1 top contribuintes: RSUL4(0.0459), MNDL3(0.0359), B1NT34(0.0317)
- M1 top detratores: CASH3(-0.1394), MGEL4(-0.1209), PLAS3(-0.1019)
- M3 top contribuintes: M1RN34(0.2056), BRKM5(0.0870), RSUL4(0.0422)
- M3 top detratores: VAMO3(-0.5048), MGEL4(-0.1191), CASH3(-0.0860)
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.

## queda_maio_2022 (2022-05-01..2022-05-31)
- M0 top contribuintes: ENMT4(0.0005), I2NG34(-0.0001), OXYP34(-0.0010)
- M0 top detratores: SOND5(-0.0388), SOND6(-0.0242), LPSB3(-0.0207)
- M1 top contribuintes: BEEF3(0.0308), DOTZ3(0.0140), OXYP34(0.0097)
- M1 top detratores: KEPL3(-0.2222), CBAV3(-0.0755), PSVM11(-0.0506)
- M3 top contribuintes: BEEF3(0.0369), OXYP34(0.0150), RECV3(0.0113)
- M3 top detratores: KEPL3(-0.2735), CBAV3(-0.0894), LPSB3(-0.0529)
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.

## salto_2025_11_07 (2025-11-07..2025-11-07)
- M0 top contribuintes: ITUB4(0.0002), ITUB3(-0.0003), P2ST34(-0.0099)
- M0 top detratores: CEEB5(-0.0209), TKNO4(-0.0131), P2ST34(-0.0099)
- M1 top contribuintes: IFCM3(2.3501), SEER3(0.0057), DOTZ3(0.0002)
- M1 top detratores: MYPK3(-0.0032), BKCH39(-0.0020), ITUB3(-0.0003)
- M3 top contribuintes: SEER3(0.0084), CEAB3(0.0022), ITUB4(0.0003)
- M3 top detratores: MYPK3(-0.0048), SUZB3(-0.0043), QUAL3(-0.0038)
- Evidência: `data/decomp_pnl_windows.parquet` e/ou `data/decomp_pnl_event_months.parquet`.

## Diferenças operacionais (giro/proteção)
- M0: avg_cash=0.0650, avg_n_positions=10.84, max_drawdown=-0.3578
- M1: avg_cash=0.1369, avg_n_positions=9.04, max_drawdown=-0.7811
- M3: avg_cash=0.1291, avg_n_positions=9.49, max_drawdown=-0.6803
- Evidência: `data/master_state_daily.parquet`, `data/ledger_trades_*.parquet`, `data/holdings_weekly_*.parquet`.
