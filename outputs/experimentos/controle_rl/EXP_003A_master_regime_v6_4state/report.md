# EXP_003A Master Regime V6 4state

## OVERALL
- OVERALL PASS

## STEPS
- S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY: PASS
- S2_CHECK_COMPILE_OR_IMPORTS: PASS
- S3_BUILD_CLOSE_AND_THEORY_LABELS_4STATE_BVSP_GSPC: PASS
- S4_BUILD_CEP_FEATURES_EXOG_BVSP_GSPC: PASS
- S5_FIT_INTERPRETABLE_MODEL_ON_BVSP_THRESHOLD_SEARCH: PASS
- S6_VALIDATE_ON_GSPC_NO_REFIT: PASS
- S7_VERIFY_4STATES_REACHABLE_AND_NEUTRAL_OR_CORRECAO_DOMINANT: PASS
- S8_WRITE_SSOT_V6_EXPLICIT_FORMULA_PSEUDOCODE_AND_BUY_MAPPING: PASS
- S9_ANTI_LEAKAGE_AUDIT_DMINUS1: PASS
- S10_GENERATE_MD_AUTOCONTIDO_MANIFEST_HASHES: PASS

## Definicao formal do criterio teorico (4 estados) + pseudocodigo
- `peak_t=max(Close_{t-252..t})`; `trough_t=min(Close_{t-252..t})`
- `drawdown_t=Close_t/peak_t-1`; `rise_t=Close_t/trough_t-1`
- BEAR se drawdown<=-0.20
- CORRECAO se -0.20<drawdown<=-0.10
- BULL se rise>=+0.20
- NEUTRO caso contrário
- aplicação causal com histerese/min_days

## Distribuicoes por classe
- calibração (^BVSP): `{'CORRECAO': 924, 'BULL': 463, 'BEAR': 255, 'NEUTRO': 221}`
- validação (^GSPC): `{'CORRECAO': 1194, 'BEAR': 337, 'BULL': 268, 'NEUTRO': 64}`

## Matriz de confusao com contagens
### calibração (^BVSP)
```json
{
  "BULL": {
    "BULL": 226,
    "BEAR": 88,
    "CORRECAO": 663,
    "NEUTRO": 133
  },
  "BEAR": {
    "BULL": 12,
    "BEAR": 122,
    "CORRECAO": 14,
    "NEUTRO": 31
  },
  "CORRECAO": {
    "BULL": 118,
    "BEAR": 9,
    "CORRECAO": 108,
    "NEUTRO": 39
  },
  "NEUTRO": {
    "BULL": 107,
    "BEAR": 36,
    "CORRECAO": 139,
    "NEUTRO": 18
  }
}
```
### validação (^GSPC)
```json
{
  "BULL": {
    "BULL": 223,
    "BEAR": 55,
    "CORRECAO": 980,
    "NEUTRO": 60
  },
  "BEAR": {
    "BULL": 7,
    "BEAR": 214,
    "CORRECAO": 0,
    "NEUTRO": 0
  },
  "CORRECAO": {
    "BULL": 0,
    "BEAR": 55,
    "CORRECAO": 64,
    "NEUTRO": 0
  },
  "NEUTRO": {
    "BULL": 38,
    "BEAR": 13,
    "CORRECAO": 150,
    "NEUTRO": 4
  }
}
```

## Metricas
| conjunto | macro_f1_4state | balanced_accuracy_4state | switches_per_year_4state |
|---|---:|---:|---:|
| ^BVSP | 0.274740 | 0.334832 | 6.222222 |
| ^GSPC | 0.293866 | 0.423712 | 4.734300 |

## Mapa 4state->BUY0/BUY1/BUY2
- `{'BULL': 'BUY2', 'BEAR': 'BUY0', 'CORRECAO': 'BUY1', 'NEUTRO': 'BUY1'}`

## Auditoria anti-leakage D-1 (amostra 30 datas)
| index_ticker | D | last_input_date_used | execution_price_date | ok_dminus1 |
|---|---|---|---|---|
| ^BVSP | 2018-07-04 | 2018-07-03 | 2018-07-04 | True |
| ^BVSP | 2019-01-18 | 2019-01-17 | 2019-01-18 | True |
| ^BVSP | 2019-08-02 | 2019-08-01 | 2019-08-02 | True |
| ^BVSP | 2020-02-13 | 2020-02-12 | 2020-02-13 | True |
| ^BVSP | 2020-08-26 | 2020-08-25 | 2020-08-26 | True |
| ^BVSP | 2021-03-16 | 2021-03-15 | 2021-03-16 | True |
| ^BVSP | 2021-09-24 | 2021-09-23 | 2021-09-24 | True |
| ^BVSP | 2022-04-08 | 2022-04-07 | 2022-04-08 | True |
| ^BVSP | 2022-10-19 | 2022-10-18 | 2022-10-19 | True |
| ^BVSP | 2023-05-04 | 2023-05-03 | 2023-05-04 | True |
| ^BVSP | 2023-11-13 | 2023-11-10 | 2023-11-13 | True |
| ^BVSP | 2024-05-28 | 2024-05-27 | 2024-05-28 | True |
| ^BVSP | 2024-12-04 | 2024-12-03 | 2024-12-04 | True |
| ^BVSP | 2025-06-23 | 2025-06-20 | 2025-06-23 | True |
| ^BVSP | 2025-12-30 | 2025-12-29 | 2025-12-30 | True |
| ^GSPC | 2018-07-04 | 2018-07-03 | 2018-07-04 | True |
| ^GSPC | 2019-01-18 | 2019-01-17 | 2019-01-18 | True |
| ^GSPC | 2019-08-02 | 2019-08-01 | 2019-08-02 | True |
| ^GSPC | 2020-02-13 | 2020-02-12 | 2020-02-13 | True |
| ^GSPC | 2020-08-26 | 2020-08-25 | 2020-08-26 | True |
| ^GSPC | 2021-03-16 | 2021-03-15 | 2021-03-16 | True |
| ^GSPC | 2021-09-24 | 2021-09-23 | 2021-09-24 | True |
| ^GSPC | 2022-04-08 | 2022-04-07 | 2022-04-08 | True |
| ^GSPC | 2022-10-19 | 2022-10-18 | 2022-10-19 | True |
| ^GSPC | 2023-05-04 | 2023-05-03 | 2023-05-04 | True |
| ^GSPC | 2023-11-13 | 2023-11-10 | 2023-11-13 | True |
| ^GSPC | 2024-05-28 | 2024-05-27 | 2024-05-28 | True |
| ^GSPC | 2024-12-04 | 2024-12-03 | 2024-12-04 | True |
| ^GSPC | 2025-06-23 | 2025-06-20 | 2025-06-23 | True |
| ^GSPC | 2025-12-30 | 2025-12-29 | 2025-12-30 | True |

## Inputs resolvidos
- source_info: `{"method": "autodiscovery", "selected": "/home/wilson/CEP_COMPRA/outputs/backtests/task_012/run_20260212_114129/consolidated/series_alinhadas_plot.parquet", "score": 7, "selected_columns": ["date", "M0_equity_idx", "M1_equity_idx", "cdi_index", "sp500_index", "bvsp_index", "cdi_index_norm", "sp500_index_norm", "bvsp_index_norm"], "candidates_found": 6}`

## Artefatos
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/close_bvsp.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/close_gspc.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/labels_theory_4state_bvsp.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/labels_theory_4state_gspc.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/cep_features_bvsp.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/cep_features_gspc.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/model_fit_summary.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/threshold_search_results.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/regime_daily_bvsp_4state.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/regime_daily_gspc_4state.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/buy_level_daily_bvsp.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/buy_level_daily_gspc.parquet`
- `/home/wilson/CEP_COMPRA/ssot_cycle2/master_regime_classifier_v6.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/report.md`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/manifest.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003A_master_regime_v6_4state/hashes.sha256`
