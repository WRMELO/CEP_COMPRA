# EXP_002A Master Regime V4C (autocontido)

## OVERALL
- OVERALL PASS

## STEPS
- S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY: PASS
- S2_WRITE_AND_PARSE_MONTHLY_TABLE: PASS
- S3_BUILD_MASTER_CLOSE_LOGRET_AND_CEP_FEATURES_EXOG: PASS
- S4_BUILD_STRUCTURED_MONTHLY_AND_DAILY_WEAKLABELS: PASS
- S5_VALIDATE_PERIODS_NUMERICALLY_AND_ASSIGN_WEIGHTS: PASS
- S6_FIT_INTERPRETABLE_MODEL_AND_THRESHOLD_SEARCH: PASS
- S7_VERIFY_BULL_AND_BEAR_REACHABLE_DAILY: PASS
- S8_WRITE_SSOT_V4_WITH_EXPLICIT_FORMULA: PASS
- S9_GENERATE_MD_AUTOCONTIDO_AND_MANIFEST_HASHES: PASS

## Tabela fonte (exata)
| Período (mm/aa–mm/aa) | Tipo  | Duração Aprox. | Ganho/Queda Pico-Low | Motivos Principais |
|------------------------|-------|----------------|----------------------|--------------------|
| 01/19–12/19           | Bull | 12 meses       | +31,58%              | Reformas (especialmente Previdência), juros baixos, recuperação pós-eleições. |
| 10/22–12/23           | Bull | 14 meses       | +22,28%              | Queda da Selic, fortalecimento de commodities, entrada de capital estrangeiro. |
| 01/25–12/25           | Bull | 12 meses       | +34%                 | PIB acima do esperado, desemprego em queda, inflação controlada, fluxo de capitais para Brasil. |
| 04/20–12/21           | Bear | 20 meses       | >-20% no período (-11,93% no ano) | Prolongamento da pandemia, incerteza fiscal, ciclo de alta de juros. |
| 01/22–10/22           | Bear | 10 meses       | Queda acumulada até low ~90k | Selic em forte alta, inflação global, risco fiscal doméstico. |
| 08/24–12/24           | Bear | 5 meses        | -10% no ano          | Piora das expectativas econômicas, saída de investidores estrangeiros, aversão a risco. |

## Parse mensal estruturado (6 linhas)
| period_mm_aa_range | tipo_raw | tipo | start_month | start_year | end_month | end_year | duracao_aprox | ganho_queda_texto | motivos | start_date | end_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01/19–12/19 | Bull | BULL | 1 | 2019 | 12 | 2019 | 12 meses | +31,58% | Reformas (especialmente Previdência), juros baixos, recuperação pós-eleições. | 2019-01-02 | 2019-12-30 |
| 10/22–12/23 | Bull | BULL | 10 | 2022 | 12 | 2023 | 14 meses | +22,28% | Queda da Selic, fortalecimento de commodities, entrada de capital estrangeiro. | 2022-10-03 | 2023-12-28 |
| 01/25–12/25 | Bull | BULL | 1 | 2025 | 12 | 2025 | 12 meses | +34% | PIB acima do esperado, desemprego em queda, inflação controlada, fluxo de capitais para Brasil. | 2025-01-02 | 2025-12-30 |
| 04/20–12/21 | Bear | BEAR | 4 | 2020 | 12 | 2021 | 20 meses | >-20% no período (-11,93% no ano) | Prolongamento da pandemia, incerteza fiscal, ciclo de alta de juros. | 2020-04-01 | 2021-12-30 |
| 01/22–10/22 | Bear | BEAR | 1 | 2022 | 10 | 2022 | 10 meses | Queda acumulada até low ~90k | Selic em forte alta, inflação global, risco fiscal doméstico. | 2022-01-03 | 2022-10-31 |
| 08/24–12/24 | Bear | BEAR | 8 | 2024 | 12 | 2024 | 5 meses | -10% no ano | Piora das expectativas econômicas, saída de investidores estrangeiros, aversão a risco. | 2024-08-01 | 2024-12-30 |

## Validação numérica por período (Close)
| period_mm_aa_range | tipo | start_date | end_date | close_start | close_end | period_return | period_drawdown | validation_pass | sample_weight | validation_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01/19–12/19 | BULL | 2019-01-02 | 2019-12-30 | 91012.000000 | 115964.000000 | 0.274162 | -0.100016 | True | 1.000000 | pass_bull_gain |
| 10/22–12/23 | BULL | 2022-10-03 | 2023-12-28 | 116134.000000 | 134185.000000 | 0.155433 | -0.183467 | False | 0.250000 | fail_bull_gain |
| 01/25–12/25 | BULL | 2025-01-02 | 2025-12-30 | 120125.000000 | 161125.000000 | 0.341311 | -0.069223 | True | 1.000000 | pass_bull_gain |
| 04/20–12/21 | BEAR | 2020-04-01 | 2021-12-30 | 70967.000000 | 104822.000000 | 0.477053 | -0.229408 | True | 1.000000 | pass_bear_drawdown |
| 01/22–10/22 | BEAR | 2022-01-03 | 2022-10-31 | 103922.000000 | 116037.000000 | 0.116578 | -0.209336 | True | 1.000000 | pass_bear_drawdown |
| 08/24–12/24 | BEAR | 2024-08-01 | 2024-12-30 | 127395.000000 | 120283.000000 | -0.055826 | -0.124323 | False | 0.250000 | fail_bear_drawdown |

## Modelo interpretável
- chosen_model: **shallow_tree**
- validation_metrics_best: `{"balanced_accuracy": 0.5652472527472527, "auc": 0.5252808988764045, "bull": {"precision": 0.7950617283950617, "recall": 0.9044943820224719, "f1": 0.8462549277266754}, "bear": {"precision": 0.38181818181818183, "recall": 0.20192307692307693, "f1": 0.2641509433962264}}`
- selection_score: `score_final = balanced_accuracy_validation - 0.05 * misto_rate_validation`

## Histerese / min_days / limiares
- best_threshold_config: `{"p_bull_enter": 0.55, "p_bull_exit_delta": 0.01, "p_bear_enter": 0.45, "p_bear_exit_delta": 0.01, "min_days": 5, "balanced_accuracy_validation": 0.5652472527472527, "misto_rate_validation": 0.0, "score_final": 0.5652472527472527}`

## Distribuição diária de regimes
- `{'BULL': 1232, 'BEAR': 624, 'MISTO': 7}`
- bull/bear alcançáveis (>0): **True**

## Inputs resolvidos
- parse_contract: `{"required_columns": ["period_mm_aa_range", "tipo"], "period_format": "MM/YY–MM/YY", "tipo_map": {"Bull": "BULL", "Bear": "BEAR"}, "expected_rows": 6, "date_interpretation_rule": {"start_month": "primeiro pregão do mês inicial", "end_month": "último pregão do mês final", "inclusive": true}}`
- master_series_source: `{"path": "/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/master_close_daily.parquet", "method": "autodiscover_globs", "candidates_found": 4, "accepted_count": 4, "selected": "/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/master_close_daily.parquet", "selected_columns": ["close", "date", "master_ticker"]}`

## Artefatos
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/monthly_ssot_source.md`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/monthly_regime_structured.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/monthly_regime_structured.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/master_close_daily.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/master_logret_daily.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/master_cep_features_daily.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/weaklabels_daily_audit.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/period_numeric_validation.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/ssot_cycle2/master_regime_classifier_v4.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/regime_daily.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/regime_daily.csv`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/report.md`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/manifest.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4c/hashes.sha256`
