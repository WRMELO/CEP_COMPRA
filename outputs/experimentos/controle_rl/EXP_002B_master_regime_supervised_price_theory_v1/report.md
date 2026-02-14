# EXP_002B Master Regime Supervised Price-Theory v1

## OVERALL
- OVERALL PASS

## STEPS
- S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY: PASS
- S2_CHECK_COMPILE_OR_IMPORTS: PASS
- S3_BUILD_PRICE_THEORY_LABELS_BVSP_AND_GSPC: PASS
- S4_BUILD_CEP_FEATURES_EXOG_BVSP_AND_GSPC: PASS
- S5_FIT_INTERPRETABLE_MODEL_ON_BVSP_AND_THRESHOLD_SEARCH: PASS
- S6_VALIDATE_ON_GSPC_NO_REFIT: PASS
- S7_VERIFY_THREE_STATES_REACHABLE_AND_NEUTRAL_DOMINANT: PASS
- S8_WRITE_SSOT_V5_EXPLICIT_FORMULA_PSEUDOCODE: PASS
- S9_GENERATE_MD_AUTOCONTIDO_MANIFEST_HASHES: PASS

## Critério Teórico Formal (Price-Only)
- `drawdown_from_peak = Close_t / rolling_max_252 - 1`
- `rise_from_trough = Close_t / rolling_min_252 - 1`
- bull_threshold = 0.2, bear_threshold = -0.2, neutral_band = 0.1
- raw labels: BULL se rise>=bull_threshold; BEAR se drawdown<=bear_threshold; CORRECAO se drawdown<=-neutral_band; NEUTRAL caso contrário
- target 3-state: BULL/BEAR/MISTO (CORRECAO+NEUTRAL => MISTO)

## Pseudocódigo de Regime Operacional
```text
p_bull = modelo_CEP_only(features_cep)
if p_bull >= p_bull_enter -> candidato BULL
elif p_bull <= p_bear_enter -> candidato BEAR
else -> candidato MISTO
aplicar histerese/min_days para confirmar troca
```

## Distribuição de Classes
| conjunto | distribuição |
|---|---|
| calibração (^BVSP) | `{'MISTO': 768, 'BULL': 767, 'BEAR': 328}` |
| validação (^GSPC) | `{'MISTO': 874, 'BULL': 647, 'BEAR': 342}` |

## Matrizes de Confusão (contagens)
### calibração (^BVSP)
```json
{
  "BULL": {
    "BULL": 477,
    "BEAR": 136,
    "MISTO": 511
  },
  "BEAR": {
    "BULL": 11,
    "BEAR": 81,
    "MISTO": 87
  },
  "MISTO": {
    "BULL": 279,
    "BEAR": 111,
    "MISTO": 170
  }
}
```
### validação (^GSPC)
```json
{
  "BULL": {
    "BULL": 568,
    "BEAR": 190,
    "MISTO": 560
  },
  "BEAR": {
    "BULL": 6,
    "BEAR": 44,
    "MISTO": 171
  },
  "MISTO": {
    "BULL": 73,
    "BEAR": 108,
    "MISTO": 143
  }
}
```

## Métricas
| conjunto | macro_f1 | balanced_accuracy | switches_per_year |
|---|---:|---:|---:|
| calibração (^BVSP) | 0.360015 | 0.515978 | 5.545894 |
| validação (^GSPC) | 0.324385 | 0.643001 | 6.086957 |

## Inputs/Autodiscovery
- source_info: `{"method": "autodiscovery_required_fields", "selected": "/home/wilson/CEP_COMPRA/outputs/backtests/task_012/run_20260212_114129/consolidated/series_alinhadas_plot.parquet", "selected_columns": ["date", "M0_equity_idx", "M1_equity_idx", "cdi_index", "sp500_index", "bvsp_index", "cdi_index_norm", "sp500_index_norm", "bvsp_index_norm"], "candidates_found": 17}`
- theory_params: `{"lookback_days_for_peaks_troughs": 252, "bull_threshold": 0.2, "bear_threshold": -0.2, "neutral_band": 0.1, "min_days": 5, "hysteresis": {"bull_exit_neutral": 0.1, "bear_exit_neutral": -0.1}}`

## Artefatos
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1/labels_theory_bvsp.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1/labels_theory_gspc.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1/cep_features_bvsp.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1/cep_features_gspc.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1/model_fit_summary.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1/threshold_search_results.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1/regime_daily_bvsp.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1/regime_daily_gspc.parquet`
- `/home/wilson/CEP_COMPRA/ssot_cycle2/master_regime_classifier_v5.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1/report.md`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1/manifest.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002B_master_regime_supervised_price_theory_v1/hashes.sha256`
