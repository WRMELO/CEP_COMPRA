# Regime Calibration Comparison V1 vs V2

## Decisão CTO (gates objetivos)
- freeze_v2: **FAIL**
- gate balanced_accuracy >= 0.58: **FAIL**
- gate switches <= 1.20 * v1: **PASS**
- gate zero endógenas em v2: **PASS**
- modelo vigente após decisão: **v1**

## SSOT v1 (JSON integral)
```json
{
  "task_id": "TASK_CEP_CYCLE2_001_MASTER_REGIME_CALIBRATION_V1",
  "version": "v1",
  "date_reference": "2026-02-13",
  "scenario_bear": "W3",
  "features_used": [
    "sinais_cep_two_sigma_neg",
    "sinais_cep_xbar_lcl",
    "sinais_cep_stress_amp",
    "sinais_cep_n_positions_m6",
    "sinais_cep_portfolio_state_risk_on"
  ],
  "orientations": {
    "sinais_cep_two_sigma_neg": 1,
    "sinais_cep_xbar_lcl": 1,
    "sinais_cep_stress_amp": 1,
    "sinais_cep_n_positions_m6": -1,
    "sinais_cep_portfolio_state_risk_on": -1
  },
  "thresholds": {
    "bull_enter": 0.4263069215483866,
    "bull_exit": 0.4177807831174189,
    "bear_enter": 0.255784152929032,
    "bear_exit": 0.2643102913599997,
    "min_days": 1
  },
  "objective_metrics": {
    "balanced_accuracy": 0.6312196812196812,
    "regime_switches": 129,
    "score_final": 0.6215928155480394
  },
  "predicted_regime_distribution": {
    "BULL": 1123,
    "BEAR": 886,
    "TRANSICAO": 1
  }
}
```

## SSOT v2 (JSON integral)
```json
{
  "task_id": "TASK_CEP_CYCLE2_001B_MASTER_REGIME_EXOG_RECALIBRATION_V2",
  "version": "v2",
  "date_reference": "2026-02-13",
  "source_v1_task_id": "TASK_CEP_CYCLE2_001_MASTER_REGIME_CALIBRATION_V1",
  "scenario_bear": "W2",
  "features_used": [
    "sinais_cep_stress_amp",
    "sinais_cep_two_sigma_neg",
    "sinais_cep_xbar_lcl",
    "sinais_cep_trend_run7",
    "sinais_cep_r_t"
  ],
  "orientations": {
    "sinais_cep_stress_amp": 1,
    "sinais_cep_two_sigma_neg": 1,
    "sinais_cep_xbar_lcl": 1,
    "sinais_cep_trend_run7": -1,
    "sinais_cep_r_t": 1
  },
  "thresholds": {
    "bull_enter": 0.2728474646395836,
    "bull_exit": 0.14938922018930156,
    "bear_enter": -0.13868001686135656,
    "bear_exit": -0.015221772411074502,
    "min_days": 3
  },
  "objective_metrics": {
    "balanced_accuracy": 0.11758558558558559,
    "regime_switches": 76,
    "score_final": 0.11191394379454081
  },
  "predicted_regime_distribution": {
    "TRANSICAO": 1743,
    "BEAR": 239,
    "BULL": 28
  },
  "endogenous_exclusion": {
    "patterns": [
      "*n_positions*",
      "*portfolio_state*",
      "*positions*",
      "*risk_on*"
    ],
    "hard_exclude_features": [
      "sinais_cep_n_positions_m6",
      "sinais_cep_portfolio_state_risk_on"
    ],
    "remaining_endogenous_in_features_used": []
  },
  "freeze_decision": {
    "status": "FAIL",
    "gates": {
      "min_balanced_accuracy": 0.58,
      "max_switches_multiplier_vs_v1": 1.2,
      "must_have_zero_endogenous_features": true,
      "gate_balanced_accuracy_pass": false,
      "gate_switches_pass": true,
      "gate_zero_endogenous_pass": true
    },
    "comparative_numbers": {
      "balanced_accuracy_v1": 0.6312196812196812,
      "balanced_accuracy_v2": 0.11758558558558559,
      "switches_v1": 129,
      "switches_v2": 76,
      "switches_limit_v2": 154.79999999999998
    },
    "active_model_after_decision": "v1"
  }
}
```

## Tabela comparativa
| métrica | v1 | v2 |
|---|---:|---:|
| balanced_accuracy | 0.631220 | 0.117586 |
| regime_switches | 129 | 76 |
| % dias BEAR | 44.0796% | 11.8905% |
| % dias BULL | 55.8706% | 1.3930% |
| % dias TRANSICAO | 0.0498% | 86.7164% |
| duração média BEAR | 13.6308 | 6.8286 |
| duração mediana BEAR | 2.0000 | 5.0000 |
| duração média BULL | 17.5469 | 9.3333 |
| duração mediana BULL | 6.5000 | 8.0000 |
| duração média TRANSICAO | 1.0000 | 44.6923 |
| duração mediana TRANSICAO | 1.0000 | 31.0000 |

## Matriz de confusão V1 (contagens)
| célula | contagem |
|---|---:|
| TP (W1 -> BULL) | 531 |
| FN (W1 -> !BULL) | 209 |
| TN (BEAR -> BEAR) | 170 |
| FP (BEAR -> !BEAR) | 142 |

## Matriz de confusão V2 (contagens)
| célula | contagem |
|---|---:|
| TP (W1 -> BULL) | 28 |
| FN (W1 -> !BULL) | 712 |
| TN (BEAR -> BEAR) | 74 |
| FP (BEAR -> !BEAR) | 301 |

## Features V2 (exógenas)
- features usadas no SSOT v2: `sinais_cep_stress_amp, sinais_cep_two_sigma_neg, sinais_cep_xbar_lcl, sinais_cep_trend_run7, sinais_cep_r_t`
- checagem endógenas remanescentes: `(nenhuma)`
- colunas finais do dataset V2 (12): `sinais_cep_master_xt, sinais_cep_stress_i, sinais_cep_stress_amp, sinais_cep_trend_run7, sinais_cep_daily_return_m6, sinais_cep_cash_ratio_m6, sinais_cep_rule_one_below_lcl, sinais_cep_rule_two_of_three_below_2sigma_neg, sinais_cep_xbar_t, sinais_cep_r_t, sinais_cep_xbar_lcl, sinais_cep_two_sigma_neg`
