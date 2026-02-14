# V1 Rule Dump Autocontido

## Base V1 (paths + sha256)
- ssot_v1: `/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration/ssot_cycle2/master_regime_classifier_v1.json` | sha256=`79e97b585fad790f63a776f3181e5efcc73a641b63e70cd838c48ce99c86259e`
- confusion_v1: `/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration/confusion_summary.json` | sha256=`2d5eeebcc8f6682358fc631507a3468a5e752bcf7a1311e43d433504e460e794`
- stability_v1: `/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration/stability_summary.json` | sha256=`11fce88b7a5b3287614355f12a5931866aa2b47351959c34932bba5464954370`
- candidate_v1: `/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration/candidate_thresholds.parquet` | sha256=`1ba6e3149642a2e35f94b8e7b791ed6cb02e5f722debb883935d05ad111a4c89`
- feature_v1: `/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration/feature_scores.parquet` | sha256=`1cd80486aa65cc689cbd3fccd1d8c8df38b83267f48d8a800ac42691f735ec8a`
- master_signals_v1: `/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration/master_signals_daily.parquet` | sha256=`05455ddd720a43a0a3ab12fa9a39a8931eac99f89c1bba8f1c83268374808efd`

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

## Regra de decisão (limiares/histerese/min_days)
| cenário_bear | features_used | bull_enter | bull_exit | bear_enter | bear_exit | min_days | balanced_accuracy | regime_switches | score_final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W3 | ["sinais_cep_two_sigma_neg", "sinais_cep_xbar_lcl", "sinais_cep_stress_amp", "sinais_cep_n_positions_m6", "sinais_cep_portfolio_state_risk_on"] | 0.426307 | 0.417781 | 0.255784 | 0.264310 | 1 | 0.631220 | 129 | 0.621593 |

## Matriz de confusão (contagens)
| célula | contagem |
|---|---:|
| TP (W1 -> BULL) | 531 |
| FN (W1 -> !BULL) | 209 |
| TN (BEAR -> BEAR) | 170 |
| FP (BEAR -> !BEAR) | 142 |

## Estabilidade V1
- switches: **129**
- duração média por regime:
  - BEAR: 13.6308
  - BULL: 17.5469
  - TRANSICAO: 1.0000
- duração mediana por regime:
  - BEAR: 2.0000
  - BULL: 6.5000
  - TRANSICAO: 1.0000
- % dias por regime:
  - BULL: 55.8706%
  - BEAR: 44.0796%
  - TRANSICAO: 0.0498%

## Features V1 e endogeneidade
- total features sinais_cep_ no dataset: **15**
- features endógenas identificadas por padrão/hard_exclude: **3**
- features usadas no SSOT v1: `sinais_cep_two_sigma_neg, sinais_cep_xbar_lcl, sinais_cep_stress_amp, sinais_cep_n_positions_m6, sinais_cep_portfolio_state_risk_on`
- endógenas usadas no SSOT v1: `sinais_cep_n_positions_m6, sinais_cep_portfolio_state_risk_on`
