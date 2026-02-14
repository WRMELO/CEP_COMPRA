# Master Regime M3 CEP Only V3 (autocontido)

## 1) Janelas W1/W2 (M3) e evidência
- W1: 2018-07-02 .. 2021-06-30
- W2: 2021-07-01 .. 2022-12-30
- evidência: `/home/wilson/CEP_COMPRA/outputs/reports/task_018/run_20260212_134037/analise_consolidada_fases_m0_m1_m3.md` :: W1 | 2018-07-02..2021-06-30 ; W2 | 2021-07-01..2022-12-30
- evidência: `/home/wilson/CEP_COMPRA/outputs/backtests/task_021_m6/run_20260213_122019/m6_vs_m3_and_others_analysis_autossuficiente.md` :: Comparação por fases W1/W2/W3 (M3 vs M6)

## 2) Features CEP (Master-only)
| feature | definição |
|---|---|
| r_t | log-retorno diário do ^BVSP |
| mr_t | amplitude móvel |r_t-r_{t-1}| |
| xbar_n | média móvel de r_t com n=N_master |
| range_n | range móvel de r_t com n=N_master |
| z_r_60 | z-score de r_t em janela 60 |
| cep_i_below_lcl | indicador r_t < I_LCL |
| cep_i_above_ucl | indicador r_t > I_UCL |
| cep_mr_above_ucl | indicador mr_t > MR_UCL |
| cep_xbar_below_lcl | indicador xbar_n < Xbar_LCL |
| cep_xbar_above_ucl | indicador xbar_n > Xbar_UCL |
| cep_r_above_ucl | indicador range_n > R_UCL |
| dist_i_lcl | distância de r_t ao limite inferior I |
| dist_i_ucl | distância de r_t ao limite superior I |
| dist_xbar_lcl | distância de xbar_n ao limite inferior Xbar |
| dist_xbar_ucl | distância de xbar_n ao limite superior Xbar |
| stress_score | combinação linear de flags downside CEP |
| upside_score | combinação linear de flags upside CEP |

## 3) Modelo interpretável (equação/pseudocódigo)
- modelo escolhido: **shallow_tree**
- fórmula: `if z(feature)<=threshold then p_bull=p_left else p_bull=p_right`
```text
input diário: x_t (features CEP do ^BVSP)
z_i = (x_i - mean_train_i) / std_train_i
if z(feature)<=threshold: p_bull=p_left else p_bull=p_right
if p_bull >= p_bull_enter => alvo=BULL
elif p_bull <= p_bear_enter => alvo=BEAR
else alvo=TRANSICAO
troca de estado efetiva somente após min_days confirmações consecutivas
```

## 4) Limiar/histerese/min_days (melhor configuração)
- p_bull_enter=0.5500, p_bull_exit=0.5400, p_bear_enter=0.4500, p_bear_exit=0.4600, min_days=3
- lambda_switch_penalty=0.00

## 5) Matriz de confusão (contagens, validação W1 vs W2)
| célula | contagem |
|---|---:|
| TP (W1->BULL) | 173 |
| FN (W1->!BULL) | 49 |
| TN (W2->BEAR) | 0 |
| FP (W2->!BEAR) | 113 |
| TRANSICAO (na validação) | 92 |

## 6) Métricas (validação)
- balanced_accuracy=0.579905
- auc=0.546201
- BULL(W1): precision=0.711934, recall=0.779279, f1=0.744086
- BEAR(W2): precision=0.467391, recall=0.380531, f1=0.419512
- score_final=0.579905 (balanced_accuracy - lambda*switch_rate)

## 7) Estabilidade (validação)
- switches=29
- duração média por regime:
  - BULL: 16.2000
  - TRANSICAO: 6.1333
- duração mediana por regime:
  - BULL: 10.0000
  - TRANSICAO: 4.0000
- % dias por regime:
  - BULL: 72.5373%
  - TRANSICAO: 27.4627%

## 8) Justificativa numérica final
- Modelo treinado com split temporal 70/30 em rótulos W1/W2.
- Seleção objetiva via score_final com penalidade explícita de trocas.
- Features estritamente derivadas de `^BVSP` + limites CEP baseline do Master.
- Gate G1_NO_ENDOGENOUS: **PASS**.

## SSOT v3 (JSON integral)
```json
{
  "task_id": "TASK_CEP_CYCLE2_001D_MASTER_REGIME_M3_CEP_ONLY_V3",
  "version": "v3",
  "master_ticker": "^BVSP",
  "label_policy": {
    "bull_true": "W1",
    "bear_true": "W2",
    "transition": "probabilidade intermediária + histerese + min_days"
  },
  "model": {
    "chosen_model": "shallow_tree",
    "formula_explicit": "if z(feature)<=threshold then p_bull=p_left else p_bull=p_right",
    "formula_parameters": {
      "feature": "xbar_n",
      "threshold": -0.213199,
      "p_if_feature_le_threshold": 0.5384615384615384,
      "p_if_feature_gt_threshold": 0.7317554240631163
    },
    "features_used": [
      "r_t",
      "mr_t",
      "xbar_n",
      "range_n",
      "z_r_60",
      "cep_i_below_lcl",
      "cep_i_above_ucl",
      "cep_mr_above_ucl",
      "cep_xbar_below_lcl",
      "cep_xbar_above_ucl",
      "cep_r_above_ucl",
      "dist_i_lcl",
      "dist_i_ucl",
      "dist_xbar_lcl",
      "dist_xbar_ucl",
      "stress_score",
      "upside_score"
    ],
    "feature_definitions": {
      "r_t": "log-retorno diário do ^BVSP",
      "mr_t": "amplitude móvel |r_t-r_{t-1}|",
      "xbar_n": "média móvel de r_t com n=N_master",
      "range_n": "range móvel de r_t com n=N_master",
      "z_r_60": "z-score de r_t em janela 60",
      "cep_i_below_lcl": "indicador r_t < I_LCL",
      "cep_i_above_ucl": "indicador r_t > I_UCL",
      "cep_mr_above_ucl": "indicador mr_t > MR_UCL",
      "cep_xbar_below_lcl": "indicador xbar_n < Xbar_LCL",
      "cep_xbar_above_ucl": "indicador xbar_n > Xbar_UCL",
      "cep_r_above_ucl": "indicador range_n > R_UCL",
      "dist_i_lcl": "distância de r_t ao limite inferior I",
      "dist_i_ucl": "distância de r_t ao limite superior I",
      "dist_xbar_lcl": "distância de xbar_n ao limite inferior Xbar",
      "dist_xbar_ucl": "distância de xbar_n ao limite superior Xbar",
      "stress_score": "combinação linear de flags downside CEP",
      "upside_score": "combinação linear de flags upside CEP"
    },
    "normalization": {
      "mean_train": {
        "r_t": 9.600872691951329e-05,
        "mr_t": 0.017984065280981808,
        "xbar_n": 0.00011728816104592858,
        "range_n": 0.03199713549826618,
        "z_r_60": -0.027576084204109313,
        "cep_i_below_lcl": 0.020512820512820513,
        "cep_i_above_ucl": 0.015384615384615385,
        "cep_mr_above_ucl": 0.03717948717948718,
        "cep_xbar_below_lcl": 0.023076923076923078,
        "cep_xbar_above_ucl": 0.00641025641025641,
        "cep_r_above_ucl": 0.06538461538461539,
        "dist_i_lcl": 0.03375403097221375,
        "dist_i_ucl": 0.036342136546230454,
        "dist_xbar_lcl": 0.016345998781248792,
        "dist_xbar_ucl": 0.018709978044576654,
        "stress_score": 0.09115384615384614,
        "upside_score": 0.020512820512820513
      },
      "std_train": {
        "r_t": 0.019055730473612954,
        "mr_t": 0.02317624904702007,
        "xbar_n": 0.008758369248863553,
        "range_n": 0.029094361556198635,
        "z_r_60": 1.0554581379568098,
        "cep_i_below_lcl": 0.14174640985728518,
        "cep_i_above_ucl": 0.12307692307692389,
        "cep_mr_above_ucl": 0.18920140832604215,
        "cep_xbar_below_lcl": 0.15014785612263826,
        "cep_xbar_above_ucl": 0.07980704870505491,
        "cep_r_above_ucl": 0.2472032917572514,
        "dist_i_lcl": 0.019055730473612954,
        "dist_i_ucl": 0.01905573047361295,
        "dist_xbar_lcl": 0.008758369248863552,
        "dist_xbar_ucl": 0.008758369248863552,
        "stress_score": 0.38645157844833483,
        "upside_score": 0.14531925451594668
      }
    }
  },
  "threshold_hysteresis_min_days": {
    "p_bull_enter": 0.55,
    "p_bull_exit": 0.54,
    "p_bear_enter": 0.45,
    "p_bear_exit": 0.46,
    "min_days": 3,
    "transition_definition": "quando p_bull fica entre zonas de entrada BEAR/BULL ou quando regra de confirmação (min_days) não fecha mudança"
  },
  "assertiveness_criterion": {
    "validation_split": {
      "train_fraction": 0.7,
      "validate_fraction": 0.30000000000000004,
      "type": "time_series_split"
    },
    "selection_score": "score_final = balanced_accuracy_validacao - lambda_switch_penalty * switch_rate_validacao",
    "best_lambda_switch_penalty": 0.0,
    "balanced_accuracy_validation": 0.5799051263653033,
    "auc_validation": 0.5462010683249622
  },
  "baseline_limits_master": {
    "i_mean": 0.0013900615139278516,
    "i_std": 0.011682694586407371,
    "i_lcl": -0.03365802224529426,
    "i_ucl": 0.03643814527314996,
    "mr_ucl": 0.04653452228626935,
    "xbar_lcl": -0.016228710620202858,
    "xbar_ucl": 0.018827266205622587,
    "r_ucl": 0.05486813382478304,
    "baseline_sessions": 60,
    "n_master": 4
  },
  "critical_input_hashes": {
    "/home/wilson/CEP_COMPRA/outputs/reports/task_018/run_20260212_134037/analise_consolidada_fases_m0_m1_m3.md": "6081289dd4e0b7603d3fc15019078608b3dfcf5457d030c525f2cf186cee4719",
    "/home/wilson/CEP_NA_BOLSA/outputs/ssot/precos_brutos/ibov/brapi/20260204/xt_ibov.csv": "677a225589c750979d072917ac152ad830bf3d113695ca71d1e44cc0c06afcc6",
    "/home/wilson/CEP_NA_BOLSA/outputs/governanca/operacao_rotina/20260209/ROTINA_GESTAO_CARTEIRA_001/artefatos/120_master_calibration.json": "fc7b70c2929276b0413c1296a0af34f68c79e239eb3b1ceac52fd4ec31b3e602",
    "/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration/ssot_cycle2/master_regime_classifier_v1.json": "79e97b585fad790f63a776f3181e5efcc73a641b63e70cd838c48ce99c86259e"
  },
  "gates": {
    "G1_NO_ENDOGENOUS": "PASS",
    "G2_EXPLICIT_FORMULA": "PASS",
    "G3_ASSERTIVENESS_EXPLAINED": "PASS",
    "G4_MD_AUTOCONTIDO": "PASS"
  }
}
```
