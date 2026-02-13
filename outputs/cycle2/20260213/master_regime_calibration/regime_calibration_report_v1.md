# Regime Calibration Report v1

- task_id: `TASK_CEP_CYCLE2_001_MASTER_REGIME_CALIBRATION_V1`
- base_dir: `outputs/cycle2/20260213/master_regime_calibration`
- leitura permitida: `/home/wilson/CEP_COMPRA`, `/home/wilson/CEP_NA_BOLSA`

## Fontes e inventario
- total_fontes_inventariadas: **6979**
- top prioridade outputs/planning/docs: **1578**

## Janelas W1/W2/W3
- metodo: **explicit_cycle_z_artifacts**
- origem principal: `/home/wilson/CEP_COMPRA/tools/task_021_m6_runner.py`
- W1: 2018-07-01 .. 2021-06-30
- W2: 2021-07-01 .. 2022-12-31
- W3: 2024-09-01 .. 2025-11-30

## Dataset de sinais
- linhas: **2010**
- colunas sinais_cep_*: **15**

## Scoring de features
### Top 5 - cenário BEAR=W2
- `sinais_cep_stress_amp` | auc=0.5169 ks=0.7527 t=5.0832 rank=1
- `sinais_cep_two_sigma_neg` | auc=0.5000 ks=0.7452 t=5.0000 rank=2
- `sinais_cep_xbar_lcl` | auc=0.5000 ks=0.7452 t=5.0000 rank=2
- `sinais_cep_trend_run7` | auc=0.5548 ks=0.5417 t=4.9318 rank=3
- `sinais_cep_r_t` | auc=0.8846 ks=0.7740 t=3.3617 rank=4

### Top 5 - cenário BEAR=W3
- `sinais_cep_two_sigma_neg` | auc=0.5000 ks=0.7523 t=5.0990 rank=1
- `sinais_cep_xbar_lcl` | auc=0.5000 ks=0.7523 t=5.0990 rank=1
- `sinais_cep_stress_amp` | auc=0.5169 ks=0.7090 t=5.0832 rank=2
- `sinais_cep_n_positions_m6` | auc=0.5904 ks=0.2981 t=4.2993 rank=3
- `sinais_cep_portfolio_state_risk_on` | auc=0.5675 ks=0.3117 t=4.2567 rank=4

## Melhor candidato
- cenário BEAR vencedor: **W3**
- features: `sinais_cep_two_sigma_neg, sinais_cep_xbar_lcl, sinais_cep_stress_amp, sinais_cep_n_positions_m6, sinais_cep_portfolio_state_risk_on`
- balanced_accuracy: **0.631220**
- regime_switches: **129**
- score_final: **0.621593**

## Confusão e estabilidade
- confusion balanced_accuracy: **0.631220**
- switches: **129**

