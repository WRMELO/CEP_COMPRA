# Revogação formal do SSOT v1

- task_id_revogação: `TASK_CEP_CYCLE2_001D_MASTER_REGIME_M3_CEP_ONLY_V3`
- ssot_revogado_path: `/home/wilson/CEP_COMPRA/outputs/cycle2/20260213/master_regime_calibration/ssot_cycle2/master_regime_classifier_v1.json`
- motivo: SSOT v1 utilizou variáveis endógenas de carteira/experimento, proibidas para regime do Master.
- impacto operacional: v1 não deve ser usado como regra vigente de regime Master.
- política substituta: regime derivado apenas de CEP do `^BVSP` (Master), com modelo interpretável e validação temporal.

## Variáveis endógenas detectadas no SSOT v1
- `sinais_cep_n_positions_m6`
- `sinais_cep_portfolio_state_risk_on`
