# Artefato da Conversa - EXP_003C

## OVERALL
**OVERALL PASS**

## STEPS
- S1_GATE_ALLOWLIST_CEP_COMPRA_ONLY: PASS
- S2_CHECK_COMPILE_OR_IMPORTS: PASS
- S3_LOAD_OR_BUILD_INPUTS_FROM_003B: PASS
- S4_BUILD_BUY3_LABELS_FROM_5STATE: PASS
- S5_FIT_BUY3_MODEL_ON_BVSP_THRESHOLD_SEARCH: PASS
- S6_VALIDATE_ON_GSPC_NO_REFIT: PASS
- S7_GENERATE_CONFUSION_BUY3_COUNTS: PASS
- S8_BUILD_BUY1_CONTINUOUS_SCORE: PASS
- S9_VERIFY_BUY3_REACHABLE_AND_STABLE: PASS
- S10_ANTI_LEAKAGE_AUDIT_DMINUS1: PASS
- S11_WRITE_SSOT_V8_EXPLICIT_FORMULA_PSEUDOCODE_SCORE: PASS
- S12_GENERATE_MD_AUTOCONTIDO_MANIFEST_HASHES: PASS

## ARTEFATOS
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/buy3_labels_bvsp.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/buy3_labels_gspc.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/model_fit_summary.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/threshold_search_results.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/buy_daily_bvsp.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/buy_daily_gspc.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/buy1_intensity_bvsp.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/buy1_intensity_gspc.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/confusion_buy3_counts.json`
- `/home/wilson/CEP_COMPRA/ssot_cycle2/master_buy_classifier_v8.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/report.md`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/manifest.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_003C_master_buy_ssot_v8/hashes.sha256`

## COMENTÁRIOS
- Runner: `scripts/exp_003c_master_buy_ssot_v8.py`.
- Modelo escolhido: `logistic_regression`.
- Métricas BUY3:
  - BVSP: `macro_f1=0.388985`, `balanced_accuracy=0.491577`, `switches/ano=5.727273`.
  - GSPC: `macro_f1=0.247375`, `balanced_accuracy=0.426822`, `switches/ano=5.447894`.
- Classes alcançáveis em ambos:
  - BVSP: `{'BUY2': 815, 'BUY0': 615, 'BUY1': 374}`
  - GSPC: `{'BUY0': 683, 'BUY1': 573, 'BUY2': 548}`
- Gate anti-leakage D-1: PASS.
