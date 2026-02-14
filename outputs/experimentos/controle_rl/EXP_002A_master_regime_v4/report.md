# EXP_002A Master Regime V4

## OVERALL
- OVERALL PASS

## STEPS
- S1_GATE_ALLOWLIST: PASS — validação de leitura/escrita em allowlist.
- S2_AUTODISCOVER_INPUTS: PASS — série master em parquet detectada; weak monthly via fallback textual auditável.
- S3_BUILD_DAILY_LOGRET_AND_CEP_FEATURES: PASS — log-ret diário e features CEP exógenas geradas.
- S4_BUILD_WEAKLABELS_DAILY_FROM_MONTHLY: PASS — weak labels mensais expandidas para diário.
- S5_FIT_INTERPRETABLE_MODEL_AND_THRESHOLD_SEARCH: PASS — logistic/tree + grid de histerese/min_days.
- S6_VERIFY_CLASSES_REACHABLE: PASS — dias BULL=181, BEAR=588, MISTO=1094.
- S7_WRITE_SSOT_V4_WITH_EXPLICIT_FORMULA: PASS — fórmula explícita + parâmetros no SSOT.
- S8_GENERATE_MD_AUTOCONTIDO_AND_MANIFEST: PASS — report autocontido + manifest + hashes.

## INPUTS RESOLVIDOS
- master_series_path: `/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446/data/bvsp_index.parquet`
- master_series_discovery: `{"method": "preferred_parquet_globs", "candidates_found": 3, "accepted_candidates": ["/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214446/data/bvsp_index.parquet", "/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214315/data/bvsp_index.parquet", "/home/wilson/CEP_COMPRA/outputs/plots/task_007/run_20260211_214244/data/bvsp_index.parquet"]}`
- weak_label_source: `{"method": "user_text_weak_indication", "bull_years": [2019, 2023, 2025], "bear_years": [2021, 2022, 2024], "notes": "Fallback intencional: não foi encontrado JSON mensal no repositório pelos globs informados."}`

## DEFINIÇÃO DE WEAK LABELS (a partir do texto)
- BULL: anos 2019, 2023, 2025 (todos os meses).
- BEAR: anos 2021, 2022, 2024 (todos os meses).
- UNLABELED: demais meses.

## MODELO INTERPRETÁVEL
- chosen_model: **logistic_regression**
- validation_metrics_best: `{"balanced_accuracy": 0.6013939393939394, "auc": 0.5035151515151515, "bull": {"precision": 0.7714285714285715, "recall": 0.324, "f1": 0.45633802816901414}, "bear": {"precision": 0.5072886297376094, "recall": 0.8787878787878788, "f1": 0.6432532347504621}}`
- selection_score: `score_final = balanced_accuracy_validation - 0.05 * misto_rate_validation`

## THRESHOLDS / HISTERSE / MIN_DAYS
- best_config: `{"p_bull_enter": 0.55, "p_bull_exit_delta": 0.01, "p_bear_enter": 0.45, "p_bear_exit_delta": 0.01, "min_days": 5, "balanced_accuracy_validation": 0.6013939393939394, "misto_rate_validation": 0.734375, "score_final": 0.5646751893939393}`

## ALCANÇABILIDADE DE CLASSES
- BULL dias: **181**
- BEAR dias: **588**
- MISTO dias: **1094**
- critério >=30 em BULL e BEAR: **PASS**

## ARTEFATOS
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4/master_logret_daily.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4/master_cep_features_daily.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4/weaklabels_daily_audit.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4/ssot_cycle2/master_regime_classifier_v4.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4/regime_daily.parquet`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4/report.md`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4/manifest.json`
- `/home/wilson/CEP_COMPRA/outputs/experimentos/controle_rl/EXP_002A_master_regime_v4/hashes.sha256`
