# TASK_CEP_COMPRA_005 - Pre-flight de dados e reproducibilidade (S0)

## Ambiente

- Python/venv: `/home/wilson/PortfolioZero/.venv`
- Parquet engine: `pyarrow` disponivel
- Git LFS: `git-lfs/3.6.1`

## Gate de dados (LFS)

Checagens executadas:

1. Verificacao de ponteiros LFS em arquivos criticos (base operacional, SSOT ACOES, SSOT BDR, limites por ticker).
2. Materializacao via:
   - `git -C /home/wilson/CEP_NA_BOLSA lfs pull`
3. Revalidacao de conteudo real (cabecalhos de dados, nao ponteiro):
   - `base_operacional_xt.csv`
   - `ssot_acoes_b3.csv`
   - `ssot_bdr_b3.csv`
   - `merged/limits_per_ticker.csv`

## Resultado S0

- **PASS**: dados efetivos disponiveis para a simulacao completa.
- Nenhum bloqueio de LFS permaneceu nos arquivos usados pela task.
