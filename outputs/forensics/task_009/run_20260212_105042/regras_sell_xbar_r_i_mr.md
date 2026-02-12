# Regras de SELL ligadas a rompimentos (Xbar/R/I/MR)

## Localização exata no código
- `CEP_COMPRA/tools/task_005_report_runner.py` (bloco de venda diária congelada):
  - `if float(r) < float(lim["i_lcl"]) or float(r) > float(lim["i_ucl"]) or mr > float(lim["mr_ucl"]): cash += pos.pop(t)`
- `CEP_NA_BOLSA/src/cep/runners/runner_backtest_fixed_limits_exp031_v1.py` (estado Master):
  - `stress_amp_raw = r > limits_xbar_r["ucl_r"]`
  - `xbar_up = xbar > limits_xbar_r["ucl_xbar"]`
  - `stress_i_down = xt < limits_imr["lcl_i"]`
  - `stress_i_up = xt > limits_imr["ucl_i"]`
  - `upside_extreme = xbar_up | stress_i_up`
  - `stress_amp = stress_amp_raw & (~upside_extreme)`

## Definição de X_t (sem inferência)
- `CEP_NA_BOLSA/docs/00_constituicao/CEP_NA_BOLSA_CONSTITUICAO_V2_20260204.md` define:
  - `X_t = log(Close_t / Close_{t-1})`
- `schema_base_operacional.json` registra as colunas da base operacional: `ticker,date,close,xt,asset_class`.

## Resposta objetiva (SELL por rompimento positivo/UCL)
- Xbar (UCL): **não há SELL direto no runner da carteira**; no Master, `xbar_up` atua como exceção de upside para não acionar preservação por amplitude.
- R (UCL): no Master, `r > ucl_r` é gatilho de estresse de amplitude (`stress_amp_raw`), com exceção de upside.
- I (UCL): no runner da carteira há SELL direto por `xt > i_ucl`.
- MR (UCL): no runner da carteira há SELL direto por `mr > mr_ucl`.
