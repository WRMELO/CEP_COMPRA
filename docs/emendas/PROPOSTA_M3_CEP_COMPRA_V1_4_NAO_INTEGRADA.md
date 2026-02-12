# PROPOSTA M3 CEP_COMPRA v1.4 (não integrada)

## Objetivo
Combinar critérios de ranking já existentes no pipeline, preservando gating sob controle.

## Ranking M3 (somente features existentes)
- score_m3 = z(score_m0) + z(ret_lookback_62) - z(vol_lookback_62)
- score_m0: média de xt no lookback (base M0)
- ret_lookback_62: soma de xt no lookback
- vol_lookback_62: desvio-padrão de xt no lookback

## Invariantes mantidos
- volume > 0 em pelo menos 50/62 dias (excludente)
- uma classe por empresa
- cap por setor 20% (UNKNOWN conta)
- mix B3 entre 50% e 80%
- gating de venda com upside_extreme e stress_amp sob controle
