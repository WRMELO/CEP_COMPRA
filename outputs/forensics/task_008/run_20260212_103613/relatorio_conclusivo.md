# TASK 008 - Relatorio conclusivo

## Secao 1 - Salto 2025-11-07
- O maior contribuinte por mark-to-market no dia foi `IFCM3`.
- Atribuicao foi feita por ticker com `mtm_gain_estimado_brl = pos_prev * (exp(xt_jump)-1)`.
- O aumento abrupto de caixa decorre de vendas no mesmo dia apos valorizacao no modelo (saida de posicao para caixa), nao de residuo contabil.

## Secao 2 - Descolamento/plano pos-2022-07
- Intervalo de baixa volatilidade detectado: `2022-10-26` a `2023-03-31`.
- Evidencias rejeitam hipoteses de caixa majoritario, bloqueio sistematico de compras, erro de rebase e erro contabil recorrente.
- O descolamento no intervalo detectado e comportamental/performance relativa da carteira frente aos benchmarks, com carteira investida e baixa volatilidade.

## Correcao aplicada
- Nenhuma correcao de codigo aplicada nesta task, pois a causa tecnica localizada em logica contabil/rebase nao foi confirmada por evidencia.

## Artefatos
- output_root: `/home/wilson/CEP_COMPRA/outputs/forensics/task_008/run_20260212_103613`
- detalhes no manifest e hashes.