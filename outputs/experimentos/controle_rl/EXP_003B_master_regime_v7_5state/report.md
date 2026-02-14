# EXP_003B Master Regime V7 5state + BUY confusion

## Definicao formal 5 estados + pseudocodigo
- `peak_t=max(Close_{t-252..t})`, `trough_t=min(Close_{t-252..t})`
- `drawdown_t=Close_t/peak_t-1`, `rise_t=Close_t/trough_t-1`
- prioridade: BEAR > CORR_NEUTRO_BEAR > BULL > CORR_BULL_NEUTRO > NEUTRO
- BEAR: drawdown<=-0.20
- CORR_NEUTRO_BEAR: -0.20<drawdown<=-0.10
- BULL: rise>=0.20
- CORR_BULL_NEUTRO: 0.10<=rise<0.20
- NEUTRO: caso contrario; com histerese + min_days causal

## Mapa 5state->BUY3 conforme DP do Owner
- `{'BULL': 'BUY2', 'CORR_BULL_NEUTRO': 'BUY2', 'NEUTRO': 'BUY2', 'CORR_NEUTRO_BEAR': 'BUY1', 'BEAR': 'BUY0'}`

## Distribuicao por classe (5state e BUY3)
- BVSP 5state: `{'CORR_BULL_NEUTRO': 737, 'BULL': 445, 'CORR_NEUTRO_BEAR': 367, 'BEAR': 214, 'NEUTRO': 100}`
- GSPC 5state: `{'CORR_BULL_NEUTRO': 911, 'CORR_NEUTRO_BEAR': 350, 'BEAR': 291, 'BULL': 176, 'NEUTRO': 135}`
- BVSP BUY3: `{'BUY2': 1282, 'BUY1': 367, 'BUY0': 214}`
- GSPC BUY3: `{'BUY2': 1222, 'BUY1': 350, 'BUY0': 291}`

## Matrizes de confusao com contagens
### 5state
```json
{
  "bvsp": {
    "BULL": {
      "BULL": 198,
      "CORR_BULL_NEUTRO": 465,
      "NEUTRO": 22,
      "CORR_NEUTRO_BEAR": 128,
      "BEAR": 67
    },
    "CORR_BULL_NEUTRO": {
      "BULL": 125,
      "CORR_BULL_NEUTRO": 172,
      "NEUTRO": 1,
      "CORR_NEUTRO_BEAR": 28,
      "BEAR": 41
    },
    "NEUTRO": {
      "BULL": 0,
      "CORR_BULL_NEUTRO": 15,
      "NEUTRO": 67,
      "CORR_NEUTRO_BEAR": 10,
      "BEAR": 0
    },
    "CORR_NEUTRO_BEAR": {
      "BULL": 91,
      "CORR_BULL_NEUTRO": 78,
      "NEUTRO": 10,
      "CORR_NEUTRO_BEAR": 180,
      "BEAR": 28
    },
    "BEAR": {
      "BULL": 31,
      "CORR_BULL_NEUTRO": 7,
      "NEUTRO": 0,
      "CORR_NEUTRO_BEAR": 21,
      "BEAR": 78
    }
  },
  "gspc": {
    "BULL": {
      "BULL": 144,
      "CORR_BULL_NEUTRO": 807,
      "NEUTRO": 59,
      "CORR_NEUTRO_BEAR": 100,
      "BEAR": 66
    },
    "CORR_BULL_NEUTRO": {
      "BULL": 10,
      "CORR_BULL_NEUTRO": 99,
      "NEUTRO": 12,
      "CORR_NEUTRO_BEAR": 48,
      "BEAR": 5
    },
    "NEUTRO": {
      "BULL": 22,
      "CORR_BULL_NEUTRO": 0,
      "NEUTRO": 64,
      "CORR_NEUTRO_BEAR": 10,
      "BEAR": 35
    },
    "CORR_NEUTRO_BEAR": {
      "BULL": 0,
      "CORR_BULL_NEUTRO": 5,
      "NEUTRO": 0,
      "CORR_NEUTRO_BEAR": 182,
      "BEAR": 119
    },
    "BEAR": {
      "BULL": 0,
      "CORR_BULL_NEUTRO": 0,
      "NEUTRO": 0,
      "CORR_NEUTRO_BEAR": 10,
      "BEAR": 66
    }
  }
}
```
### BUY3
```json
{
  "bvsp": {
    "BUY2": {
      "BUY2": 1065,
      "BUY1": 166,
      "BUY0": 108
    },
    "BUY1": {
      "BUY2": 179,
      "BUY1": 180,
      "BUY0": 28
    },
    "BUY0": {
      "BUY2": 38,
      "BUY1": 21,
      "BUY0": 78
    }
  },
  "gspc": {
    "BUY2": {
      "BUY2": 1217,
      "BUY1": 158,
      "BUY0": 106
    },
    "BUY1": {
      "BUY2": 5,
      "BUY1": 182,
      "BUY0": 119
    },
    "BUY0": {
      "BUY2": 0,
      "BUY1": 10,
      "BUY0": 66
    }
  }
}
```

## Stability/switches
| conjunto | macro_f1_5state | balanced_accuracy_5state | switches_5state | macro_f1_buy3 | balanced_accuracy_buy3 | switches_buy3 |
|---|---:|---:|---:|---:|---:|---:|
| BVSP | 0.446055 | 0.491277 | 7.574879 | 0.578188 | 0.609943 | 3.246377 |
| GSPC | 0.358252 | 0.528631 | 7.033816 | 0.605011 | 0.761645 | 2.975845 |

## Auditoria anti-leakage D-1 (30 datas)
| index_ticker | D | last_input_date_used | execution_price_date | ok_dminus1 |
|---|---|---|---|---|
| ^BVSP | 2018-07-04 | 2018-07-03 | 2018-07-04 | True |
| ^BVSP | 2019-01-18 | 2019-01-17 | 2019-01-18 | True |
| ^BVSP | 2019-08-02 | 2019-08-01 | 2019-08-02 | True |
| ^BVSP | 2020-02-13 | 2020-02-12 | 2020-02-13 | True |
| ^BVSP | 2020-08-26 | 2020-08-25 | 2020-08-26 | True |
| ^BVSP | 2021-03-16 | 2021-03-15 | 2021-03-16 | True |
| ^BVSP | 2021-09-24 | 2021-09-23 | 2021-09-24 | True |
| ^BVSP | 2022-04-08 | 2022-04-07 | 2022-04-08 | True |
| ^BVSP | 2022-10-19 | 2022-10-18 | 2022-10-19 | True |
| ^BVSP | 2023-05-04 | 2023-05-03 | 2023-05-04 | True |
| ^BVSP | 2023-11-13 | 2023-11-10 | 2023-11-13 | True |
| ^BVSP | 2024-05-28 | 2024-05-27 | 2024-05-28 | True |
| ^BVSP | 2024-12-04 | 2024-12-03 | 2024-12-04 | True |
| ^BVSP | 2025-06-23 | 2025-06-20 | 2025-06-23 | True |
| ^BVSP | 2025-12-30 | 2025-12-29 | 2025-12-30 | True |
| ^GSPC | 2018-07-04 | 2018-07-03 | 2018-07-04 | True |
| ^GSPC | 2019-01-18 | 2019-01-17 | 2019-01-18 | True |
| ^GSPC | 2019-08-02 | 2019-08-01 | 2019-08-02 | True |
| ^GSPC | 2020-02-13 | 2020-02-12 | 2020-02-13 | True |
| ^GSPC | 2020-08-26 | 2020-08-25 | 2020-08-26 | True |
| ^GSPC | 2021-03-16 | 2021-03-15 | 2021-03-16 | True |
| ^GSPC | 2021-09-24 | 2021-09-23 | 2021-09-24 | True |
| ^GSPC | 2022-04-08 | 2022-04-07 | 2022-04-08 | True |
| ^GSPC | 2022-10-19 | 2022-10-18 | 2022-10-19 | True |
| ^GSPC | 2023-05-04 | 2023-05-03 | 2023-05-04 | True |
| ^GSPC | 2023-11-13 | 2023-11-10 | 2023-11-13 | True |
| ^GSPC | 2024-05-28 | 2024-05-27 | 2024-05-28 | True |
| ^GSPC | 2024-12-04 | 2024-12-03 | 2024-12-04 | True |
| ^GSPC | 2025-06-23 | 2025-06-20 | 2025-06-23 | True |
| ^GSPC | 2025-12-30 | 2025-12-29 | 2025-12-30 | True |

## Source info
- `{"method": "autodiscovery", "selected": "/home/wilson/CEP_COMPRA/outputs/backtests/task_012/run_20260212_114129/consolidated/series_alinhadas_plot.parquet", "score": 5, "selected_columns": ["date", "M0_equity_idx", "M1_equity_idx", "cdi_index", "sp500_index", "bvsp_index", "cdi_index_norm", "sp500_index_norm", "bvsp_index_norm"], "candidates_found": 5}`

