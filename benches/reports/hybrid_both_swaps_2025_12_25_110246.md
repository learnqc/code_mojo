# Benchmark Report
**Generated:** 2025-12-25 11:02:46
**Date(s):** 2025_12_25
**Total runs:** 20
**Parameters:** n, value, swap
**Benchmarks:** pure_v2_ms, pure_v3_ms, hybrid_ms

## Results

| n | value | swap | pure_v2_ms | pure_v3_ms | hybrid_ms | Winner |
|---|---|---|---|---|---|---|
| 10 | 4.7 | True | 0.39 | 0.41 | **0.34*** | hybrid_ms (1.21x) |
| 10 | 4.7 | False | 0.39 | 0.36 | **0.33*** | hybrid_ms (1.18x) |
| 15 | 4.7 | True | 1.63 | **1.44*** | 2.29 | pure_v3_ms (1.59x) |
| 15 | 4.7 | False | 1.91 | 1.76 | **1.67*** | hybrid_ms (1.14x) |
| 20 | 4.7 | True | 37.94 | **34.21*** | 45.44 | pure_v3_ms (1.33x) |
| 20 | 4.7 | False | 54.26 | **47.98*** | 55.89 | pure_v3_ms (1.16x) |
| 25 | 4.7 | True | 2111.39 | **1489.29*** | 1613.79 | pure_v3_ms (1.42x) |
| 25 | 4.7 | False | 1769.10 | **1381.26*** | 1424.46 | pure_v3_ms (1.28x) |
| 27 | 4.7 | True | 8099.61 | **6486.80*** | 7175.03 | pure_v3_ms (1.25x) |
| 27 | 4.7 | False | 10549.00 | **9096.44*** | 9319.61 | pure_v3_ms (1.16x) |
| 10 | 4.7 | True | 1.81 | **0.38*** | 0.86 | pure_v3_ms (4.76x) |
| 10 | 4.7 | False | **0.55*** | 0.83 | 1.84 | pure_v2_ms (3.35x) |
| 15 | 4.7 | True | 3.32 | **1.99*** | 2.12 | pure_v3_ms (1.67x) |
| 15 | 4.7 | False | **2.50*** | 11.96 | 7.64 | pure_v2_ms (4.78x) |
| 20 | 4.7 | True | **166.01*** | 174.29 | 202.86 | pure_v2_ms (1.22x) |
| 20 | 4.7 | False | 203.85 | 193.77 | **120.47*** | hybrid_ms (1.69x) |
| 25 | 4.7 | True | 3881.69 | **2795.00*** | 2937.98 | pure_v3_ms (1.39x) |
| 25 | 4.7 | False | 3212.80 | 2545.66 | **1952.41*** | hybrid_ms (1.65x) |
| 27 | 4.7 | True | 8426.12 | **6367.70*** | 6968.77 | pure_v3_ms (1.32x) |
| 27 | 4.7 | False | 7252.46 | **5866.54*** | 6019.93 | pure_v3_ms (1.24x) |

### Legend
- **Bold*** = Fastest for this parameter combination
- Winner column shows speedup vs slowest

## Summary Statistics

### pure_v2_ms
- Mean: 2288.84 ms
- Min: 0.39 ms
- Max: 10549.00 ms
- Std: 3382.00 ms

### pure_v3_ms
- Mean: 1824.90 ms
- Min: 0.36 ms
- Max: 9096.44 ms
- Std: 2753.83 ms

### hybrid_ms
- Mean: 1892.69 ms
- Min: 0.33 ms
- Max: 9319.61 ms
- Std: 2903.72 ms
