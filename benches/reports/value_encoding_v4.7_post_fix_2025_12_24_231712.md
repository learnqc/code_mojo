# Benchmark Report
**Generated:** 2025-12-24 23:17:12
**Date(s):** 2025_12_24
**Total runs:** 25
**Parameters:** n, value
**Benchmarks:** execute, execute_simd, execute_simd_v2, execute_fused_v3, qiskit_ms

## Results

| n | value | execute | execute_simd | execute_simd_v2 | execute_fused_v3 | qiskit_ms | Winner |
|---|---|---|---|---|---|---|---|
| 3 | 4.7 | 0.03 | **0.01*** | 0.01 | 0.03 | 0.43 | execute_simd (43.00x) |
| 4 | 4.7 | 0.02 | **0.01*** | 0.01 | 0.08 | 0.61 | execute_simd (61.00x) |
| 5 | 4.7 | 0.02 | **0.01*** | 0.01 | 0.22 | 0.80 | execute_simd (80.00x) |
| 6 | 4.7 | 0.03 | **0.02*** | 0.02 | 0.11 | 1.08 | execute_simd (54.00x) |
| 7 | 4.7 | 0.03 | **0.02*** | 0.02 | 0.15 | 1.36 | execute_simd (68.00x) |
| 8 | 4.7 | **0.04*** | 0.04 | 0.05 | 0.19 | 1.77 | execute (44.25x) |
| 9 | 4.7 | 0.15 | **0.07*** | 0.09 | 0.28 | 2.01 | execute_simd (28.71x) |
| 10 | 4.7 | **0.15*** | 0.66 | 0.41 | 0.39 | 2.46 | execute (16.40x) |
| 11 | 4.7 | **0.27*** | 0.53 | 0.48 | 0.46 | 2.84 | execute (10.52x) |
| 12 | 4.7 | **0.51*** | 0.70 | 0.62 | 0.73 | 3.53 | execute (6.92x) |
| 13 | 4.7 | 0.98 | 1.04 | **0.79*** | 1.11 | 4.55 | execute_simd_v2 (5.76x) |
| 14 | 4.7 | 2.09 | 1.51 | **1.12*** | 1.45 | 5.95 | execute_simd_v2 (5.31x) |
| 15 | 4.7 | 4.55 | 2.61 | **1.74*** | 2.94 | 11.70 | execute_simd_v2 (6.72x) |
| 16 | 4.7 | 9.93 | 4.12 | **3.43*** | 5.62 | 14.80 | execute_simd_v2 (4.31x) |
| 17 | 4.7 | 22.12 | 12.07 | **5.98*** | 12.04 | 19.49 | execute_simd_v2 (3.70x) |
| 18 | 4.7 | 50.24 | 15.70 | **8.56*** | 25.32 | 27.96 | execute_simd_v2 (5.87x) |
| 19 | 4.7 | 107.32 | 32.50 | **16.95*** | 54.26 | 46.35 | execute_simd_v2 (6.33x) |
| 20 | 4.7 | 239.03 | 93.64 | 84.44 | 125.07 | **81.40*** | qiskit_ms (2.94x) |
| 21 | 4.7 | 522.47 | 160.69 | **84.83*** | 268.52 | 177.85 | execute_simd_v2 (6.16x) |
| 22 | 4.7 | 1142.58 | 347.99 | **200.09*** | 573.12 | 319.04 | execute_simd_v2 (5.71x) |
| 23 | 4.7 | 2504.08 | 687.51 | **364.79*** | 1166.71 | 604.53 | execute_simd_v2 (6.86x) |
| 24 | 4.7 | 5322.82 | 1470.27 | **809.72*** | 2568.26 | 1236.37 | execute_simd_v2 (6.57x) |
| 25 | 4.7 | 11601.29 | 3483.05 | **1696.84*** | 5515.67 | 2653.90 | execute_simd_v2 (6.84x) |
| 26 | 4.7 | 24933.72 | 7585.81 | **3533.50*** | 11561.31 | 4927.03 | execute_simd_v2 (7.06x) |
| 27 | 4.7 | 52852.10 | 15789.87 | **7293.09*** | 25273.78 | 10102.62 | execute_simd_v2 (7.25x) |

### Legend
- **Bold*** = Fastest for this parameter combination
- Winner column shows speedup vs slowest

## Summary Statistics

### execute
- Mean: 3972.66 ms
- Min: 0.02 ms
- Max: 52852.10 ms
- Std: 11298.37 ms

### execute_simd
- Mean: 1187.62 ms
- Min: 0.01 ms
- Max: 15789.87 ms
- Std: 3385.42 ms

### execute_simd_v2
- Mean: 564.30 ms
- Min: 0.01 ms
- Max: 7293.09 ms
- Std: 1567.63 ms

### execute_fused_v3
- Mean: 1886.31 ms
- Min: 0.03 ms
- Max: 25273.78 ms
- Std: 5375.03 ms

### qiskit_ms
- Mean: 810.02 ms
- Min: 0.43 ms
- Max: 10102.62 ms
- Std: 2181.89 ms
