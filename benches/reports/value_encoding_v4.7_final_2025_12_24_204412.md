# Benchmark Report
**Generated:** 2025-12-24 20:44:12
**Date(s):** 2025_12_24
**Total runs:** 25
**Parameters:** n, value
**Benchmarks:** execute, execute_simd, execute_simd_v2, execute_fused_v3, qiskit_ms

## Results

| n | value | execute | execute_simd | execute_simd_v2 | execute_fused_v3 | qiskit_ms | Winner |
|---|---|---|---|---|---|---|---|
| 3 | 4.7 | 0.01 | **0.00*** | 0.01 | 0.03 | 0.45 | execute_simd |
| 4 | 4.7 | **0.01*** | 0.01 | 0.01 | 0.06 | 0.60 | execute (60.00x) |
| 5 | 4.7 | **0.01*** | 0.01 | 0.01 | 0.09 | 0.79 | execute (79.00x) |
| 6 | 4.7 | **0.01*** | 0.01 | 0.01 | 0.12 | 1.03 | execute (103.00x) |
| 7 | 4.7 | **0.02*** | 0.02 | 0.02 | 0.14 | 1.30 | execute (65.00x) |
| 8 | 4.7 | **0.03*** | 0.03 | 0.03 | 0.20 | 1.64 | execute (54.67x) |
| 9 | 4.7 | **0.05*** | 0.05 | 0.05 | 0.25 | 2.01 | execute (40.20x) |
| 10 | 4.7 | **0.10*** | 0.31 | 0.37 | 0.32 | 2.46 | execute (24.60x) |
| 11 | 4.7 | **0.19*** | 0.39 | 0.53 | 0.45 | 3.08 | execute (16.21x) |
| 12 | 4.7 | **0.40*** | 0.77 | 0.86 | 0.83 | 3.63 | execute (9.07x) |
| 13 | 4.7 | 0.91 | 1.02 | **0.79*** | 1.28 | 4.52 | execute_simd_v2 (5.72x) |
| 14 | 4.7 | 2.02 | 2.68 | **1.55*** | 3.06 | 5.97 | execute_simd_v2 (3.85x) |
| 15 | 4.7 | 4.48 | 2.81 | **1.63*** | 2.66 | 11.43 | execute_simd_v2 (7.01x) |
| 16 | 4.7 | 9.92 | 5.24 | **2.45*** | 5.35 | 14.54 | execute_simd_v2 (5.93x) |
| 17 | 4.7 | 21.96 | 18.51 | **8.34*** | 16.64 | 18.95 | execute_simd_v2 (2.63x) |
| 18 | 4.7 | 48.62 | 23.46 | **8.36*** | 24.37 | 27.85 | execute_simd_v2 (5.82x) |
| 19 | 4.7 | 107.00 | 65.20 | **18.14*** | 50.97 | 46.77 | execute_simd_v2 (5.90x) |
| 20 | 4.7 | 238.87 | 125.23 | **33.87*** | 119.34 | 79.77 | execute_simd_v2 (7.05x) |
| 21 | 4.7 | 519.97 | 216.32 | **74.97*** | 251.80 | 150.80 | execute_simd_v2 (6.94x) |
| 22 | 4.7 | 1137.58 | 477.46 | **167.38*** | 523.11 | 299.76 | execute_simd_v2 (6.80x) |
| 23 | 4.7 | 2472.54 | 1050.94 | **382.23*** | 1128.24 | 631.77 | execute_simd_v2 (6.47x) |
| 24 | 4.7 | 5345.18 | 2273.61 | **834.17*** | 2660.78 | 1284.33 | execute_simd_v2 (6.41x) |
| 25 | 4.7 | 11501.53 | 4728.16 | **1561.31*** | 5285.57 | 2557.68 | execute_simd_v2 (7.37x) |
| 26 | 4.7 | 24470.93 | 10187.22 | **3343.17*** | 11777.20 | 4981.64 | execute_simd_v2 (7.32x) |
| 27 | 4.7 | 52322.71 | 21409.70 | **6917.73*** | 25083.09 | 10241.78 | execute_simd_v2 (7.56x) |

### Legend
- **Bold*** = Fastest for this parameter combination
- Winner column shows speedup vs slowest

## Summary Statistics

### execute
- Mean: 3928.20 ms
- Min: 0.01 ms
- Max: 52322.71 ms
- Std: 11170.19 ms

### execute_simd
- Mean: 1623.57 ms
- Min: 0.00 ms
- Max: 21409.70 ms
- Std: 4583.36 ms

### execute_simd_v2
- Mean: 534.32 ms
- Min: 0.01 ms
- Max: 6917.73 ms
- Std: 1486.06 ms

### execute_fused_v3
- Mean: 1877.44 ms
- Min: 0.03 ms
- Max: 25083.09 ms
- Std: 5353.09 ms

### qiskit_ms
- Mean: 814.98 ms
- Min: 0.45 ms
- Max: 10241.78 ms
- Std: 2207.43 ms
