# Benchmark Report
**Generated:** 2025-12-24 18:43:02
**Date(s):** 2025_12_24
**Total runs:** 19
**Parameters:** n, value
**Benchmarks:** execute, execute_simd, execute_simd_v2, execute_fused_v3

## Results

| n | value | execute | execute_simd | execute_simd_v2 | execute_fused_v3 | Winner |
|---|---|---|---|---|---|---|
| 10 | 4.7 | **0.10*** | 0.29 | 0.41 | 0.34 | execute (4.10x) |
| 11 | 4.7 | **0.19*** | 0.41 | 0.47 | 0.45 | execute (2.47x) |
| 12 | 4.7 | **0.41*** | 0.58 | 0.56 | 0.58 | execute (1.41x) |
| 13 | 4.7 | 0.92 | 0.94 | **0.76*** | 0.96 | execute_simd_v2 (1.26x) |
| 14 | 4.7 | 2.02 | 1.54 | **1.03*** | 1.35 | execute_simd_v2 (1.96x) |
| 15 | 4.7 | 4.49 | 2.71 | **1.60*** | 2.72 | execute_simd_v2 (2.81x) |
| 16 | 4.7 | 9.95 | 5.24 | **2.29*** | 5.16 | execute_simd_v2 (4.34x) |
| 17 | 4.7 | 21.84 | 10.64 | **4.30*** | 12.18 | execute_simd_v2 (5.08x) |
| 18 | 4.7 | 48.36 | 22.29 | **8.05*** | 30.73 | execute_simd_v2 (6.01x) |
| 19 | 4.7 | 106.66 | 53.60 | **15.48*** | 49.06 | execute_simd_v2 (6.89x) |
| 20 | 4.7 | 234.67 | 101.66 | **33.20*** | 107.85 | execute_simd_v2 (7.07x) |
| 21 | 4.7 | 521.25 | 221.88 | **83.48*** | 252.11 | execute_simd_v2 (6.24x) |
| 22 | 4.7 | 1128.71 | 468.86 | **170.54*** | 508.55 | execute_simd_v2 (6.62x) |
| 23 | 4.7 | 2457.71 | 1012.20 | **342.50*** | 1100.95 | execute_simd_v2 (7.18x) |
| 24 | 4.7 | 5307.77 | 2135.99 | **744.66*** | 2508.72 | execute_simd_v2 (7.13x) |
| 25 | 4.7 | 11397.12 | 4558.09 | **1513.00*** | 5135.84 | execute_simd_v2 (7.53x) |
| 26 | 4.7 | 24413.15 | 9927.07 | **3308.74*** | 11186.05 | execute_simd_v2 (7.38x) |
| 27 | 4.7 | 52209.90 | 20982.68 | **6733.84*** | 24191.93 | execute_simd_v2 (7.75x) |
| 28 | 4.7 | 111100.14 | 46179.87 | **14615.72*** | 50456.55 | execute_simd_v2 (7.60x) |

### Legend
- **Bold*** = Fastest for this parameter combination
- Winner column shows speedup vs slowest

## Summary Statistics

### execute
- Mean: 10998.18 ms
- Min: 0.10 ms
- Max: 111100.14 ms
- Std: 26687.87 ms

### execute_simd
- Mean: 4509.82 ms
- Min: 0.29 ms
- Max: 46179.87 ms
- Std: 11030.14 ms

### execute_simd_v2
- Mean: 1451.61 ms
- Min: 0.41 ms
- Max: 14615.72 ms
- Std: 3500.93 ms

### execute_fused_v3
- Mean: 5029.06 ms
- Min: 0.34 ms
- Max: 50456.55 ms
- Std: 12161.03 ms
