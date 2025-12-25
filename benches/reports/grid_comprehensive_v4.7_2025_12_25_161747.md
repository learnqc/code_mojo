# Benchmark Report
**Generated:** 2025-12-25 16:17:47
**Date(s):** 2025_12_25
**Total runs:** 12
**Parameters:** n, value
**Benchmarks:** generic_ms, simd_v2_ms, grid_2row_ms, grid_4row_ms, grid_8row_ms

## Results

| n | value | generic_ms | simd_v2_ms | grid_2row_ms | grid_4row_ms | grid_8row_ms | Winner |
|---|---|---|---|---|---|---|---|
| 3 | 4.7 | **0.00*** | 0.00 | 1.58 | 1.41 | 1.45 | generic_ms |
| 5 | 4.7 | **0.00*** | 0.00 | 2.74 | 2.59 | 2.55 | generic_ms |
| 7 | 4.7 | **0.00*** | 0.00 | 5.00 | 5.71 | 1.77 | generic_ms |
| 10 | 4.7 | **0.03*** | 0.79 | 0.13 | 4.70 | 8.50 | generic_ms (283.33x) |
| 12 | 4.7 | **0.14*** | 6.07 | 7.75 | 7.05 | 7.64 | generic_ms (55.36x) |
| 15 | 4.7 | **1.09*** | 6.31 | 8.14 | 10.47 | 10.20 | generic_ms (9.61x) |
| 17 | 4.7 | **5.26*** | 11.36 | 10.24 | 10.21 | 13.06 | generic_ms (2.48x) |
| 20 | 4.7 | 47.72 | 54.83 | **40.49*** | 81.85 | 56.40 | grid_2row_ms (2.02x) |
| 22 | 4.7 | 215.77 | 111.77 | 84.43 | **53.03*** | 145.97 | grid_4row_ms (4.07x) |
| 24 | 4.7 | 932.17 | 378.04 | 346.60 | **226.94*** | 297.43 | grid_4row_ms (4.11x) |
| 25 | 4.7 | 1868.23 | 703.28 | 717.13 | **459.90*** | 775.14 | grid_4row_ms (4.06x) |
| 27 | 4.7 | 8403.31 | 3412.37 | 3436.73 | **2523.56*** | 3051.07 | grid_4row_ms (3.33x) |

### Legend
- **Bold*** = Fastest for this parameter combination
- Winner column shows speedup vs slowest

## Summary Statistics

### generic_ms
- Mean: 956.14 ms
- Min: 0.00 ms
- Max: 8403.31 ms
- Std: 2310.36 ms

### simd_v2_ms
- Mean: 390.40 ms
- Min: 0.00 ms
- Max: 3412.37 ms
- Std: 934.01 ms

### grid_2row_ms
- Mean: 388.41 ms
- Min: 0.13 ms
- Max: 3436.73 ms
- Std: 941.81 ms

### grid_4row_ms
- Mean: 282.29 ms
- Min: 1.41 ms
- Max: 2523.56 ms
- Std: 688.29 ms

### grid_8row_ms
- Mean: 364.27 ms
- Min: 1.45 ms
- Max: 3051.07 ms
- Std: 838.20 ms
