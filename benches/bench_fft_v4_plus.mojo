"""
Benchmark FFT V4 Plus against V4 to measure improvement.
"""
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.fft_v4 import fft_v4
from butterfly.core.fft_v4_plus import fft_v4_plus
from benchmark import keep, run, Unit


fn run_bench[n: Int]() raises:
    print(
        "--------------------------------------------------------------------------------"
    )
    print("Benchmarking N =", n)
    var state = QuantumState(n)

    @parameter
    fn bench_v4():
        fft_v4(state, block_log=12)
        keep(state.re.unsafe_ptr())

    @parameter
    fn bench_v4_plus():
        fft_v4_plus(state, block_log=12)
        keep(state.re.unsafe_ptr())

    print("  FFT V4 (Original):")
    var report_v4 = run[bench_v4](2, 7)
    report_v4.print(Unit.ms)

    print("  FFT V4 Plus (Optimized):")
    var report_v4_plus = run[bench_v4_plus](2, 5)
    report_v4_plus.print(Unit.ms)

    var speedup = report_v4.mean(Unit.ns) / report_v4_plus.mean(Unit.ns)
    var improvement = (speedup - 1.0) * 100.0

    print("  Speedup:        ", speedup, "x")
    if speedup > 1.0:
        print("  Improvement:    +", improvement, "%")
    else:
        print("  Regression:     ", improvement, "%")


fn main() raises:
    print("=" * 80)
    print("FFT V4 vs V4 Plus Performance Comparison")
    print("=" * 80)
    print()
    print("V4 Plus optimizations:")
    print("  1. Vectorized/parallelized scaling")
    print("  2. Specialized SIMD kernels for small strides (1, 2, 4)")
    print("  3. FMA (Fused Multiply-Add) operations")
    print("  4. Optimized buffer management")
    print()
    print("=" * 80)
    print()

    # Small N (exercises small-stride optimizations)
    run_bench[10]()

    # Medium N
    run_bench[15]()

    # Large N (exercises all optimizations)
    run_bench[20]()

    # Very large N
    run_bench[25]()

    print()
    print("=" * 80)
    print("Summary:")
    print(
        "  V4 Plus targets the scalar fallback bottleneck identified in"
        " profiling"
    )
    print("  Expected improvement: 20-50% depending on problem size")
    print("=" * 80)
