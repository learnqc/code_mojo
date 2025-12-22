"""
Benchmark v5 (Subspace) against v4 (Global Synthesis) for full-range FFT.
"""
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.fft_v4 import fft_v4
from butterfly.core.fft_v5 import apply_cft_v5_contiguous
from butterfly.core.fft_v5_junie import apply_cft_v5_contiguous as apply_cft_v5_junie
from benchmark import keep, run, Unit


fn run_bench[n: Int]() raises:
    print(
        "--------------------------------------------------------------------------------"
    )
    print("Benchmarking N =", n)
    var state = QuantumState(n)
    var targets = List[Int]()
    for i in range(n):
        targets.append(i)

    @parameter
    fn bench_v4():
        # v4: Global, Table-based + Packing + BitReverse + Scaling
        fft_v4(state, block_log=12)

        keep(state.re.unsafe_ptr())

    @parameter
    fn bench_v5():
        # v5: Subspace, On-the-fly + Local Swap
        # No scaling included in apply_cft_v5_contiguous yet
        apply_cft_v5_contiguous(state, targets, inverse=False, do_swap=True)

        keep(state.re.unsafe_ptr())

    @parameter
    fn bench_v5_junie():
        # v5_junie: Optimized Subspace (SIMD + Tables)
        apply_cft_v5_junie(state, targets, inverse=False, do_swap=True)

        keep(state.re.unsafe_ptr())

    print("  FFT V4 (Global Synthesis):")
    var report_v4 = run[bench_v4](2, 5)
    report_v4.print(Unit.ms)

    print("  CFT V5 (Original Subspace):")
    var report_v5 = run[bench_v5](2, 5)
    report_v5.print(Unit.ms)

    print("  CFT V5 JUNIE (Optimized Subspace):")
    var report_v5_junie = run[bench_v5_junie](2, 5)
    report_v5_junie.print(Unit.ms)

    var v5_speedup = report_v5.mean(Unit.ns) / report_v5_junie.mean(Unit.ns)
    print("  Junie v5 is ", v5_speedup, "x FASTER than Original v5")

    var v4_speedup = report_v4.mean(Unit.ns) / report_v5_junie.mean(Unit.ns)
    if v4_speedup > 1.0:
        print("  Junie v5 is ", v4_speedup, "x FASTER than v4")
    else:
        print("  Junie v5 is ", 1.0 / v4_speedup, "x SLOWER than v4")


fn main() raises:
    # Small N
    run_bench[10]()
    # Medium N
    run_bench[15]()
    # Large N
    run_bench[20]()

    print("\n" + "=" * 80)
    print("PARTIAL FFT BENCHMARK (k=10, N=15)")
    print("=" * 80)
    var state = QuantumState(15)
    var targets = List[Int]()
    for i in range(10):
        targets.append(i)

    @parameter
    fn bench_v5_partial():
        apply_cft_v5_contiguous(state, targets, inverse=False, do_swap=True)

        keep(state.re.unsafe_ptr())

    @parameter
    fn bench_v5_junie_partial():
        apply_cft_v5_junie(state, targets, inverse=False, do_swap=True)

        keep(state.re.unsafe_ptr())

    print("  CFT V5 (Original Subspace):")
    var report_v5 = run[bench_v5_partial](2, 5)
    report_v5.print(Unit.ms)

    print("  CFT V5 JUNIE (Optimized Subspace):")
    var report_v5_junie = run[bench_v5_junie_partial](2, 5)
    report_v5_junie.print(Unit.ms)

    var speedup = report_v5.mean(Unit.ns) / report_v5_junie.mean(Unit.ns)
    print("  Junie v5 is ", speedup, "x FASTER than Original v5")
