"""
Benchmark v5 (Subspace) against v4 (Global Synthesis) for full-range FFT.
"""
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.fft_v4 import fft_v4
from butterfly.core.fft_v5 import apply_cft_v5_contiguous
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
        try:
            # v4: Global, Table-based + Packing + BitReverse + Scaling
            fft_v4(state, block_log=12)
        except:
            pass
        keep(state.re.unsafe_ptr())

    @parameter
    fn bench_v5():
        try:
            # v5: Subspace, On-the-fly + Local Swap
            # No scaling included in apply_cft_v5_contiguous yet
            apply_cft_v5_contiguous(state, targets, inverse=False, do_swap=True)
        except:
            pass
        keep(state.re.unsafe_ptr())

    print("  FFT V4 (Global Synthesis):")
    var report_v4 = run[bench_v4](2, 5)
    report_v4.print(Unit.ms)

    print("  CFT V5 (Subspace On-the-fly):")
    var report_v5 = run[bench_v5](2, 5)
    report_v5.print(Unit.ms)

    var speedup = report_v4.mean(Unit.ns) / report_v5.mean(Unit.ns)
    if speedup > 1.0:
        print("  v5 is ", speedup, "x FASTER than v4")
    else:
        print("  v5 is ", 1.0 / speedup, "x SLOWER than v4")


fn main() raises:
    # Small N
    run_bench[10]()
    # Medium N
    run_bench[15]()
    # Large N
    run_bench[20]()
