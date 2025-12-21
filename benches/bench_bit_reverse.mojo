"""
Benchmark Bit Reversal at N=25.
"""
from butterfly.core.state import QuantumState, bit_reverse_state
from benchmark import keep, run, Unit


fn bench_bit_reverse[n: Int]() raises:
    print(
        "--------------------------------------------------------------------------------"
    )
    print("Benchmarking Bit Reversal N =", n)
    var state = QuantumState(n)

    @parameter
    fn bench_parallel():
        bit_reverse_state(state, parallel=True)
        keep(state.re.unsafe_ptr())

    @parameter
    fn bench_scalar():
        bit_reverse_state(state, parallel=False)
        keep(state.re.unsafe_ptr())

    print("  Parallel SIMD Bit Reverse:")
    var report_p = run[bench_parallel]()
    report_p.print(Unit.ms)

    # print("  Scalar Bit Reverse:")
    # var report_s = run[bench_scalar]()
    # report_s.print(Unit.ms)


fn main() raises:
    bench_bit_reverse[25]()
