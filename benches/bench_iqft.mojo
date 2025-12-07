import benchmark
from butterfly.core.state import (
    QuantumState,
    iqft_interval,
    iqft_simd,
    generate_state,
)
from butterfly.utils.visualization import print_state


fn main() raises:
    alias n = 14
    print("Benchmarking IQFT n={}...".format(String(n)))

    # Initialize random_state once locally
    var random_state = generate_state(n)

    @parameter
    fn bench_iqft_interval_no_swap():
        try:
            var state = random_state
            iqft_interval(state, [j for j in range(n)], swap=False)
        except:
            pass

    @parameter
    fn bench_iqft_interval_with_swap():
        try:
            var state = random_state
            iqft_interval(state, [j for j in range(n)], swap=True)
        except:
            pass

    @parameter
    fn bench_iqft_simd_no_swap():
        try:
            var state = random_state
            iqft_simd[1 << n](state, [j for j in range(n)], swap=False)
        except:
            pass

    @parameter
    fn bench_iqft_simd_with_swap():
        try:
            var state = random_state
            iqft_simd[1 << n](state, [j for j in range(n)], swap=True)
        except:
            pass

    var report_interval_no_swap = benchmark.run[bench_iqft_interval_no_swap](
        5, 100
    )
    # report_no_swap.print_full(unit=benchmark.Unit.ms)
    print(
        "Interval No Swap:",
        report_interval_no_swap.mean(benchmark.Unit.ms),
        "ms",
    )

    var report_interval_swap = benchmark.run[bench_iqft_interval_with_swap](
        5, 100
    )
    print(
        "Interval With Swap:",
        report_interval_swap.mean(benchmark.Unit.ms),
        "ms",
    )

    var report_simd_no_swap = benchmark.run[bench_iqft_simd_no_swap](5, 100)
    # report_no_swap.print_full(unit=benchmark.Unit.ms)
    print(
        "Simd No Swap:",
        report_simd_no_swap.mean(benchmark.Unit.ms),
        "ms",
    )

    var report_simd_swap = benchmark.run[bench_iqft_simd_with_swap](5, 100)
    print(
        "Simd With Swap:",
        report_simd_swap.mean(benchmark.Unit.ms),
        "ms",
    )
