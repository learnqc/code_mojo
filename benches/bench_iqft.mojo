import benchmark
from butterfly.core.state import (
    QuantumState,
    iqft_interval,
    iqft_simd,
    generate_state,
)
from butterfly.algos.value_encoding import encode_value_interval, iqft_via_fft
from butterfly.utils.visualization import print_state
from butterfly.core.classical_fft import (
    fft_dit,
    fft_dif,
    fft_dif_parallel,
    fft_dif_parallel_simd,
    fft_dif_parallel_simd_ndbuffer,
    fft_dif_parallel_ndbuffer,
    fft_dif_parallel_fastdiv,
)

from butterfly.core.fft import fft
from butterfly.core.fft_numpy_style import fft_numpy_style_simd, fft_numpy_style
from butterfly.core.fft_fma_optimized import fft_fma_opt


fn main() raises:
    print_state(encode_value_interval[21](4.7))
    alias n = 21
    # Initialize random_state once locally
    var random_state = generate_state(n)

    print("Benchmarking IQFT n={}...".format(String(n)))

    @parameter
    fn bench_iqft_interval_no_swap():
        var state = random_state
        iqft_interval(state, [j for j in range(n)], swap=False)

    @parameter
    fn bench_iqft_interval_with_swap():
        var state = random_state
        iqft_interval(state, [j for j in range(n)], swap=True)

    @parameter
    fn bench_iqft_simd_no_swap():
        var state = random_state
        iqft_simd[1 << n](state, [j for j in range(n)], swap=False)

    @parameter
    fn bench_iqft_simd_with_swap():
        var state = random_state
        iqft_simd[1 << n](state, [j for j in range(n)], swap=True)

    @parameter
    fn bench_fft():
        var state = random_state
        benchmark.keep(state[0])
        # fft(state)
        # iqft_via_fft(state)
        # qfft(state)
        # fft_numpy_style_simd(state)
        # fft_numpy_style(state)
        # fft_fma_opt[1 << n](state)
        # fft_dit(state)
        # fft_dif(state)
        fft_dif_parallel(state)
        # fft_dif_parallel_fastdiv(state)
        # fft_dif_parallel_simd(state)
        # fft_dif_parallel_simd_ndbuffer(state)
        # fft_dif_parallel_ndbuffer(state)

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

    var report_fft = benchmark.run[bench_fft](5, 100)
    print(
        "FFT:",
        report_fft.mean(benchmark.Unit.ms),
        "ms",
    )
