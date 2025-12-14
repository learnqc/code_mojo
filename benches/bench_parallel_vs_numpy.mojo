import benchmark
from python import Python
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.classical_fft import (
    fft_dif,
    fft_dif_parallel,
    fft_dif_parallel_simd,
    fft_dif_parallel_simd_ndbuffer,
)
from butterfly.core.types import FloatType


fn main() raises:
    # Initialize Python
    var np = Python.import_module("numpy")
    var time = Python.import_module("time")

    # Create signal entirely in Python to avoid operator issues
    var init_signal_code = (
        "lambda size, np: np.random.rand(size) + 1j * np.random.rand(size)"
    )
    var make_signal = Python.evaluate(init_signal_code)

    print(
        "Benchmarking Scalar vs Parallel vs Par-SIMD vs Par-NDBuffer vs NumPy"
        " (n=8 to 21)"
    )
    print(
        "-------------------------------------------------------------------------------------------------------------"
    )
    print(
        "n    | Size      | Scalar (ms) | Parallel (ms) | Par-SIMD (ms) |"
        " Par-NDBuf (ms) | NumPy (ms) | Speedup"
    )
    print(
        "-----|-----------|-------------|---------------|---------------|----------------|------------|--------"
    )

    # Range of n to benchmark
    for n in range(1, 23):
        var size = 1 << n

        # Prepare Mojo state
        var mojo_state = generate_state(n, seed=42)

        # Prepare NumPy signal
        var np_signal = make_signal(size, np)

        @parameter
        fn bench_mojo_scalar():
            var s = mojo_state
            fft_dif(s)
            benchmark.keep(s.re[0])

        @parameter
        fn bench_mojo_parallel():
            var s = mojo_state
            fft_dif_parallel(s)
            benchmark.keep(s.re[0])

        @parameter
        fn bench_mojo_parallel_simd():
            var s = mojo_state
            fft_dif_parallel_simd(s)
            benchmark.keep(s.re[0])

        @parameter
        fn bench_mojo_parallel_ndbuffer():
            var s = mojo_state
            fft_dif_parallel_simd_ndbuffer(s)
            benchmark.keep(s.re[0])

        @parameter
        fn bench_numpy():
            try:
                var res = np.fft.fft(np_signal)
                _ = res
            except:
                pass

        var report_scalar = benchmark.run[bench_mojo_scalar](5, 100)
        var report_parallel = benchmark.run[bench_mojo_parallel](5, 100)
        # Using fewer iterations for large n might be wise, but keeping consistent for now
        var report_parallel_simd = benchmark.run[bench_mojo_parallel_simd](
            5, 100
        )
        var report_parallel_ndbuffer = benchmark.run[
            bench_mojo_parallel_ndbuffer
        ](5, 100)
        var report_numpy = benchmark.run[bench_numpy](5, 100)

        var t_scalar = report_scalar.mean(benchmark.Unit.ms)
        var t_parallel = report_parallel.mean(benchmark.Unit.ms)
        var t_parallel_simd = report_parallel_simd.mean(benchmark.Unit.ms)
        var t_parallel_ndbuffer = report_parallel_ndbuffer.mean(
            benchmark.Unit.ms
        )
        var t_numpy = report_numpy.mean(benchmark.Unit.ms)

        # Find best mojo time
        var best_mojo = t_parallel  # Default for large n
        var algo = String(" (Par)")
        if t_scalar < best_mojo:
            best_mojo = t_scalar
            algo = String(" (Scalar)")
        if t_parallel_simd < best_mojo:
            best_mojo = t_parallel_simd
            algo = String(" (ParSIMD)")
        if t_parallel_ndbuffer < best_mojo:
            best_mojo = t_parallel_ndbuffer
            algo = String(" (NDBuf)")

        var speedup = t_numpy / best_mojo

        print(
            n,
            "   |",
            size,
            "   |",
            metrics_format(t_scalar),
            "    |",
            metrics_format(t_parallel),
            "      |",
            metrics_format(t_parallel_simd),
            "      |",
            metrics_format(t_parallel_ndbuffer),
            "       |",
            metrics_format(t_numpy),
            "   |",
            metrics_format(speedup) + algo,
        )


# Simple number formatting helper
fn metrics_format(val: Float64) -> String:
    # Hacky formatting since we don't have full f-string support for float precision easily
    # Just simple padding manually?
    # Or just print raw for now.
    return String(val)
