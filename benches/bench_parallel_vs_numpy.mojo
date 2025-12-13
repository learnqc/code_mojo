import benchmark
from python import Python
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.classical_fft import fft_dif, fft_dif_parallel
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

    print("Benchmarking Scalar vs Parallel FFT vs NumPy FFT (n=8 to 21)")
    print(
        "-----------------------------------------------------------------------------"
    )
    print(
        "n    | Size      | Scalar (ms) | Parallel (ms) | NumPy (ms) | Best"
        " Mojo vs NumPy"
    )
    print(
        "-----|-----------|-------------|---------------|------------|-------------------"
    )

    # Range of n to benchmark
    for n in range(1, 22):
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
        fn bench_numpy():
            try:
                var res = np.fft.fft(np_signal)
                _ = res
            except:
                pass

        var report_scalar = benchmark.run[bench_mojo_scalar](5, 100)
        var report_parallel = benchmark.run[bench_mojo_parallel](5, 100)
        var report_numpy = benchmark.run[bench_numpy](5, 100)

        var t_scalar = report_scalar.mean(benchmark.Unit.ms)
        var t_parallel = report_parallel.mean(benchmark.Unit.ms)
        var t_numpy = report_numpy.mean(benchmark.Unit.ms)

        var best_mojo = t_scalar if t_scalar < t_parallel else t_parallel
        var speedup = t_numpy / best_mojo

        var algo = String(" (Scalar)") if t_scalar < t_parallel else String(
            " (Par)"
        )

        print(
            n,
            "   |",
            size,
            "   |",
            metrics_format(t_scalar),
            "    |",
            metrics_format(t_parallel),
            "      |",
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
