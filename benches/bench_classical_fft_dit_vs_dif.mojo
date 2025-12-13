import benchmark
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.classical_fft import (
    fft_dit,
    fft_dif,
    fft_dit_simd,
    fft_dif_simd,
    fft_dit_parallel,
    fft_dif_parallel,
)
from butterfly.core.types import FloatType
from testing import assert_almost_equal


fn verify_implementation(
    n: Int, name: String, func: fn (mut QuantumState, Bool) -> None
) raises:
    print("Verifying " + name + " for n =", n)
    var state1 = generate_state(n, seed=42)
    var state2 = generate_state(n, seed=42)

    fft_dit(state1)  # Baseline
    func(state2, False)

    var max_diff: FloatType = 0.0
    for i in range(state1.size()):
        var diff_re = abs(state1.re[i] - state2.re[i])
        var diff_im = abs(state1.im[i] - state2.im[i])
        max_diff = max(max_diff, diff_re)
        max_diff = max(max_diff, diff_im)

    print("Max difference:", max_diff)
    if max_diff < 1e-5:
        print("✅ Correctness PASSED")
    else:
        print("❌ Correctness FAILED")
        raise Error("Results do not match!")


fn main() raises:
    alias n = 14
    verify_implementation(n, "fft_dif", fft_dif)
    verify_implementation(n, "fft_dit_simd", fft_dit_simd)
    verify_implementation(n, "fft_dif_simd", fft_dif_simd)
    verify_implementation(n, "fft_dit_parallel", fft_dit_parallel)
    verify_implementation(n, "fft_dif_parallel", fft_dif_parallel)

    print("\nBenchmarking n =", n, "(", 1 << n, "elements)")

    # Pre-generate state to avoid measuring generation time
    # Note: In-place FFT requires copying the state for each run if we want to measure exactly the same transform.
    # However, FFT performance is generally data-independent (except for denormals maybe), so we can just reuse the state
    # or re-generate. Re-generating inside the benchmark loop might dominate time.
    # Copying is better.

    var base_state = generate_state(n, seed=123)

    @parameter
    fn bench_dit():
        var s = base_state
        fft_dit(s)
        benchmark.keep(s.re[0])

    @parameter
    fn bench_dif():
        var s = base_state
        fft_dif(s)
        benchmark.keep(s.re[0])

    @parameter
    fn bench_dit_simd():
        var s = base_state
        fft_dit_simd(s)
        benchmark.keep(s.re[0])

    @parameter
    fn bench_dif_simd():
        var s = base_state
        fft_dif_simd(s)
        benchmark.keep(s.re[0])

    @parameter
    fn bench_dit_parallel():
        var s = base_state
        fft_dit_parallel(s)
        benchmark.keep(s.re[0])

    @parameter
    fn bench_dif_parallel():
        var s = base_state
        fft_dif_parallel(s)
        benchmark.keep(s.re[0])

    var report_dit = benchmark.run[bench_dit](5, 100)
    print("fft_dit (Scalar):")
    report_dit.print_full(benchmark.Unit.ms)

    var report_dif = benchmark.run[bench_dif](5, 100)
    print("fft_dif (Scalar):")
    report_dif.print_full(benchmark.Unit.ms)

    var report_dit_simd = benchmark.run[bench_dit_simd](5, 100)
    print("fft_dit (SIMD):")
    report_dit_simd.print_full(benchmark.Unit.ms)

    var report_dif_simd = benchmark.run[bench_dif_simd](5, 100)
    print("fft_dif (SIMD):")
    report_dif_simd.print_full(benchmark.Unit.ms)

    var report_dit_parallel = benchmark.run[bench_dit_parallel](5, 100)
    print("fft_dit (Parallel):")
    report_dit_parallel.print_full(benchmark.Unit.ms)

    var report_dif_parallel = benchmark.run[bench_dif_parallel](5, 100)
    print("fft_dif (Parallel):")
    report_dif_parallel.print_full(benchmark.Unit.ms)

    var t_dit = report_dit.mean(benchmark.Unit.ms)
    var t_dif = report_dif.mean(benchmark.Unit.ms)
    var t_dit_simd = report_dit_simd.mean(benchmark.Unit.ms)
    var t_dif_simd = report_dif_simd.mean(benchmark.Unit.ms)
    var t_dit_parallel = report_dit_parallel.mean(benchmark.Unit.ms)
    var t_dif_parallel = report_dif_parallel.mean(benchmark.Unit.ms)

    print("\nSummary:")
    print("Scalar DIT: ", t_dit, "ms")
    print("Scalar DIF: ", t_dif, "ms")
    print("SIMD DIT:   ", t_dit_simd, "ms")
    print("SIMD DIF:   ", t_dif_simd, "ms")
    print("Parallel DIT:", t_dit_parallel, "ms")
    print("Parallel DIF:", t_dif_parallel, "ms")

    print("SIMD DIT Speedup: ", t_dit / t_dit_simd, "x")
    print("SIMD DIF Speedup: ", t_dif / t_dif_simd, "x")
    print("Parallel DIT Speedup: ", t_dit / t_dit_parallel, "x")
    print("Parallel DIF Speedup: ", t_dif / t_dif_parallel, "x")
