"""
Focused benchmark on twiddle factor computation overhead.
Tests the cost of cos/sin computation in tight loops.
"""

from butterfly.core.state import QuantumState, generate_state
from butterfly.core.types import *
from butterfly.core.gates import cis
from time import perf_counter_ns
from math import log2, cos, sin


fn apply_phase_cis_each(mut state: QuantumState, start: Int, count: Int, angle: FloatType):
    """Call cis() for each element - most overhead."""
    for i in range(count):
        var idx = start + i
        state[idx] = state[idx] * cis(angle)


fn apply_phase_cossin_each(mut state: QuantumState, start: Int, count: Int, angle: FloatType):
    """Call cos/sin for each element - medium overhead."""
    for i in range(count):
        var idx = start + i
        var re = state.re[idx]
        var im = state.im[idx]
        var c = cos(angle)
        var s = sin(angle)
        state.re[idx] = re * c - im * s
        state.im[idx] = re * s + im * c


fn apply_phase_precomputed(mut state: QuantumState, start: Int, count: Int, angle: FloatType):
    """Precompute cos/sin once (FFT-style) - minimal overhead."""
    var w_re = cos(angle)
    var w_im = sin(angle)

    for i in range(count):
        var idx = start + i
        var re = state.re[idx]
        var im = state.im[idx]
        state.re[idx] = re * w_re - im * w_im
        state.im[idx] = re * w_im + im * w_re


fn benchmark_cis_each[N: Int](count: Int, num_iterations: Int) -> Float64:
    """Benchmark cis() per element."""
    var total_time: Float64 = 0.0
    var angle = FloatType(0.5)

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        apply_phase_cis_each(state, 0, count, angle)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn benchmark_cossin_each[N: Int](count: Int, num_iterations: Int) -> Float64:
    """Benchmark cos/sin per element."""
    var total_time: Float64 = 0.0
    var angle = FloatType(0.5)

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        apply_phase_cossin_each(state, 0, count, angle)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn benchmark_precomputed[N: Int](count: Int, num_iterations: Int) -> Float64:
    """Benchmark precomputed twiddle."""
    var total_time: Float64 = 0.0
    var angle = FloatType(0.5)

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        apply_phase_precomputed(state, 0, count, angle)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn print_comparison(name: String, count: Int, cis_time: Float64, cossin_time: Float64, precomp_time: Float64):
    """Print benchmark results."""
    print(name + " (", count, " elements):")
    print("  cis() each:      ", cis_time * 1e6, " μs")
    print("  cos/sin each:    ", cossin_time * 1e6, " μs")
    print("  Precomputed:     ", precomp_time * 1e6, " μs")

    var speedup_cossin = cis_time / cossin_time
    var speedup_precomp = cis_time / precomp_time
    var speedup_vs_cossin = cossin_time / precomp_time

    print("  cos/sin vs cis:       ", speedup_cossin, "x")
    print("  Precomputed vs cis:   ", speedup_precomp, "x (", (speedup_precomp - 1.0) * 100, "% faster)")
    print("  Precomputed vs cos/sin:", speedup_vs_cossin, "x (", (speedup_vs_cossin - 1.0) * 100, "% faster)")
    print()


fn main():
    print("=" * 80)
    print("Twiddle Factor Computation Overhead Benchmark")
    print("Measuring cost of cos/sin in tight loops")
    print("=" * 80)
    print()

    alias num_iterations = 5000

    # Small count - overhead may not matter
    print("Benchmarking with various element counts - ", num_iterations, " iterations")
    print("-" * 80)

    alias N256 = 256

    # Count = 4 (small)
    var cis_4 = benchmark_cis_each[N256](4, num_iterations)
    var cossin_4 = benchmark_cossin_each[N256](4, num_iterations)
    var precomp_4 = benchmark_precomputed[N256](4, num_iterations)
    print_comparison("N=256", 4, cis_4, cossin_4, precomp_4)

    # Count = 16
    var cis_16 = benchmark_cis_each[N256](16, num_iterations)
    var cossin_16 = benchmark_cossin_each[N256](16, num_iterations)
    var precomp_16 = benchmark_precomputed[N256](16, num_iterations)
    print_comparison("N=256", 16, cis_16, cossin_16, precomp_16)

    # Count = 64
    var cis_64 = benchmark_cis_each[N256](64, num_iterations)
    var cossin_64 = benchmark_cossin_each[N256](64, num_iterations)
    var precomp_64 = benchmark_precomputed[N256](64, num_iterations)
    print_comparison("N=256", 64, cis_64, cossin_64, precomp_64)

    # Count = 128
    var cis_128 = benchmark_cis_each[N256](128, num_iterations)
    var cossin_128 = benchmark_cossin_each[N256](128, num_iterations)
    var precomp_128 = benchmark_precomputed[N256](128, num_iterations)
    print_comparison("N=256", 128, cis_128, cossin_128, precomp_128)

    # Count = 256 (full state)
    alias N1024 = 1024
    var cis_256 = benchmark_cis_each[N1024](256, num_iterations)
    var cossin_256 = benchmark_cossin_each[N1024](256, num_iterations)
    var precomp_256 = benchmark_precomputed[N1024](256, num_iterations)
    print_comparison("N=1024", 256, cis_256, cossin_256, precomp_256)

    # Count = 512
    var cis_512 = benchmark_cis_each[N1024](512, num_iterations)
    var cossin_512 = benchmark_cossin_each[N1024](512, num_iterations)
    var precomp_512 = benchmark_precomputed[N1024](512, num_iterations)
    print_comparison("N=1024", 512, cis_512, cossin_512, precomp_512)

    print("=" * 80)
    print("Benchmark completed!")
    print("=" * 80)
    print()
    print("Key Insights:")
    print("  - cis() creates Amplitude objects, adding allocation overhead")
    print("  - cos/sin each iteration is expensive (transcendental functions)")
    print("  - Precomputed eliminates cos/sin overhead from inner loop")
    print("  - Benefit increases with element count")
