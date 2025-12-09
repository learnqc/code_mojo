"""
Benchmark for realistic quantum computing sizes (up to 15 qubits = 32,768 elements).
Tests twiddle factor computation overhead at scale.
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
        var state = generate_state(Int(log2(Float64(N))), seed=i % 100)  # Limit seeds
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
        var state = generate_state(Int(log2(Float64(N))), seed=i % 100)
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
        var state = generate_state(Int(log2(Float64(N))), seed=i % 100)
        var start = perf_counter_ns()
        apply_phase_precomputed(state, 0, count, angle)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn print_comparison(name: String, count: Int, cis_time: Float64, cossin_time: Float64, precomp_time: Float64):
    """Print benchmark results."""
    print(name + " (", count, " elements):")
    print("  cis() each:         ", cis_time * 1e3, " ms")
    print("  cos/sin each:       ", cossin_time * 1e3, " ms")
    print("  Precomputed (FFT):  ", precomp_time * 1e3, " ms")

    var speedup_cossin = cis_time / cossin_time
    var speedup_precomp = cis_time / precomp_time
    var speedup_vs_cossin = cossin_time / precomp_time

    if speedup_precomp > 1.0:
        print("  Precomputed vs cis:    ", speedup_precomp, "x (", (speedup_precomp - 1.0) * 100, "% FASTER)")
    else:
        print("  Precomputed vs cis:    ", speedup_precomp, "x (", (1.0 - speedup_precomp) * 100, "% slower)")

    if speedup_vs_cossin > 1.0:
        print("  Precomputed vs cos/sin:", speedup_vs_cossin, "x (", (speedup_vs_cossin - 1.0) * 100, "% FASTER)")
    else:
        print("  Precomputed vs cos/sin:", speedup_vs_cossin, "x (", (1.0 - speedup_vs_cossin) * 100, "% slower)")
    print()


fn main():
    print("=" * 85)
    print("Large-Scale Twiddle Factor Benchmark (Realistic Quantum Sizes)")
    print("Testing up to 15 qubits (32,768 elements)")
    print("=" * 85)
    print()

    # 10 qubits = 1,024 elements
    print("=" * 85)
    print("10 QUBITS (N=1,024 elements)")
    print("=" * 85)
    alias N1024 = 1024
    alias iter_1024 = 500

    var cis_1k = benchmark_cis_each[N1024](1024, iter_1024)
    var cossin_1k = benchmark_cossin_each[N1024](1024, iter_1024)
    var precomp_1k = benchmark_precomputed[N1024](1024, iter_1024)
    print_comparison("Full state", 1024, cis_1k, cossin_1k, precomp_1k)

    # Half state
    var cis_512 = benchmark_cis_each[N1024](512, iter_1024)
    var cossin_512 = benchmark_cossin_each[N1024](512, iter_1024)
    var precomp_512 = benchmark_precomputed[N1024](512, iter_1024)
    print_comparison("Half state", 512, cis_512, cossin_512, precomp_512)

    # 12 qubits = 4,096 elements
    print("=" * 85)
    print("12 QUBITS (N=4,096 elements)")
    print("=" * 85)
    alias N4096 = 4096
    alias iter_4096 = 200

    var cis_4k = benchmark_cis_each[N4096](4096, iter_4096)
    var cossin_4k = benchmark_cossin_each[N4096](4096, iter_4096)
    var precomp_4k = benchmark_precomputed[N4096](4096, iter_4096)
    print_comparison("Full state", 4096, cis_4k, cossin_4k, precomp_4k)

    # Half state
    var cis_2k = benchmark_cis_each[N4096](2048, iter_4096)
    var cossin_2k = benchmark_cossin_each[N4096](2048, iter_4096)
    var precomp_2k = benchmark_precomputed[N4096](2048, iter_4096)
    print_comparison("Half state", 2048, cis_2k, cossin_2k, precomp_2k)

    # 13 qubits = 8,192 elements
    print("=" * 85)
    print("13 QUBITS (N=8,192 elements)")
    print("=" * 85)
    alias N8192 = 8192
    alias iter_8192 = 100

    var cis_8k = benchmark_cis_each[N8192](8192, iter_8192)
    var cossin_8k = benchmark_cossin_each[N8192](8192, iter_8192)
    var precomp_8k = benchmark_precomputed[N8192](8192, iter_8192)
    print_comparison("Full state", 8192, cis_8k, cossin_8k, precomp_8k)

    # Half state
    var cis_4k_half = benchmark_cis_each[N8192](4096, iter_8192)
    var cossin_4k_half = benchmark_cossin_each[N8192](4096, iter_8192)
    var precomp_4k_half = benchmark_precomputed[N8192](4096, iter_8192)
    print_comparison("Half state", 4096, cis_4k_half, cossin_4k_half, precomp_4k_half)

    # 14 qubits = 16,384 elements
    print("=" * 85)
    print("14 QUBITS (N=16,384 elements)")
    print("=" * 85)
    alias N16384 = 16384
    alias iter_16384 = 50

    var cis_16k = benchmark_cis_each[N16384](16384, iter_16384)
    var cossin_16k = benchmark_cossin_each[N16384](16384, iter_16384)
    var precomp_16k = benchmark_precomputed[N16384](16384, iter_16384)
    print_comparison("Full state", 16384, cis_16k, cossin_16k, precomp_16k)

    # Half state
    var cis_8k_half = benchmark_cis_each[N16384](8192, iter_16384)
    var cossin_8k_half = benchmark_cossin_each[N16384](8192, iter_16384)
    var precomp_8k_half = benchmark_precomputed[N16384](8192, iter_16384)
    print_comparison("Half state", 8192, cis_8k_half, cossin_8k_half, precomp_8k_half)

    # 15 qubits = 32,768 elements
    print("=" * 85)
    print("15 QUBITS (N=32,768 elements) - Maximum Target Size")
    print("=" * 85)
    alias N32768 = 32768
    alias iter_32768 = 25

    var cis_32k = benchmark_cis_each[N32768](32768, iter_32768)
    var cossin_32k = benchmark_cossin_each[N32768](32768, iter_32768)
    var precomp_32k = benchmark_precomputed[N32768](32768, iter_32768)
    print_comparison("Full state", 32768, cis_32k, cossin_32k, precomp_32k)

    # Half state
    var cis_16k_half = benchmark_cis_each[N32768](16384, iter_32768)
    var cossin_16k_half = benchmark_cossin_each[N32768](16384, iter_32768)
    var precomp_16k_half = benchmark_precomputed[N32768](16384, iter_32768)
    print_comparison("Half state", 16384, cis_16k_half, cossin_16k_half, precomp_16k_half)

    print("=" * 85)
    print("Benchmark completed!")
    print("=" * 85)
    print()
    print("Summary:")
    print("  - At large scales (1000+ elements), precomputed twiddle factors show")
    print("    significant performance benefits over repeated cos/sin computation")
    print("  - FFT-style precomputation is essential for quantum circuits with")
    print("    many controlled phase gates on large quantum states")
    print("  - The overhead of cos/sin calls scales linearly with element count")
