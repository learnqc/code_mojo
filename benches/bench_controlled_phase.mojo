"""
Realistic benchmark simulating controlled phase gate operations.
Measures the actual scenario in c_transform_interval_p where we process
multiple blocks of elements with phase rotations.
"""

from butterfly.core.state import QuantumState, generate_state, c_transform_interval_p
from butterfly.core.types import *
from butterfly.core.gates import cis
from time import perf_counter_ns
from math import log2, cos, sin


fn c_transform_interval_p_precomputed(
    mut state: QuantumState, control: Int, target: Int, angle: FloatType
):
    """FFT-style implementation with precomputed twiddle factor."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()

    # Precompute twiddle factor ONCE
    var w_re = cos(angle)
    var w_im = sin(angle)

    if target < control:
        for k in range(size // (2 * c_stride)):
            var block_start = k * 2 * c_stride + c_stride
            for j in range(c_stride // (2 * t_stride)):
                var sub_start = block_start + j * 2 * t_stride
                for idx in range(sub_start, sub_start + t_stride):
                    var re = state.re[idx + t_stride]
                    var im = state.im[idx + t_stride]
                    state.re[idx + t_stride] = re * w_re - im * w_im
                    state.im[idx + t_stride] = re * w_im + im * w_re
    else:
        for k in range(size // (2 * t_stride)):
            var base = k * 2 * t_stride
            var num_periods = t_stride // (2 * c_stride)
            for p in range(num_periods):
                var p_start = base + p * 2 * c_stride + c_stride
                for idx in range(p_start, p_start + c_stride):
                    var re = state.re[idx + t_stride]
                    var im = state.im[idx + t_stride]
                    state.re[idx + t_stride] = re * w_re - im * w_im
                    state.im[idx + t_stride] = re * w_im + im * w_re


fn benchmark_original[N: Int](
    control: Int, target: Int, angle: FloatType, num_iterations: Int
) -> Float64:
    """Benchmark original c_transform_interval_p with cis()."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i % 50)
        var start = perf_counter_ns()
        c_transform_interval_p(state, control, target, angle)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn benchmark_precomputed[N: Int](
    control: Int, target: Int, angle: FloatType, num_iterations: Int
) -> Float64:
    """Benchmark precomputed twiddle factor version."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i % 50)
        var start = perf_counter_ns()
        c_transform_interval_p_precomputed(state, control, target, angle)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn print_comparison(name: String, original: Float64, precomputed: Float64):
    """Print benchmark results."""
    print(name + ":")
    print("  Original (cis):     ", original * 1e3, " ms")
    print("  Precomputed (FFT):  ", precomputed * 1e3, " ms")

    var speedup = original / precomputed
    if speedup > 1.0:
        print("  Speedup:            ", speedup, "x")
        print("  Performance gain:   ", (speedup - 1.0) * 100, "% FASTER")
    else:
        print("  Slowdown:           ", 1.0 / speedup, "x")
        print("  Performance loss:   ", (1.0 - speedup) * 100, "% slower")
    print()


fn main():
    print("=" * 85)
    print("Controlled Phase Gate Benchmark (c_transform_interval_p)")
    print("Realistic scenarios for quantum circuits up to 15 qubits")
    print("=" * 85)
    print()

    var angle = FloatType(0.5)

    # 10 qubits scenarios
    print("=" * 85)
    print("10 QUBITS (N=1,024)")
    print("=" * 85)
    alias N1024 = 1024
    alias iter_10 = 200

    # Scenario 1: control=7, target=3 (many small blocks)
    print("Scenario: control=7, target=3 (many small blocks)")
    print("-" * 85)
    var orig_10_1 = benchmark_original[N1024](7, 3, angle, iter_10)
    var pre_10_1 = benchmark_precomputed[N1024](7, 3, angle, iter_10)
    print_comparison("c=7, t=3", orig_10_1, pre_10_1)

    # Scenario 2: control=5, target=8 (large blocks)
    print("Scenario: control=5, target=8 (large blocks)")
    print("-" * 85)
    var orig_10_2 = benchmark_original[N1024](5, 8, angle, iter_10)
    var pre_10_2 = benchmark_precomputed[N1024](5, 8, angle, iter_10)
    print_comparison("c=5, t=8", orig_10_2, pre_10_2)

    # 12 qubits scenarios
    print("=" * 85)
    print("12 QUBITS (N=4,096)")
    print("=" * 85)
    alias N4096 = 4096
    alias iter_12 = 150

    # Scenario 1: control=9, target=4 (many small blocks)
    print("Scenario: control=9, target=4 (many small blocks)")
    print("-" * 85)
    var orig_12_1 = benchmark_original[N4096](9, 4, angle, iter_12)
    var pre_12_1 = benchmark_precomputed[N4096](9, 4, angle, iter_12)
    print_comparison("c=9, t=4", orig_12_1, pre_12_1)

    # Scenario 2: control=6, target=10 (large blocks)
    print("Scenario: control=6, target=10 (large blocks)")
    print("-" * 85)
    var orig_12_2 = benchmark_original[N4096](6, 10, angle, iter_12)
    var pre_12_2 = benchmark_precomputed[N4096](6, 10, angle, iter_12)
    print_comparison("c=6, t=10", orig_12_2, pre_12_2)

    # 13 qubits scenarios
    print("=" * 85)
    print("13 QUBITS (N=8,192)")
    print("=" * 85)
    alias N8192 = 8192
    alias iter_13 = 100

    # Scenario 1: control=10, target=5 (many blocks)
    print("Scenario: control=10, target=5 (many blocks)")
    print("-" * 85)
    var orig_13_1 = benchmark_original[N8192](10, 5, angle, iter_13)
    var pre_13_1 = benchmark_precomputed[N8192](10, 5, angle, iter_13)
    print_comparison("c=10, t=5", orig_13_1, pre_13_1)

    # Scenario 2: control=3, target=11 (large blocks)
    print("Scenario: control=3, target=11 (large blocks)")
    print("-" * 85)
    var orig_13_2 = benchmark_original[N8192](3, 11, angle, iter_13)
    var pre_13_2 = benchmark_precomputed[N8192](3, 11, angle, iter_13)
    print_comparison("c=3, t=11", orig_13_2, pre_13_2)

    # 14 qubits scenarios
    print("=" * 85)
    print("14 QUBITS (N=16,384)")
    print("=" * 85)
    alias N16384 = 16384
    alias iter_14 = 75

    # Scenario 1: control=11, target=6
    print("Scenario: control=11, target=6 (many blocks)")
    print("-" * 85)
    var orig_14_1 = benchmark_original[N16384](11, 6, angle, iter_14)
    var pre_14_1 = benchmark_precomputed[N16384](11, 6, angle, iter_14)
    print_comparison("c=11, t=6", orig_14_1, pre_14_1)

    # Scenario 2: control=4, target=12
    print("Scenario: control=4, target=12 (large blocks)")
    print("-" * 85)
    var orig_14_2 = benchmark_original[N16384](4, 12, angle, iter_14)
    var pre_14_2 = benchmark_precomputed[N16384](4, 12, angle, iter_14)
    print_comparison("c=4, t=12", orig_14_2, pre_14_2)

    # 15 qubits scenarios - MAXIMUM
    print("=" * 85)
    print("15 QUBITS (N=32,768) - MAXIMUM TARGET SIZE")
    print("=" * 85)
    alias N32768 = 32768
    alias iter_15 = 50

    # Scenario 1: control=12, target=7
    print("Scenario: control=12, target=7 (many blocks)")
    print("-" * 85)
    var orig_15_1 = benchmark_original[N32768](12, 7, angle, iter_15)
    var pre_15_1 = benchmark_precomputed[N32768](12, 7, angle, iter_15)
    print_comparison("c=12, t=7", orig_15_1, pre_15_1)

    # Scenario 2: control=5, target=13
    print("Scenario: control=5, target=13 (large blocks)")
    print("-" * 85)
    var orig_15_2 = benchmark_original[N32768](5, 13, angle, iter_15)
    var pre_15_2 = benchmark_precomputed[N32768](5, 13, angle, iter_15)
    print_comparison("c=5, t=13", orig_15_2, pre_15_2)

    # Scenario 3: control=8, target=10 (balanced)
    print("Scenario: control=8, target=10 (balanced)")
    print("-" * 85)
    var orig_15_3 = benchmark_original[N32768](8, 10, angle, iter_15)
    var pre_15_3 = benchmark_precomputed[N32768](8, 10, angle, iter_15)
    print_comparison("c=8, t=10", orig_15_3, pre_15_3)

    print("=" * 85)
    print("Benchmark completed!")
    print("=" * 85)
    print()
    print("Key Findings:")
    print("  - FFT-style precomputation eliminates repeated cos/sin calls")
    print("  - Benefit scales with number of elements affected")
    print("  - Most significant for controlled gates on large quantum states")
    print("  - Essential optimization for quantum algorithms like QFT, QPE")
