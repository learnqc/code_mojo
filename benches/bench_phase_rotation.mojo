from butterfly.core.state import QuantumState, generate_state
from butterfly.core.types import *
from butterfly.core.gates import cis
from time import perf_counter_ns
from math import log2, cos, sin


fn c_transform_interval_p_original(
    mut state: QuantumState, control: Int, target: Int, angle: FloatType
):
    """Original implementation using cis() for each element."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()

    if target < control:
        # Iterate over control blocks
        for k in range(size // (2 * c_stride)):
            var block_start = k * 2 * c_stride + c_stride
            for j in range(c_stride // (2 * t_stride)):
                var sub_start = block_start + j * 2 * t_stride
                for idx in range(sub_start, sub_start + t_stride):
                    state[idx + t_stride] = state[idx + t_stride] * cis(angle)
    else:
        # target > control
        for k in range(size // (2 * t_stride)):
            var base = k * 2 * t_stride
            var num_periods = t_stride // (2 * c_stride)
            for p in range(num_periods):
                var p_start = base + p * 2 * c_stride + c_stride
                for idx in range(p_start, p_start + c_stride):
                    state[idx + t_stride] = state[idx + t_stride] * cis(angle)


fn c_transform_interval_p_precomputed(
    mut state: QuantumState, control: Int, target: Int, angle: FloatType
):
    """FFT-style implementation with precomputed twiddle factor."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()

    # Precompute twiddle factor (like FFT does once per group)
    var w_re = cos(angle)
    var w_im = sin(angle)

    if target < control:
        # Iterate over control blocks
        for k in range(size // (2 * c_stride)):
            var block_start = k * 2 * c_stride + c_stride
            for j in range(c_stride // (2 * t_stride)):
                var sub_start = block_start + j * 2 * t_stride
                for idx in range(sub_start, sub_start + t_stride):
                    # Complex multiplication: state[idx + t_stride] *= (w_re + i*w_im)
                    var re = state.re[idx + t_stride]
                    var im = state.im[idx + t_stride]
                    state.re[idx + t_stride] = re * w_re - im * w_im
                    state.im[idx + t_stride] = re * w_im + im * w_re
    else:
        # target > control
        for k in range(size // (2 * t_stride)):
            var base = k * 2 * t_stride
            var num_periods = t_stride // (2 * c_stride)
            for p in range(num_periods):
                var p_start = base + p * 2 * c_stride + c_stride
                for idx in range(p_start, p_start + c_stride):
                    # Complex multiplication
                    var re = state.re[idx + t_stride]
                    var im = state.im[idx + t_stride]
                    state.re[idx + t_stride] = re * w_re - im * w_im
                    state.im[idx + t_stride] = re * w_im + im * w_re


fn c_transform_interval_p_inline(
    mut state: QuantumState, control: Int, target: Int, angle: FloatType
):
    """Implementation with inline complex multiplication, no cis() or Amplitude."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()

    if target < control:
        # Iterate over control blocks
        for k in range(size // (2 * c_stride)):
            var block_start = k * 2 * c_stride + c_stride
            for j in range(c_stride // (2 * t_stride)):
                var sub_start = block_start + j * 2 * t_stride
                for idx in range(sub_start, sub_start + t_stride):
                    # Inline: state *= cos(angle) + i*sin(angle)
                    var re = state.re[idx + t_stride]
                    var im = state.im[idx + t_stride]
                    var c = cos(angle)
                    var s = sin(angle)
                    state.re[idx + t_stride] = re * c - im * s
                    state.im[idx + t_stride] = re * s + im * c
    else:
        # target > control
        for k in range(size // (2 * t_stride)):
            var base = k * 2 * t_stride
            var num_periods = t_stride // (2 * c_stride)
            for p in range(num_periods):
                var p_start = base + p * 2 * c_stride + c_stride
                for idx in range(p_start, p_start + c_stride):
                    var re = state.re[idx + t_stride]
                    var im = state.im[idx + t_stride]
                    var c = cos(angle)
                    var s = sin(angle)
                    state.re[idx + t_stride] = re * c - im * s
                    state.im[idx + t_stride] = re * s + im * c


fn benchmark_original[N: Int](
    control: Int, target: Int, angle: FloatType, num_iterations: Int
) -> Float64:
    """Benchmark original implementation with cis()."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        c_transform_interval_p_original(state, control, target, angle)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn benchmark_precomputed[N: Int](
    control: Int, target: Int, angle: FloatType, num_iterations: Int
) -> Float64:
    """Benchmark precomputed twiddle factor implementation."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        c_transform_interval_p_precomputed(state, control, target, angle)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn benchmark_inline[N: Int](
    control: Int, target: Int, angle: FloatType, num_iterations: Int
) -> Float64:
    """Benchmark inline implementation."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        c_transform_interval_p_inline(state, control, target, angle)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn print_comparison(name: String, original: Float64, precomputed: Float64, inline: Float64):
    """Print benchmark results with speedup calculation."""
    print(name + ":")
    print("  Original (cis):     ", original * 1e6, " μs")
    print("  Precomputed twiddle:", precomputed * 1e6, " μs")
    print("  Inline cos/sin:     ", inline * 1e6, " μs")

    var speedup_pre = original / precomputed
    var speedup_inline = original / inline

    if speedup_pre > 1.0:
        print("  Precomputed speedup:", speedup_pre, "x (", (speedup_pre - 1.0) * 100, "% faster)")
    else:
        print("  Precomputed slowdown:", 1.0 / speedup_pre, "x (", (1.0 - speedup_pre) * 100, "% slower)")

    if speedup_inline > 1.0:
        print("  Inline speedup:     ", speedup_inline, "x (", (speedup_inline - 1.0) * 100, "% faster)")
    else:
        print("  Inline slowdown:    ", 1.0 / speedup_inline, "x (", (1.0 - speedup_inline) * 100, "% slower)")
    print()


fn main():
    print("=" * 80)
    print("Phase Rotation Benchmark (c_transform_interval_p)")
    print("Comparing twiddle factor computation strategies")
    print("=" * 80)
    print()

    alias num_iterations = 1000
    var angle = FloatType(0.5)  # Common phase angle

    # Benchmark N=256, control=5, target=2 (target < control)
    print("N=256, control=5, target=2 (target < control) - ", num_iterations, " iterations")
    print("-" * 80)
    alias N256 = 256
    var orig_1 = benchmark_original[N256](5, 2, angle, num_iterations)
    var pre_1 = benchmark_precomputed[N256](5, 2, angle, num_iterations)
    var inline_1 = benchmark_inline[N256](5, 2, angle, num_iterations)
    print_comparison("N=256, c=5, t=2", orig_1, pre_1, inline_1)

    # Benchmark N=256, control=2, target=5 (target > control)
    print("N=256, control=2, target=5 (target > control) - ", num_iterations, " iterations")
    print("-" * 80)
    var orig_2 = benchmark_original[N256](2, 5, angle, num_iterations)
    var pre_2 = benchmark_precomputed[N256](2, 5, angle, num_iterations)
    var inline_2 = benchmark_inline[N256](2, 5, angle, num_iterations)
    print_comparison("N=256, c=2, t=5", orig_2, pre_2, inline_2)

    # Benchmark N=1024, control=7, target=3 (target < control)
    print("N=1024, control=7, target=3 (target < control) - ", num_iterations, " iterations")
    print("-" * 80)
    alias N1024 = 1024
    var orig_3 = benchmark_original[N1024](7, 3, angle, num_iterations)
    var pre_3 = benchmark_precomputed[N1024](7, 3, angle, num_iterations)
    var inline_3 = benchmark_inline[N1024](7, 3, angle, num_iterations)
    print_comparison("N=1024, c=7, t=3", orig_3, pre_3, inline_3)

    # Benchmark N=1024, control=3, target=7 (target > control)
    print("N=1024, control=3, target=7 (target > control) - ", num_iterations, " iterations")
    print("-" * 80)
    var orig_4 = benchmark_original[N1024](3, 7, angle, num_iterations)
    var pre_4 = benchmark_precomputed[N1024](3, 7, angle, num_iterations)
    var inline_4 = benchmark_inline[N1024](3, 7, angle, num_iterations)
    print_comparison("N=1024, c=3, t=7", orig_4, pre_4, inline_4)

    # Benchmark N=4096, control=8, target=4 (target < control)
    print("N=4096, control=8, target=4 (target < control) - ", num_iterations, " iterations")
    print("-" * 80)
    alias N4096 = 4096
    var orig_5 = benchmark_original[N4096](8, 4, angle, num_iterations)
    var pre_5 = benchmark_precomputed[N4096](8, 4, angle, num_iterations)
    var inline_5 = benchmark_inline[N4096](8, 4, angle, num_iterations)
    print_comparison("N=4096, c=8, t=4", orig_5, pre_5, inline_5)

    # Benchmark N=4096, control=4, target=8 (target > control)
    print("N=4096, control=4, target=8 (target > control) - ", num_iterations, " iterations")
    print("-" * 80)
    var orig_6 = benchmark_original[N4096](4, 8, angle, num_iterations)
    var pre_6 = benchmark_precomputed[N4096](4, 8, angle, num_iterations)
    var inline_6 = benchmark_inline[N4096](4, 8, angle, num_iterations)
    print_comparison("N=4096, c=4, t=8", orig_6, pre_6, inline_6)

    print("=" * 80)
    print("Benchmark completed!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - Original: Uses cis() for each element (Amplitude multiply)")
    print("  - Precomputed: Computes cos/sin once, reuses twiddle (FFT-style)")
    print("  - Inline: Computes cos/sin per element but avoids Amplitude overhead")
    print()
    print("Expected: Precomputed should be fastest (single cos/sin computation)")
