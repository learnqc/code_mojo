from butterfly.core.state import QuantumState, generate_state
from butterfly.core.types import *
from butterfly.core.gates import H
from time import perf_counter_ns
from math import log2, sqrt


fn transform_h_original(mut state: QuantumState, target: Int):
    """Original transform_h implementation."""
    l = state.size()
    stride = 1 << target
    r = 0
    for j in range(l // 2):
        idx = 2 * j - r  # r = j%stride
        state[idx] = (state[idx] + state[idx + stride]) * sq_half
        state[idx + stride] = state[idx] - state[idx + stride] * sq2

        r += 1
        if r == stride:
            r = 0


fn transform_h_fft_style(mut state: QuantumState, target: Int):
    """FFT-style butterfly implementation for Hadamard gate.

    Uses the standard butterfly pattern: u +/- t
    For Hadamard with w=1: t = v, so we get (u+v)/√2 and (u-v)/√2
    """
    var l = state.size()
    var stride = 1 << target
    var scale = FloatType(1.0 / sqrt(2.0))

    var r = 0
    for j in range(l // 2):
        var idx = 2 * j - r  # r = j%stride

        # Load u and v
        var u_re = state.re[idx]
        var u_im = state.im[idx]
        var v_re = state.re[idx + stride]
        var v_im = state.im[idx + stride]

        # Butterfly: (u+v)/√2 and (u-v)/√2
        state.re[idx] = (u_re + v_re) * scale
        state.im[idx] = (u_im + v_im) * scale
        state.re[idx + stride] = (u_re - v_re) * scale
        state.im[idx + stride] = (u_im - v_im) * scale

        r += 1
        if r == stride:
            r = 0


fn transform_h_block_style(mut state: QuantumState, target: Int):
    """Block-based implementation avoiding modulo.

    Processes blocks of 2*stride elements, computing butterflies
    for the first stride elements in each block.
    """
    var l = state.size()
    var stride = 1 << target
    var scale = FloatType(1.0 / sqrt(2.0))

    for k in range(l // (2 * stride)):
        for idx in range(k * 2 * stride, k * 2 * stride + stride):
            # Load u and v
            var u_re = state.re[idx]
            var u_im = state.im[idx]
            var v_re = state.re[idx + stride]
            var v_im = state.im[idx + stride]

            # Butterfly: (u+v)/√2 and (u-v)/√2
            state.re[idx] = (u_re + v_re) * scale
            state.im[idx] = (u_im + v_im) * scale
            state.re[idx + stride] = (u_re - v_re) * scale
            state.im[idx + stride] = (u_im - v_im) * scale


fn benchmark_original[N: Int](target: Int, num_iterations: Int) -> Float64:
    """Benchmark original transform_h implementation."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        transform_h_original(state, target)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn benchmark_fft_style[N: Int](target: Int, num_iterations: Int) -> Float64:
    """Benchmark FFT-style butterfly implementation."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        transform_h_fft_style(state, target)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn benchmark_block_style[N: Int](target: Int, num_iterations: Int) -> Float64:
    """Benchmark block-based implementation."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        transform_h_block_style(state, target)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn print_comparison(name: String, original: Float64, fft: Float64, block: Float64):
    """Print benchmark results with speedup calculation."""
    print(name + ":")
    print("  Original:    ", original * 1e6, " μs")
    print("  FFT-style:   ", fft * 1e6, " μs")
    print("  Block-style: ", block * 1e6, " μs")

    var speedup_fft = original / fft
    var speedup_block = original / block

    if speedup_fft > 1.0:
        print("  FFT speedup:   ", speedup_fft, "x (", (speedup_fft - 1.0) * 100, "% faster)")
    else:
        print("  FFT slowdown:  ", 1.0 / speedup_fft, "x (", (1.0 - speedup_fft) * 100, "% slower)")

    if speedup_block > 1.0:
        print("  Block speedup: ", speedup_block, "x (", (speedup_block - 1.0) * 100, "% faster)")
    else:
        print("  Block slowdown:", 1.0 / speedup_block, "x (", (1.0 - speedup_block) * 100, "% slower)")
    print()


fn main():
    print("=" * 75)
    print("Hadamard Transform Benchmark")
    print("Comparing transform_h implementations")
    print("=" * 75)
    print()

    alias num_iterations = 1000

    # Benchmark N=256, target=0 (smallest stride)
    print("Benchmarking N=256 (8 qubits), target=0 - ", num_iterations, " iterations")
    print("-" * 75)
    alias N256 = 256
    var orig_256_0 = benchmark_original[N256](0, num_iterations)
    var fft_256_0 = benchmark_fft_style[N256](0, num_iterations)
    var block_256_0 = benchmark_block_style[N256](0, num_iterations)
    print_comparison("N=256, target=0 (stride=1)", orig_256_0, fft_256_0, block_256_0)

    # Benchmark N=256, target=4 (mid stride)
    print("Benchmarking N=256 (8 qubits), target=4 - ", num_iterations, " iterations")
    print("-" * 75)
    var orig_256_4 = benchmark_original[N256](4, num_iterations)
    var fft_256_4 = benchmark_fft_style[N256](4, num_iterations)
    var block_256_4 = benchmark_block_style[N256](4, num_iterations)
    print_comparison("N=256, target=4 (stride=16)", orig_256_4, fft_256_4, block_256_4)

    # Benchmark N=256, target=7 (largest stride)
    print("Benchmarking N=256 (8 qubits), target=7 - ", num_iterations, " iterations")
    print("-" * 75)
    var orig_256_7 = benchmark_original[N256](7, num_iterations)
    var fft_256_7 = benchmark_fft_style[N256](7, num_iterations)
    var block_256_7 = benchmark_block_style[N256](7, num_iterations)
    print_comparison("N=256, target=7 (stride=128)", orig_256_7, fft_256_7, block_256_7)

    # Benchmark N=1024, target=0
    print("Benchmarking N=1024 (10 qubits), target=0 - ", num_iterations, " iterations")
    print("-" * 75)
    alias N1024 = 1024
    var orig_1024_0 = benchmark_original[N1024](0, num_iterations)
    var fft_1024_0 = benchmark_fft_style[N1024](0, num_iterations)
    var block_1024_0 = benchmark_block_style[N1024](0, num_iterations)
    print_comparison("N=1024, target=0 (stride=1)", orig_1024_0, fft_1024_0, block_1024_0)

    # Benchmark N=1024, target=5
    print("Benchmarking N=1024 (10 qubits), target=5 - ", num_iterations, " iterations")
    print("-" * 75)
    var orig_1024_5 = benchmark_original[N1024](5, num_iterations)
    var fft_1024_5 = benchmark_fft_style[N1024](5, num_iterations)
    var block_1024_5 = benchmark_block_style[N1024](5, num_iterations)
    print_comparison("N=1024, target=5 (stride=32)", orig_1024_5, fft_1024_5, block_1024_5)

    # Benchmark N=1024, target=9
    print("Benchmarking N=1024 (10 qubits), target=9 - ", num_iterations, " iterations")
    print("-" * 75)
    var orig_1024_9 = benchmark_original[N1024](9, num_iterations)
    var fft_1024_9 = benchmark_fft_style[N1024](9, num_iterations)
    var block_1024_9 = benchmark_block_style[N1024](9, num_iterations)
    print_comparison("N=1024, target=9 (stride=512)", orig_1024_9, fft_1024_9, block_1024_9)

    print("=" * 75)
    print("Benchmark completed!")
    print("=" * 75)
    print()
    print("Summary:")
    print("  - Original: Uses (u+v)*scale, then reads stale value (BUG in line 2)")
    print("  - FFT-style: Loads u,v separately, computes (u+v)*scale and (u-v)*scale")
    print("  - Block-style: Same as FFT but avoids modulo with explicit blocks")
