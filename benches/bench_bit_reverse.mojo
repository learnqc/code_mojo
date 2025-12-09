from butterfly.core.state import QuantumState, generate_state, bit_reverse_state
from butterfly.core.fft_simd import bit_reverse_index
from butterfly.core.types import *
from time import perf_counter_ns
from math import log2
from bit.bit import bit_reverse


fn bit_reverse_state_original(mut state: QuantumState):
    """Original implementation using bit.bit.bit_reverse."""
    n = Int(log2(Float64(state.size())))
    s_re = List[FloatType](length=1 << n, fill=0.0)
    s_im = List[FloatType](length=1 << n, fill=0.0)

    for i in range(1 << n):
        idx = Int(bit_reverse(SIMD[DType.uint64, 1](i))[0] >> (64 - n))
        s_re[i] = state.re[idx]
        s_im[i] = state.im[idx]

    state.re = s_re^
    state.im = s_im^


fn bit_reverse_state_loop(mut state: QuantumState):
    """New implementation using bit_reverse_index loop."""
    n = Int(log2(Float64(state.size())))
    s_re = List[FloatType](length=1 << n, fill=0.0)
    s_im = List[FloatType](length=1 << n, fill=0.0)

    for i in range(1 << n):
        idx = bit_reverse_index(i, n)
        s_re[i] = state.re[idx]
        s_im[i] = state.im[idx]

    state.re = s_re^
    state.im = s_im^


fn benchmark_original[N: Int](num_iterations: Int) -> Float64:
    """Benchmark original bit_reverse implementation."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        bit_reverse_state_original(state)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn benchmark_loop[N: Int](num_iterations: Int) -> Float64:
    """Benchmark loop-based bit_reverse_index implementation."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        bit_reverse_state_loop(state)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn print_comparison(name: String, original_time: Float64, loop_time: Float64):
    """Print benchmark results with speedup calculation."""
    var speedup = original_time / loop_time
    print(name + ":")
    print("  Original (bit_reverse):     ", original_time * 1000, " ms")
    print("  Loop (bit_reverse_index):   ", loop_time * 1000, " ms")
    if speedup > 1.0:
        print("  Speedup:                    ", speedup, "x")
        print("  Loop is FASTER by:          ", (speedup - 1.0) * 100, "%")
    else:
        print("  Slowdown:                   ", 1.0 / speedup, "x")
        print("  Loop is SLOWER by:          ", (1.0 - speedup) * 100, "%")
    print()


fn main():
    print("=" * 70)
    print("Bit-Reversal Performance Benchmark")
    print("Comparing bit_reverse (SIMD) vs bit_reverse_index (loop)")
    print("=" * 70)
    print()

    alias num_iterations = 100

    # Benchmark N=16
    print("Benchmarking N=16 (4 qubits) - ", num_iterations, " iterations")
    print("-" * 70)
    alias N16 = 16
    var original_16 = benchmark_original[N16](num_iterations)
    var loop_16 = benchmark_loop[N16](num_iterations)
    print_comparison("N=16", original_16, loop_16)

    # Benchmark N=64
    print("Benchmarking N=64 (6 qubits) - ", num_iterations, " iterations")
    print("-" * 70)
    alias N64 = 64
    var original_64 = benchmark_original[N64](num_iterations)
    var loop_64 = benchmark_loop[N64](num_iterations)
    print_comparison("N=64", original_64, loop_64)

    # Benchmark N=256
    print("Benchmarking N=256 (8 qubits) - ", num_iterations, " iterations")
    print("-" * 70)
    alias N256 = 256
    var original_256 = benchmark_original[N256](num_iterations)
    var loop_256 = benchmark_loop[N256](num_iterations)
    print_comparison("N=256", original_256, loop_256)

    # Benchmark N=1024
    print("Benchmarking N=1024 (10 qubits) - ", num_iterations, " iterations")
    print("-" * 70)
    alias N1024 = 1024
    var original_1024 = benchmark_original[N1024](num_iterations)
    var loop_1024 = benchmark_loop[N1024](num_iterations)
    print_comparison("N=1024", original_1024, loop_1024)

    # Benchmark N=4096
    print("Benchmarking N=4096 (12 qubits) - ", num_iterations, " iterations")
    print("-" * 70)
    alias N4096 = 4096
    var original_4096 = benchmark_original[N4096](num_iterations)
    var loop_4096 = benchmark_loop[N4096](num_iterations)
    print_comparison("N=4096", original_4096, loop_4096)

    print("=" * 70)
    print("Benchmark completed!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  - Original uses SIMD bit_reverse instruction (hardware-optimized)")
    print("  - Loop uses manual bit-reversal loop")
    print("  - Original should be faster for most sizes due to hardware support")
