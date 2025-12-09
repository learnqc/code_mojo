from butterfly.core.state import QuantumState, generate_state
from butterfly.core.fft import fft, ifft
from butterfly.core.fft_simd import fft_simd, ifft_simd, fft_simd_parallel
from butterfly.core.types import *
from time import perf_counter_ns
from math import log2


fn benchmark_fft[N: Int](num_iterations: Int) -> Float64:
    """Benchmark standard FFT."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        fft(state)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9  # Convert to seconds

    return total_time / Float64(num_iterations)


fn benchmark_fft_simd[N: Int](num_iterations: Int) -> Float64:
    """Benchmark SIMD FFT."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        fft_simd[N](state)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn benchmark_fft_simd_parallel[N: Int](num_iterations: Int) -> Float64:
    """Benchmark parallel SIMD FFT."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        fft_simd_parallel[N](state)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn benchmark_ifft[N: Int](num_iterations: Int) -> Float64:
    """Benchmark standard IFFT."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        ifft(state)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn benchmark_ifft_simd[N: Int](num_iterations: Int) -> Float64:
    """Benchmark SIMD IFFT."""
    var total_time: Float64 = 0.0

    for i in range(num_iterations):
        var state = generate_state(Int(log2(Float64(N))), seed=i)
        var start = perf_counter_ns()
        ifft_simd[N](state)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / Float64(num_iterations)


fn print_speedup(name: String, standard_time: Float64, optimized_time: Float64):
    """Print benchmark results with speedup calculation."""
    var speedup = standard_time / optimized_time
    print(name + ":")
    print("  Standard:  ", standard_time * 1000, " ms")
    print("  Optimized: ", optimized_time * 1000, " ms")
    print("  Speedup:   ", speedup, "x")
    if speedup > 1.0:
        print("  Improvement: ", (speedup - 1.0) * 100, "%")
    else:
        print("  Slowdown:    ", (1.0 - speedup) * 100, "%")
    print()


fn main():
    print("=" * 60)
    print("FFT Performance Benchmark: Standard vs SIMD vs Parallel")
    print("=" * 60)
    print()

    alias num_iterations = 100

    # Benchmark N=16
    print("Benchmarking N=16 (4 qubits) - ", num_iterations, " iterations")
    print("-" * 60)
    alias N16 = 16
    var fft_16 = benchmark_fft[N16](num_iterations)
    var fft_simd_16 = benchmark_fft_simd[N16](num_iterations)
    var fft_parallel_16 = benchmark_fft_simd_parallel[N16](num_iterations)

    print_speedup("FFT N=16 (SIMD)", fft_16, fft_simd_16)
    print_speedup("FFT N=16 (Parallel)", fft_16, fft_parallel_16)

    # Benchmark N=64
    print("Benchmarking N=64 (6 qubits) - ", num_iterations, " iterations")
    print("-" * 60)
    alias N64 = 64
    var fft_64 = benchmark_fft[N64](num_iterations)
    var fft_simd_64 = benchmark_fft_simd[N64](num_iterations)
    var fft_parallel_64 = benchmark_fft_simd_parallel[N64](num_iterations)

    print_speedup("FFT N=64 (SIMD)", fft_64, fft_simd_64)
    print_speedup("FFT N=64 (Parallel)", fft_64, fft_parallel_64)

    # Benchmark N=256
    print("Benchmarking N=256 (8 qubits) - ", num_iterations, " iterations")
    print("-" * 60)
    alias N256 = 256
    var fft_256 = benchmark_fft[N256](num_iterations)
    var fft_simd_256 = benchmark_fft_simd[N256](num_iterations)
    var fft_parallel_256 = benchmark_fft_simd_parallel[N256](num_iterations)

    print_speedup("FFT N=256 (SIMD)", fft_256, fft_simd_256)
    print_speedup("FFT N=256 (Parallel)", fft_256, fft_parallel_256)

    # Benchmark N=1024
    print("Benchmarking N=1024 (10 qubits) - ", num_iterations, " iterations")
    print("-" * 60)
    alias N1024 = 1024
    var fft_1024 = benchmark_fft[N1024](num_iterations)
    var fft_simd_1024 = benchmark_fft_simd[N1024](num_iterations)
    var fft_parallel_1024 = benchmark_fft_simd_parallel[N1024](num_iterations)

    print_speedup("FFT N=1024 (SIMD)", fft_1024, fft_simd_1024)
    print_speedup("FFT N=1024 (Parallel)", fft_1024, fft_parallel_1024)

    # IFFT Benchmarks
    print()
    print("=" * 60)
    print("IFFT Performance Benchmark")
    print("=" * 60)
    print()

    print("Benchmarking IFFT N=256 - ", num_iterations, " iterations")
    print("-" * 60)
    var ifft_256 = benchmark_ifft[N256](num_iterations)
    var ifft_simd_256 = benchmark_ifft_simd[N256](num_iterations)

    print_speedup("IFFT N=256 (SIMD)", ifft_256, ifft_simd_256)

    print("Benchmarking IFFT N=1024 - ", num_iterations, " iterations")
    print("-" * 60)
    var ifft_1024 = benchmark_ifft[N1024](num_iterations)
    var ifft_simd_1024 = benchmark_ifft_simd[N1024](num_iterations)

    print_speedup("IFFT N=1024 (SIMD)", ifft_1024, ifft_simd_1024)

    print("=" * 60)
    print("Benchmark completed!")
    print("=" * 60)
