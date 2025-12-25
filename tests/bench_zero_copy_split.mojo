"""
Performance benchmark: Zero-copy vs Copy-based vs Regular execution.
"""

from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.two_row_state import transform_split_parallel
from butterfly.core.execute_split_runtime import execute_split_parallel_runtime
from time import perf_counter_ns
from math import pi


fn benchmark_n[n: Int]() raises:
    """Benchmark at a specific n value."""
    alias row_size = 1 << (n - 1)
    alias iters = 3

    print("n=" + String(n) + " (row_size=" + String(row_size) + ")")
    print("-" * 60)

    # Create circuit that doesn't touch last qubit
    var circuit = QuantumCircuit(n)
    for i in range(n - 1):
        circuit.h(i)
        circuit.p(i, pi / 4.0)

    # Benchmark 1: Regular execution
    var total_regular: Int = 0
    for _ in range(iters):
        var state = QuantumState(n)
        var t0 = perf_counter_ns()
        circuit.execute(state)
        var t1 = perf_counter_ns()
        total_regular += t1 - t0
    var time_regular = Float64(total_regular) / Float64(iters) / 1_000_000.0

    # Benchmark 2: Copy-based split execution
    var total_copy: Int = 0
    for _ in range(iters):
        var state = QuantumState(n)
        var t0 = perf_counter_ns()
        execute_split_parallel_runtime(state, circuit)
        var t1 = perf_counter_ns()
        total_copy += t1 - t0
    var time_copy = Float64(total_copy) / Float64(iters) / 1_000_000.0

    # Benchmark 3: Zero-copy split execution (gate by gate)
    var total_zero_copy: Int = 0
    for _ in range(iters):
        var state = QuantumState(n)
        var t0 = perf_counter_ns()
        # Apply each gate using zero-copy split
        for i in range(n - 1):
            from butterfly.core.gates import H, P

            transform_split_parallel[row_size](state, i, H)
            transform_split_parallel[row_size](state, i, P(pi / 4.0))
        var t1 = perf_counter_ns()
        total_zero_copy += t1 - t0
    var time_zero_copy = Float64(total_zero_copy) / Float64(iters) / 1_000_000.0

    # Print results
    print("Regular execution:     " + String(time_regular) + " ms")
    print("Copy-based split:      " + String(time_copy) + " ms")
    print("Zero-copy split:       " + String(time_zero_copy) + " ms")
    print()

    var speedup_copy = time_regular / time_copy
    var speedup_zero = time_regular / time_zero_copy

    print("Copy-based speedup:    " + String(speedup_copy) + "x")
    print("Zero-copy speedup:     " + String(speedup_zero) + "x")
    print()


fn main() raises:
    print("=" * 60)
    print("Split-State Performance Benchmark")
    print("=" * 60)
    print()

    benchmark_n[20]()
    benchmark_n[22]()
    benchmark_n[24]()
    benchmark_n[25]()

    print("=" * 60)
