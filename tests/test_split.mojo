"""
Performance benchmark: Zero-copy vs Copy-based vs Regular execution.
"""

from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.two_row_state import transform_split_parallel
from butterfly.core.execute_split_runtime import execute_split_parallel_runtime
from butterfly.utils.visualization import print_state
from time import perf_counter_ns
from math import pi

from butterfly.algos.value_encoding_circuit import encode_value_circuit


fn benchmark_n[n: Int]() raises:
    """Benchmark at a specific n value."""
    alias row_size = 1 << (n - 1)
    alias iters = 1

    print("n=" + String(n) + " (row_size=" + String(row_size) + ")")
    print("-" * 60)

    # Create circuit that doesn't touch last qubit
    swap = False
    var circuit = QuantumCircuit(n)
    # for i in range(n - 1):
    #     circuit.h(i)
    #     circuit.p(i, pi / 4.0)

    # circuit.cp(0, 1, pi / 4.0)
    # for i in range(1, n - 1):
    #     circuit.cp(i, i - 1, pi / 4.0)

    from butterfly.algos.qft import iqft

    v = 4.7

    for j in range(n):
        circuit.h(j)
    for j in range(n):
        if swap:
            circuit.p(j, 2 * pi / 2 ** (n - j) * v)
        else:
            circuit.p(j, 2 * pi / 2 ** (j + 1) * v)

    var targets = [n - 1 - j for j in range(n)]
    if swap:
        targets = [j for j in range(n)]
    iqft(circuit=circuit, targets=targets, do_swap=swap)
    # for j in reversed(range(len(targets))):
    #     circuit.h(targets[j])
    #     for k in reversed(range(j)):
    #         # cp signature: (control, target, theta)
    #         circuit.cp(targets[j], targets[k], -pi / (2 ** (j - k)))

    # Benchmark 1: Fused V3 (Champion)
    var total_fused: Int = 0
    for _ in range(iters):
        var state = QuantumState(n)
        var t0 = perf_counter_ns()
        circuit.execute_fused_v3_dynamic(state)
        var t1 = perf_counter_ns()
        total_fused += Int(t1 - t0)
        print_state(state^)
    var time_fused = Float64(total_fused) / Float64(iters) / 1_000_000.0

    # Benchmark 2: Sequential split execution
    var total_sequential: Int = 0
    for _ in range(iters):
        var state = QuantumState(n)
        var t0 = perf_counter_ns()
        execute_split_parallel_runtime(state, circuit)
        var t1 = perf_counter_ns()
        total_sequential += Int(t1 - t0)
        print_state(state^)
    var time_sequential = (
        Float64(total_sequential) / Float64(iters) / 1_000_000.0
    )

    # Benchmark 3: Batched parallel split execution
    from butterfly.core.execute_split_parallel_batched import (
        execute_split_parallel_batched,
    )

    var total_batched: Int = 0
    for _ in range(iters):
        var state = QuantumState(n)
        var t0 = perf_counter_ns()
        execute_split_parallel_batched(state, circuit)
        var t1 = perf_counter_ns()
        total_batched += Int(t1 - t0)
        print_state(state^)
    var time_batched = Float64(total_batched) / Float64(iters) / 1_000_000.0

    # Benchmark 4: Zero-copy split execution (gate by gate) - COMMENTED OUT (crashes)
    # var total_zero_copy: Int = 0
    # for _ in range(iters):
    #     var state = QuantumState(n)
    #     var t0 = perf_counter_ns()
    #     # Apply each gate using zero-copy split
    #     for i in range(n - 1):
    #         from butterfly.core.gates import H, P
    #         transform_split_parallel[row_size](state, i, H)
    #         transform_split_parallel[row_size](state, i, P(pi / 4.0))
    #     var t1 = perf_counter_ns()
    #     total_zero_copy += Int(t1 - t0)
    # var time_zero_copy = Float64(total_zero_copy) / Float64(iters) / 1_000_000.0

    # Print results
    print("Fused V3 (Champion):   " + String(time_fused) + " ms")
    print("Sequential split:      " + String(time_sequential) + " ms")
    print("Batched parallel split:" + String(time_batched) + " ms")
    # print("Zero-copy split:       " + String(time_zero_copy) + " ms")
    print()

    var speedup_sequential = time_fused / time_sequential
    var speedup_batched = time_fused / time_batched
    # var speedup_zero = time_fused / time_zero_copy

    print("Sequential speedup:    " + String(speedup_sequential) + "x")
    print("Batched speedup:       " + String(speedup_batched) + "x")
    # print("Zero-copy speedup:     " + String(speedup_zero) + "x")

    print()


fn main() raises:
    print("=" * 60)
    print("Split-State Performance Benchmark")
    print("=" * 60)
    print()

    benchmark_n[3]()

    # benchmark_n[20]()
    # benchmark_n[22]()
    # benchmark_n[24]()
    # benchmark_n[25]()

    print("=" * 60)
