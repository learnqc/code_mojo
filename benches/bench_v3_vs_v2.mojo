"""
Benchmark comparing execute_fused_v3 vs execute_simd_v2.
Tests various circuit sizes with different gate patterns.
"""

from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.gates import H, X, Z, RZ
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.core.execute_simd_v2_dispatch import (
    execute_transformations_simd_v2,
)
from time import perf_counter_ns
from math import pi


fn benchmark_v2[N: Int](circuit: QuantumCircuit, iters: Int) -> Float64:
    """Benchmark v2 executor."""
    var total_time: Float64 = 0.0

    for _ in range(iters):
        var state = circuit.state.copy()
        var start = perf_counter_ns()
        execute_transformations_simd_v2[N](state, circuit.transformations)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / iters


fn benchmark_v3[N: Int](circuit: QuantumCircuit, iters: Int) -> Float64:
    """Benchmark v3 executor."""
    var total_time: Float64 = 0.0

    for _ in range(iters):
        var state = circuit.state.copy()
        var start = perf_counter_ns()
        execute_fused_v3[N](state, circuit.transformations)
        var end = perf_counter_ns()
        total_time += Float64(end - start) / 1e9

    return total_time / iters


fn create_mixed_circuit(n: Int) -> QuantumCircuit:
    """Create a circuit with mixed local and global gates."""
    var circuit = QuantumCircuit(n)

    # Local gates (qubits 0-10)
    for i in range(min(n, 11)):
        circuit.h(i)
        circuit.x(i)
        circuit.z(i)

    # Global gates (qubits 11+)
    for i in range(11, n):
        circuit.h(i)
        circuit.rz(i, pi / 4)

    return circuit^


fn create_global_heavy_circuit(n: Int) -> QuantumCircuit:
    """Create a circuit with mostly global gates."""
    var circuit = QuantumCircuit(n)

    # Mostly global gates
    for i in range(n):
        circuit.h(i)
        circuit.rz(i, pi / 3)
        circuit.x(i)

    return circuit^


fn create_local_heavy_circuit(n: Int) -> QuantumCircuit:
    """Create a circuit with mostly local gates."""
    var circuit = QuantumCircuit(n)

    # Mostly local gates
    for i in range(min(n, 11)):
        for _ in range(5):
            circuit.h(i)
            circuit.x(i)
            circuit.z(i)

    # Few global gates
    for i in range(11, n):
        circuit.h(i)

    return circuit^


fn main():
    print("=" * 60)
    print("Radix-4 Fusion Executor (v3) vs SIMD v2 Benchmark")
    print("=" * 60)

    alias iters = 10

    # Test different qubit counts
    alias qubit_counts = List[Int](15, 20, 25)

    for q_idx in range(3):
        var n: Int
        if q_idx == 0:
            n = 15
        elif q_idx == 1:
            n = 20
        else:
            n = 25

        print("\n" + "=" * 60)
        print("N =", n, "qubits")
        print("=" * 60)

        # Mixed circuit
        print("\n--- Mixed Circuit (local + global gates) ---")
        var mixed = create_mixed_circuit(n)
        print("Gates:", len(mixed.transformations))

        var v2_time: Float64
        var v3_time: Float64

        if n == 15:
            v2_time = benchmark_v2[15](mixed, iters)
            v3_time = benchmark_v3[15](mixed, iters)
        elif n == 20:
            v2_time = benchmark_v2[20](mixed, iters)
            v3_time = benchmark_v3[20](mixed, iters)
        else:
            v2_time = benchmark_v2[25](mixed, iters)
            v3_time = benchmark_v3[25](mixed, iters)

        print("v2 time:", v2_time, "s")
        print("v3 time:", v3_time, "s")
        print("Speedup:", v2_time / v3_time, "x")

        # Global-heavy circuit
        print("\n--- Global-Heavy Circuit ---")
        var global_heavy = create_global_heavy_circuit(n)
        print("Gates:", len(global_heavy.transformations))

        if n == 15:
            v2_time = benchmark_v2[15](global_heavy, iters)
            v3_time = benchmark_v3[15](global_heavy, iters)
        elif n == 20:
            v2_time = benchmark_v2[20](global_heavy, iters)
            v3_time = benchmark_v3[20](global_heavy, iters)
        else:
            v2_time = benchmark_v2[25](global_heavy, iters)
            v3_time = benchmark_v3[25](global_heavy, iters)

        print("v2 time:", v2_time, "s")
        print("v3 time:", v3_time, "s")
        print("Speedup:", v2_time / v3_time, "x")

        # Local-heavy circuit
        print("\n--- Local-Heavy Circuit ---")
        var local_heavy = create_local_heavy_circuit(n)
        print("Gates:", len(local_heavy.transformations))

        if n == 15:
            v2_time = benchmark_v2[15](local_heavy, iters)
            v3_time = benchmark_v3[15](local_heavy, iters)
        elif n == 20:
            v2_time = benchmark_v2[20](local_heavy, iters)
            v3_time = benchmark_v3[20](local_heavy, iters)
        else:
            v2_time = benchmark_v2[25](local_heavy, iters)
            v3_time = benchmark_v3[25](local_heavy, iters)

        print("v2 time:", v2_time, "s")
        print("v3 time:", v3_time, "s")
        print("Speedup:", v2_time / v3_time, "x")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
