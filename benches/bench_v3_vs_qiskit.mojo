"""
Comprehensive benchmark comparing v3 executor vs Qiskit for value encoding.
Tests N=3 to N=29 with adaptive iteration counts.
Fair comparison: circuit/transformations built once, only execution is timed.
"""
from time import perf_counter_ns
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.algos.value_encoding import encode_value_circuit
from butterfly.core.execute_fused_v3 import execute_fused_v3


fn benchmark_v3[N: Int](value: Float64, iterations: Int) -> Float64:
    """Benchmark v3 executor for given problem size.

    Fair comparison: Build circuit once (warmup), then time only execution.
    """
    # Build circuit once (like Qiskit's transpile)
    var circuit = QuantumCircuit(N)
    encode_value_circuit(circuit, N, value)
    var transformations = circuit.transformations.copy()

    # Warmup
    var warmup_state = QuantumState(N)
    execute_fused_v3[1 << N](warmup_state, transformations, block_log=20)

    # Timed iterations - only execution, not circuit building
    var total_time: Float64 = 0.0
    for iter in range(iterations):
        var state = QuantumState(N)  # Fresh state each time

        var t0 = perf_counter_ns()
        execute_fused_v3[1 << N](state, transformations, block_log=20)
        var t1 = perf_counter_ns()

        total_time += Float64(t1 - t0)

    return total_time / Float64(iterations) / 1e9  # Return mean time in seconds


fn main():
    alias value = 0.5

    print("=" * 80)
    print("V3 Executor vs Qiskit - Value Encoding Benchmark")
    print("Fair comparison: transformations built once, only execution timed")
    print("=" * 80)
    print()
    print("N    | Iters | V3 Time (s)")
    print("-" * 80)

    # N=3 to N=15 (10 iterations)
    print("3    | 10    |", benchmark_v3[3](value, 10))
    print("4    | 10    |", benchmark_v3[4](value, 10))
    print("5    | 10    |", benchmark_v3[5](value, 10))
    print("6    | 10    |", benchmark_v3[6](value, 10))
    print("7    | 10    |", benchmark_v3[7](value, 10))
    print("8    | 10    |", benchmark_v3[8](value, 10))
    print("9    | 10    |", benchmark_v3[9](value, 10))
    print("10   | 10    |", benchmark_v3[10](value, 10))
    print("11   | 10    |", benchmark_v3[11](value, 10))
    print("12   | 10    |", benchmark_v3[12](value, 10))
    print("13   | 10    |", benchmark_v3[13](value, 10))
    print("14   | 10    |", benchmark_v3[14](value, 10))
    print("15   | 10    |", benchmark_v3[15](value, 10))

    # N=16 to N=20 (5 iterations)
    print("16   | 5     |", benchmark_v3[16](value, 5))
    print("17   | 5     |", benchmark_v3[17](value, 5))
    print("18   | 5     |", benchmark_v3[18](value, 5))
    print("19   | 5     |", benchmark_v3[19](value, 5))
    print("20   | 5     |", benchmark_v3[20](value, 5))

    # N=21 to N=25 (3 iterations)
    print("21   | 3     |", benchmark_v3[21](value, 3))
    print("22   | 3     |", benchmark_v3[22](value, 3))
    print("23   | 3     |", benchmark_v3[23](value, 3))
    print("24   | 3     |", benchmark_v3[24](value, 3))
    print("25   | 3     |", benchmark_v3[25](value, 3))

    # N=26 to N=29 (2 iterations)
    print("26   | 2     |", benchmark_v3[26](value, 2))
    print("27   | 2     |", benchmark_v3[27](value, 2))
    print("28   | 2     |", benchmark_v3[28](value, 2))
    print("29   | 2     |", benchmark_v3[29](value, 2))

    print("=" * 80)
    print()
    print(
        "Compare with Qiskit results in: benches/qiskit_results_all_gates.txt"
    )
