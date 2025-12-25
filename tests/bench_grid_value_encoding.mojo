"""
Benchmark GridQuantumState with the SAME circuit (value encoding).
"""

from butterfly.core.grid_state import GridQuantumState
from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from time import perf_counter_ns


fn benchmark_same_circuit[n: Int, row_bits: Int]() raises:
    """Benchmark using the same circuit for both regular and grid."""
    alias row_size = 1 << (n - row_bits)
    alias iters = 3

    print("n=" + String(n) + ", row_bits=" + String(row_bits))

    # Create value encoding circuit
    var circuits = encode_value_circuits_runtime(n, 42)

    print(
        "  Circuit: "
        + String(len(circuits[0].transformations))
        + " transformations"
    )

    # Benchmark regular execution
    var total_regular: Int = 0
    for _ in range(iters):
        var state = QuantumState(n)
        var t0 = perf_counter_ns()
        circuits[0].execute(state)
        var t1 = perf_counter_ns()
        total_regular += Int(t1 - t0)
    var time_regular = Float64(total_regular) / Float64(iters) / 1_000_000.0

    # Benchmark grid execution
    var total_grid: Int = 0
    for _ in range(iters):
        var state = GridQuantumState(n, row_bits)
        var t0 = perf_counter_ns()
        state.execute[row_size](circuits[0])
        var t1 = perf_counter_ns()
        total_grid += Int(t1 - t0)
    var time_grid = Float64(total_grid) / Float64(iters) / 1_000_000.0

    print("  Regular: " + String(time_regular) + " ms")
    print("  Grid:    " + String(time_grid) + " ms")

    var speedup = time_regular / time_grid

    print("  Speedup: " + String(speedup) + "x")
    print()


fn main() raises:
    print("=" * 60)
    print("GridQuantumState: Value Encoding Benchmark")
    print("(Same circuit for both regular and grid)")
    print("=" * 60)
    print()

    # Test with 1 row (baseline - should match regular)
    print("--- 1 Row (row_bits=0) - Baseline ---")
    benchmark_same_circuit[20, 0]()
    benchmark_same_circuit[22, 0]()
    benchmark_same_circuit[24, 0]()

    # Test with 2 rows
    print("--- 2 Rows (row_bits=1) ---")
    benchmark_same_circuit[20, 1]()
    benchmark_same_circuit[22, 1]()
    benchmark_same_circuit[24, 1]()

    # Test with 4 rows
    print("--- 4 Rows (row_bits=2) ---")
    benchmark_same_circuit[20, 2]()
    benchmark_same_circuit[22, 2]()
    benchmark_same_circuit[24, 2]()

    print("=" * 60)
