"""
Comprehensive benchmark: All strategies vs GridQuantumState.
Compares Generic, SIMD v2, Fused v3 against Grid (2 and 4 rows).
"""

from butterfly.core.grid_state import GridQuantumState
from butterfly.core.state import QuantumState
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from time import perf_counter_ns


fn benchmark_all_strategies[n: Int]() raises:
    """Compare all execution strategies at compile-time n."""
    alias iters = 3

    print("n=" + String(n))
    print("-" * 60)

    # Create circuit
    var circuits = encode_value_circuits_runtime(n, 42)
    print("Circuit: " + String(len(circuits[0].transformations)) + " gates")
    print()

    # Benchmark Generic (execute)
    var total_generic: Int = 0
    for _ in range(iters):
        var state = QuantumState(n)
        var t0 = perf_counter_ns()
        circuits[0].execute(state)
        var t1 = perf_counter_ns()
        total_generic += Int(t1 - t0)
    var time_generic = Float64(total_generic) / Float64(iters) / 1_000_000.0

    # Benchmark SIMD v2
    var total_v2: Int = 0
    for _ in range(iters):
        var state = QuantumState(n)
        var t0 = perf_counter_ns()
        circuits[0].execute_simd_v2_dynamic(state)
        var t1 = perf_counter_ns()
        total_v2 += Int(t1 - t0)
    var time_v2 = Float64(total_v2) / Float64(iters) / 1_000_000.0

    # Benchmark Fused v3
    var total_v3: Int = 0
    for _ in range(iters):
        var state = QuantumState(n)
        var t0 = perf_counter_ns()
        circuits[0].execute_fused_v3[n](state)
        var t1 = perf_counter_ns()
        total_v3 += Int(t1 - t0)
    var time_v3 = Float64(total_v3) / Float64(iters) / 1_000_000.0

    # Benchmark Grid 2 rows
    alias row_size_2 = 1 << (n - 1)
    var total_grid2: Int = 0
    for _ in range(iters):
        var state = GridQuantumState(n, 1)
        var t0 = perf_counter_ns()
        state.execute[row_size_2](circuits[0])
        var t1 = perf_counter_ns()
        total_grid2 += Int(t1 - t0)
    var time_grid2 = Float64(total_grid2) / Float64(iters) / 1_000_000.0

    # Benchmark Grid 4 rows
    alias row_size_4 = 1 << (n - 2)
    var total_grid4: Int = 0
    for _ in range(iters):
        var state = GridQuantumState(n, 2)
        var t0 = perf_counter_ns()
        state.execute[row_size_4](circuits[0])
        var t1 = perf_counter_ns()
        total_grid4 += Int(t1 - t0)
    var time_grid4 = Float64(total_grid4) / Float64(iters) / 1_000_000.0

    # Print results
    print("Generic:    " + String(time_generic) + " ms")
    print("SIMD v2:    " + String(time_v2) + " ms")
    print("Fused v3:   " + String(time_v3) + " ms")
    print("Grid 2-row: " + String(time_grid2) + " ms")
    print("Grid 4-row: " + String(time_grid4) + " ms")
    print()

    # Find best
    var best_time = time_generic
    var best_name = "Generic"

    if time_v2 < best_time:
        best_time = time_v2
        best_name = "SIMD v2"
    if time_v3 < best_time:
        best_time = time_v3
        best_name = "Fused v3"
    if time_grid2 < best_time:
        best_time = time_grid2
        best_name = "Grid 2-row"
    if time_grid4 < best_time:
        best_time = time_grid4
        best_name = "Grid 4-row"

    print("WINNER: " + best_name + " (" + String(best_time) + " ms)")
    print()

    # Speedups vs v3 (current best regular strategy)
    print("Speedups vs Fused v3:")
    print("  Grid 2-row: " + String(time_v3 / time_grid2) + "x")
    print("  Grid 4-row: " + String(time_v3 / time_grid4) + "x")
    print()


fn main() raises:
    print("=" * 60)
    print("Complete Strategy Comparison")
    print("=" * 60)
    print()

    benchmark_all_strategies[20]()
    benchmark_all_strategies[22]()
    benchmark_all_strategies[24]()

    print("=" * 60)
