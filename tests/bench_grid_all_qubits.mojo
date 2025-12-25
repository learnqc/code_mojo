"""
Benchmark GridQuantumState with ALL qubits (row + column operations).
"""

from butterfly.core.grid_state import GridQuantumState
from butterfly.core.state import QuantumState, transform
from butterfly.core.gates import H, P
from math import pi
from time import perf_counter_ns


fn benchmark_all_qubits[n: Int, row_bits: Int]() raises:
    """Benchmark with operations on ALL qubits."""
    alias row_size = 1 << (n - row_bits)
    alias iters = 3

    print("n=" + String(n) + ", row_bits=" + String(row_bits))
    print("  Operations on ALL " + String(n) + " qubits")
    print("  - Qubits 0-" + String(n - row_bits - 1) + ": Row ops (NDBuffer)")
    print(
        "  - Qubits "
        + String(n - row_bits)
        + "-"
        + String(n - 1)
        + ": Column ops (List)"
    )

    # Benchmark regular
    var total_regular: Int = 0
    for _ in range(iters):
        var state = QuantumState(n)
        var t0 = perf_counter_ns()
        for i in range(n):
            transform(state, i, H)
            transform(state, i, P(pi / 4.0))
        var t1 = perf_counter_ns()
        total_regular += Int(t1 - t0)
    var time_regular = Float64(total_regular) / Float64(iters) / 1_000_000.0

    # Benchmark grid
    var total_grid: Int = 0
    for _ in range(iters):
        var state = GridQuantumState(n, row_bits)
        var t0 = perf_counter_ns()
        for i in range(n):
            state.transform[row_size](i, H)
            state.transform[row_size](i, P(pi / 4.0))
        var t1 = perf_counter_ns()
        total_grid += Int(t1 - t0)
    var time_grid = Float64(total_grid) / Float64(iters) / 1_000_000.0

    print("  Regular: " + String(time_regular) + " ms")
    print("  Grid:    " + String(time_grid) + " ms")

    var speedup = time_regular / time_grid
    if speedup > 1.0:
        print("  Speedup: " + String(speedup) + "x ✓")
    else:
        print("  Slowdown: " + String(1.0 / speedup) + "x ✗")
    print()


fn main() raises:
    print("=" * 60)
    print("GridQuantumState: ALL Qubits Benchmark")
    print("(Including both row and column operations)")
    print("=" * 60)
    print()

    # Test with 2 rows
    print("--- 2 Rows (row_bits=1) ---")
    benchmark_all_qubits[20, 1]()
    benchmark_all_qubits[22, 1]()
    benchmark_all_qubits[24, 1]()

    print("--- 4 Rows (row_bits=2) ---")
    benchmark_all_qubits[20, 2]()
    benchmark_all_qubits[22, 2]()
    benchmark_all_qubits[24, 2]()

    print("=" * 60)
