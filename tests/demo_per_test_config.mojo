"""
Demo showing how to programmatically set configs for each test case.

This creates config files on-the-fly with specific worker counts.
"""

from butterfly.utils.config import create_test_config, get_workers
from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.execute_as_grid import execute_as_grid
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from time import perf_counter_ns


fn benchmark_with_workers(n: Int, workers: Int) raises -> Float64:
    """Benchmark execute_as_grid with specific worker count."""

    # Create config with this worker count
    create_test_config(
        v_grid_row_workers=workers, v_grid_column_workers=workers
    )

    # Verify config loaded
    var actual_workers = get_workers("v_grid_rows")
    if actual_workers != workers:
        print("WARNING: Expected", workers, "workers but got", actual_workers)

    # Create circuit
    var circuits = encode_value_circuits_runtime(n, 3.14, swap=False)
    var circuit = circuits[0].copy()

    # Benchmark
    var state = QuantumState(n)
    var col_bits = n - 3

    var t0 = perf_counter_ns()
    execute_as_grid(state, circuit, col_bits)
    var t1 = perf_counter_ns()

    return Float64(t1 - t0) / 1_000_000.0


fn main() raises:
    print("Per-Test-Case Config Demo")
    print("=" * 60)
    print("Testing execute_as_grid with different worker counts\n")

    alias n = 20
    var worker_counts = List[Int](1, 2, 4, 8, 0)  # 0 = unlimited

    print("n =", n, "qubits\n")

    for i in range(len(worker_counts)):
        var workers = worker_counts[i]
        var label = String(workers) if workers > 0 else "unlimited"

        print("Testing with", label, "workers...")
        var time_ms = benchmark_with_workers(n, workers)
        print("  Time:", round(time_ms, 2), "ms\n")

    print("=" * 60)
    print("✓ All worker counts tested!")
    print("\nUsage in tests:")
    print("  create_test_config(v_grid_row_workers=4, v_grid_column_workers=4)")
    print("  # Now execute_as_grid uses 4 workers")
