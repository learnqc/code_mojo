"""
Benchmark GridQuantumState vs all existing strategies.

Compares:
- Generic: Standard execute()
- SIMD v2: execute_simd_v2_dynamic()
- Grid 2-row: GridQuantumState with row_bits=1
- Grid 4-row: GridQuantumState with row_bits=2
- Grid 8-row: GridQuantumState with row_bits=3
"""
# pixi run mojo run -I . benches/value_encoding/bench_grid_vs_all.mojo

from butterfly.utils.benchmark_runner import BenchmarkRunner
from butterfly.core.grid_state import GridQuantumState
from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.types import FloatType
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from collections import Dict
from time import perf_counter_ns
from benchmark import keep


fn bench_n[
    n: Int
](mut runner: BenchmarkRunner, value: FloatType, iters: Int) raises:
    """Benchmark for specific n with compile-time dispatch."""
    var params = Dict[String, String]()
    params["n"] = String(n)
    params["value"] = String(value)

    runner.log_progress("n=" + String(n) + ", value=" + String(value))

    # Create circuit
    var circuits = encode_value_circuits_runtime(n, value)

    # Benchmark Generic
    runner.log_progress("  [1/5] Running Generic...")
    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        var s = QuantumState(n)
        circuits[0].execute(s)
        keep(s.re.unsafe_ptr())
    var t1 = Int(perf_counter_ns())
    var time_generic = Float64(t1 - t0) / 1_000_000.0 / iters
    runner.add_result(params, "generic_ms", time_generic)

    # Benchmark SIMD v2
    runner.log_progress("  [2/5] Running SIMD v2...")
    t0 = Int(perf_counter_ns())
    for _ in range(iters):
        var s = QuantumState(n)
        circuits[0].execute_simd_v2_dynamic(s)
        keep(s.re.unsafe_ptr())
    t1 = Int(perf_counter_ns())
    var time_v2 = Float64(t1 - t0) / 1_000_000.0 / iters
    runner.add_result(params, "simd_v2_ms", time_v2)

    # Benchmark Grid 2 rows
    runner.log_progress("  [3/5] Running Grid Quantum State (2 rows)...")
    alias row_size_2 = 1 << (n - 1)
    t0 = Int(perf_counter_ns())
    for _ in range(iters):
        var s = GridQuantumState(n, 1)
        s.execute[row_size_2](circuits[0])
        keep(s.re.unsafe_ptr())
    t1 = Int(perf_counter_ns())
    var time_grid2 = Float64(t1 - t0) / 1_000_000.0 / iters
    runner.add_result(params, "grid_2row_ms", time_grid2)

    # Benchmark Grid 4 rows
    runner.log_progress("  [4/5] Running Grid Quantum State (4 rows)...")
    alias row_size_4 = 1 << (n - 2)
    t0 = Int(perf_counter_ns())
    for _ in range(iters):
        var s = GridQuantumState(n, 2)
        s.execute[row_size_4](circuits[0])
        keep(s.re.unsafe_ptr())
    t1 = Int(perf_counter_ns())
    var time_grid4 = Float64(t1 - t0) / 1_000_000.0 / iters
    runner.add_result(params, "grid_4row_ms", time_grid4)

    # Benchmark Grid 8 rows
    runner.log_progress("  [5/5] Running Grid Quantum State (8 rows)...")
    alias row_size_8 = 1 << (n - 3)
    t0 = Int(perf_counter_ns())
    for _ in range(iters):
        var s = GridQuantumState(n, 3)
        s.execute[row_size_8](circuits[0])
        keep(s.re.unsafe_ptr())
    t1 = Int(perf_counter_ns())
    var time_grid8 = Float64(t1 - t0) / 1_000_000.0 / iters
    runner.add_result(params, "grid_8row_ms", time_grid8)


fn main() raises:
    var runner = BenchmarkRunner("Grid vs All Strategies (n=3-27)")

    var param_cols = List[String]()
    param_cols.append("n")
    param_cols.append("value")
    runner.set_param_columns(param_cols^)

    var bench_cols = List[String]()
    bench_cols.append("generic_ms")
    bench_cols.append("simd_v2_ms")
    bench_cols.append("grid_2row_ms")
    bench_cols.append("grid_4row_ms")
    bench_cols.append("grid_8row_ms")
    runner.set_bench_columns(bench_cols^)

    print("Benchmarking GridQuantumState: n=3-27, 2/4/8 rows")
    print("=" * 80)
    print()

    alias value = 4.7

    # Comprehensive benchmark n=3 to 27
    bench_n[3](runner, value, 20)
    bench_n[5](runner, value, 20)
    bench_n[7](runner, value, 20)
    bench_n[10](runner, value, 10)
    bench_n[12](runner, value, 10)
    bench_n[15](runner, value, 10)
    bench_n[17](runner, value, 5)
    bench_n[20](runner, value, 5)
    bench_n[22](runner, value, 3)
    bench_n[24](runner, value, 3)
    bench_n[25](runner, value, 3)
    bench_n[27](runner, value, 2)

    runner.print_table(show_winner=True)

    # Save results - automated by run_benchmark_suite.py
    runner.save_csv("grid_comprehensive")

    print("\nStrategies:")
    print("  generic_ms    = Standard execute()")
    print("  simd_v2_ms    = SIMD v2 dynamic")
    print("  grid_2row_ms  = GridQuantumState with 2 rows")
    print("  grid_4row_ms  = GridQuantumState with 4 rows")
    print("  grid_8row_ms  = GridQuantumState with 8 rows")
    print("\nResults saved with timestamp:", String(runner.timestamp))
