"""
Benchmark hybrid execution strategy vs Qiskit for value encoding.

Compares:
- Pure v2: entire circuit with SIMD v2
- Pure v3: entire circuit with fused v3  
- Hybrid: prep with v3, IQFT with v2
- Qiskit: reference implementation

Tests both swap=True and swap=False variants.
"""

from butterfly.utils.benchmark_runner import BenchmarkRunner
from butterfly.core.circuit import QuantumCircuit
from butterfly.algos.value_encoding_circuit import (
    encode_value_circuits_runtime,
    encode_value_circuit,
)
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.core.execution_strategy import SIMD_V2, FUSED_V3
from collections import Dict
from time import perf_counter_ns
from benchmark import keep
from python import Python


fn bench_n[
    n: Int
](mut runner: BenchmarkRunner, value: FloatType, swap: Bool, iters: Int) raises:
    """Benchmark for specific n with compile-time dispatch."""
    var params = Dict[String, String]()
    params["n"] = String(n)
    params["value"] = String(value)
    params["swap"] = "True" if swap else "False"

    print(
        "n="
        + String(n)
        + ", value="
        + String(value)
        + ", swap="
        + ("True" if swap else "False")
    )

    # Build circuit using canonical function
    var circuit = QuantumCircuit(n)
    encode_value_circuit[n](circuit, value, swap)

    # Benchmark pure v2
    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        var c = circuit.copy()
        var s = QuantumState(n)
        c.execute_with_strategy[n](s, SIMD_V2)
        keep(s.re.unsafe_ptr())
    var t1 = Int(perf_counter_ns())
    var time_v2 = Float64(t1 - t0) / 1_000_000.0 / iters
    runner.add_result(params, "pure_v2_ms", time_v2)

    # Benchmark pure v3
    t0 = Int(perf_counter_ns())
    for _ in range(iters):
        var c = circuit.copy()
        var s = QuantumState(n)
        c.execute_with_strategy[n](s, FUSED_V3)
        keep(s.re.unsafe_ptr())
    t1 = Int(perf_counter_ns())
    var time_v3 = Float64(t1 - t0) / 1_000_000.0 / iters
    runner.add_result(params, "pure_v3_ms", time_v3)

    # Benchmark hybrid v3+v2
    t0 = Int(perf_counter_ns())
    for _ in range(iters):
        var cs = encode_value_circuits_runtime(n, value, swap)
        var state = QuantumState(n)
        cs[0].execute_with_strategy[n](state, FUSED_V3)
        cs[1].execute_with_strategy[n](state, SIMD_V2)
        cs[2].execute_with_strategy[n](state, SIMD_V2)
        keep(state.re.unsafe_ptr())
    t1 = Int(perf_counter_ns())
    var time_hybrid = Float64(t1 - t0) / 1_000_000.0 / iters
    runner.add_result(params, "hybrid_ms", time_hybrid)

    # Benchmark Qiskit
    var py_benchmarks = Python.import_module("external_benchmarks")
    var qiskit_time_str = py_benchmarks.benchmark_qiskit_value_encoding_str(
        n, value, iters
    )
    var qiskit_time_ms = atof(String(qiskit_time_str))
    runner.add_result(params, "qiskit_ms", qiskit_time_ms)


fn main() raises:
    # Setup Python path
    var sys = Python.import_module("sys")
    sys.path.append(".")
    sys.path.append("benches")

    var runner = BenchmarkRunner("Value Encoding - Hybrid vs Qiskit (v=4.7)")

    var param_cols = List[String]()
    param_cols.append("n")
    param_cols.append("value")
    param_cols.append("swap")
    runner.set_param_columns(param_cols^)

    var bench_cols = List[String]()
    bench_cols.append("pure_v2_ms")
    bench_cols.append("pure_v3_ms")
    bench_cols.append("hybrid_ms")
    bench_cols.append("qiskit_ms")
    runner.set_bench_columns(bench_cols^)

    print("Benchmarking hybrid execution strategy vs Qiskit")
    print("=" * 80)
    print()

    alias value = 4.7

    # Test both swap variants for selected sizes
    bench_n[10](runner, value, True, 10)
    bench_n[10](runner, value, False, 10)

    bench_n[15](runner, value, True, 10)
    bench_n[15](runner, value, False, 10)

    bench_n[20](runner, value, True, 5)
    bench_n[20](runner, value, False, 5)

    bench_n[25](runner, value, True, 3)
    bench_n[25](runner, value, False, 3)

    bench_n[27](runner, value, True, 2)
    bench_n[27](runner, value, False, 2)

    runner.print_table(show_winner=True)

    var folder_name = "2025_12_25"
    var output_path = "benches/results/" + folder_name + "/hybrid_vs_qiskit"
    runner.save_csv(output_path)

    print("\nStrategies:")
    print("  pure_v2_ms  = Entire circuit with SIMD v2")
    print("  pure_v3_ms  = Entire circuit with fused v3")
    print("  hybrid_ms   = Prep with v3, IQFT with v2")
    print("  qiskit_ms   = Qiskit Aer simulator")
    print("\nResults saved with timestamp:", String(runner.timestamp))
