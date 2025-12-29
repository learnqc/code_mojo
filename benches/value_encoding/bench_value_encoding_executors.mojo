"""
Benchmark value encoding with compile-time execute_with_strategy[n, strategy].

Compares all Mojo strategies against Qiskit using:
- execute_with_strategy[n, strategy]() - Compile-time specialized dispatch
- Gold standard encode_value_circuit() for circuit building
- BenchmarkRunner for results
"""

from butterfly.utils.benchmark_runner import BenchmarkRunner
from butterfly.core.circuit import QuantumCircuit, execute_with_strategy
from butterfly.core.execution_strategy import (
    ExecutionStrategy,
    GENERIC,
    SIMD_STRATEGY,
    SIMD_V2,
    FUSED_V3,
)
from butterfly.algos.value_encoding import encode_value_circuit
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from collections import Dict
from time import perf_counter_ns
from python import Python
from benchmark import keep


fn build_value_encoding_circuit(n: Int, value: FloatType) -> QuantumCircuit:
    """Build value encoding circuit using gold standard implementation."""
    var circuit = QuantumCircuit(n)
    encode_value_circuit(circuit, n, value)
    return circuit^


fn main() raises:
    # Setup Python path for external_benchmarks
    var sys = Python.import_module("sys")
    sys.path.append(".")
    sys.path.append("benches")

    var runner = BenchmarkRunner("Value Encoding - All Strategies (v=4.7)")

    # Configure columns
    var param_cols = List[String]()
    param_cols.append("n")
    param_cols.append("value")
    runner.set_param_columns(param_cols^)

    var bench_cols = List[String]()
    bench_cols.append("execute")
    bench_cols.append("execute_simd")
    bench_cols.append("execute_simd_v2")
    bench_cols.append("execute_fused_v3")
    bench_cols.append("qiskit_ms")
    runner.set_bench_columns(bench_cols^)

    print("Benchmarking value encoding with different executors")
    print("=" * 80)
    print()

    alias value = 4.7

    for n in range(3, 28):  # 3 to 27
        print("n=" + String(n) + ", value=" + String(value))

        # Create parameter dict
        var params = Dict[String, String]()
        params["n"] = String(n)
        params["value"] = String(value)

        # Build circuit once
        var circuit_template = build_value_encoding_circuit(n, value)

        # Dynamic iteration count based on n
        var iters: Int
        if n <= 15:
            iters = 10
        elif n <= 20:
            iters = 5
        elif n <= 25:
            iters = 3
        else:
            iters = 2

        # Benchmark execute()
        var t0 = Int(perf_counter_ns())
        for _ in range(iters):
            var c = circuit_template.copy()
            var s = c.run()
            keep(s.re.unsafe_ptr())
        var t1 = Int(perf_counter_ns())
        var time_exec = Float64(t1 - t0) / 1_000_000.0 / iters
        runner.add_result(params, "execute", time_exec)

        # Benchmark execute_simd()
        t0 = Int(perf_counter_ns())
        for _ in range(iters):
            var c = circuit_template.copy()
            var s = c.run_simd_dynamic()
            keep(s.re.unsafe_ptr())
        t1 = Int(perf_counter_ns())
        var time_simd = Float64(t1 - t0) / 1_000_000.0 / iters
        runner.add_result(params, "execute_simd", time_simd)

        # Benchmark execute_simd_v2()
        t0 = Int(perf_counter_ns())
        for _ in range(iters):
            var c = circuit_template.copy()
            var s = c.run_simd_v2_dynamic()
            keep(s.re.unsafe_ptr())
        t1 = Int(perf_counter_ns())
        var time_simd_v2 = Float64(t1 - t0) / 1_000_000.0 / iters
        runner.add_result(params, "execute_simd_v2", time_simd_v2)

        # Benchmark execute_optimized()
        t0 = Int(perf_counter_ns())
        for _ in range(iters):
            var c = circuit_template.copy()
            var s = c.run_fused_v3_dynamic()
            keep(s.re.unsafe_ptr())
        t1 = Int(perf_counter_ns())
        var time_opt = Float64(t1 - t0) / 1_000_000.0 / iters
        runner.add_result(params, "execute_fused_v3", time_opt)

        # Benchmark Qiskit (using string conversion workaround)
        var py_benchmarks = Python.import_module(
            "butterfly.utils.external_benchmarks"
        )
        var qiskit_time_str = py_benchmarks.benchmark_qiskit_value_encoding_str(
            n, value, iters
        )
        var qiskit_time_ms = atof(String(qiskit_time_str))
        runner.add_result(params, "qiskit_ms", qiskit_time_ms)

    # Print results table with winner
    runner.print_table(show_winner=True)

    # Save to CSV with folder organization
    # Specify folder name for organization (e.g., date or run identifier)
    var folder_name = "2025_12_24"  # You can change this per run
    var output_path = (
        "benches/results/" + folder_name + "/value_encoding_executors"
    )
    runner.save_csv(output_path)

    print("\nExecutors:")
    print("  execute          = Generic executor")
    print("  execute_simd     = SIMD optimized")
    print("  execute_simd_v2  = SIMD v2 with dispatch")
    print("  execute_fused_v3 = With fusion optimization")
    print("  qiskit_ms        = Qiskit with transpilation")
    print("\nResults saved with timestamp:", String(runner.timestamp))

    # Save to CSV - automated by run_benchmark_suite.py
    runner.save_csv("value_encoding_executors")
