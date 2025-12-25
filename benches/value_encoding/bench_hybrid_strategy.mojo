"""
Benchmark hybrid execution strategy for value encoding.

Compares:
- Pure v2: entire circuit with SIMD v2
- Pure v3: entire circuit with fused v3  
- Hybrid: prep with v3, IQFT with v2

Uses compile-time dispatch with execute_with_strategy[n].
"""

from butterfly.utils.benchmark_runner import BenchmarkRunner
from butterfly.core.circuit import QuantumCircuit, QuantumRegister
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.core.execution_strategy import SIMD_V2, FUSED_V3
from collections import Dict
from time import perf_counter_ns
from math import pi
from benchmark import keep


fn build_circuit(n: Int, value: FloatType, swap: Bool) -> QuantumCircuit:
    """Build value encoding circuit."""
    var circuit = QuantumCircuit(n)
    for j in range(n):
        circuit.h(j)
    for j in range(n):
        if swap:
            circuit.p(j, 2 * pi / 2 ** (n - j) * value)
        else:
            circuit.p(j, 2 * pi / 2 ** (j + 1) * value)
    targets = [n - 1 - j for j in range(n)]
    if swap:
        targets = [j for j in range(n)]
    circuit.iqft(QuantumRegister("", targets[0], len(targets)), swap=swap)
    return circuit^


fn bench_n[
    n: Int
](mut runner: BenchmarkRunner, value: FloatType, iters: Int) raises:
    """Benchmark for specific n with compile-time dispatch."""
    var params = Dict[String, String]()
    params["n"] = String(n)
    params["value"] = String(value)

    print("n=" + String(n) + ", value=" + String(value))

    # Build circuits
    var circuit = build_circuit(n, value, swap=True)
    var circuits = encode_value_circuits_runtime(n, value, swap=True)

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
        var cs = encode_value_circuits_runtime(n, value, swap=True)
        var state = QuantumState(n)
        cs[0].execute_with_strategy[n](state, FUSED_V3)
        cs[1].execute_with_strategy[n](state, SIMD_V2)
        cs[2].execute_with_strategy[n](state, SIMD_V2)
        keep(state.re.unsafe_ptr())
    t1 = Int(perf_counter_ns())
    var time_hybrid = Float64(t1 - t0) / 1_000_000.0 / iters
    runner.add_result(params, "hybrid_ms", time_hybrid)


fn main() raises:
    var runner = BenchmarkRunner(
        "Value Encoding - Hybrid Strategy (v=4.7, swap=True)"
    )

    var param_cols = List[String]()
    param_cols.append("n")
    param_cols.append("value")
    runner.set_param_columns(param_cols^)

    var bench_cols = List[String]()
    bench_cols.append("pure_v2_ms")
    bench_cols.append("pure_v3_ms")
    bench_cols.append("hybrid_ms")
    runner.set_bench_columns(bench_cols^)

    print("Benchmarking hybrid execution strategy (swap=True only)")
    print("=" * 80)
    print()

    alias value = 4.7

    # Compile-time dispatch for each n
    bench_n[10](runner, value, 10)
    bench_n[11](runner, value, 10)
    bench_n[12](runner, value, 10)
    bench_n[13](runner, value, 10)
    bench_n[14](runner, value, 10)
    bench_n[15](runner, value, 10)
    bench_n[16](runner, value, 5)
    bench_n[17](runner, value, 5)
    bench_n[18](runner, value, 5)
    bench_n[19](runner, value, 5)
    bench_n[20](runner, value, 5)
    bench_n[21](runner, value, 3)
    bench_n[22](runner, value, 3)
    bench_n[23](runner, value, 3)
    bench_n[24](runner, value, 3)
    bench_n[25](runner, value, 3)
    bench_n[26](runner, value, 2)
    bench_n[27](runner, value, 2)

    runner.print_table(show_winner=True)

    var folder_name = "2025_12_25"
    var output_path = "benches/results/" + folder_name + "/hybrid_strategy"
    runner.save_csv(output_path)

    print("\nStrategies:")
    print("  pure_v2_ms  = Entire circuit with SIMD v2")
    print("  pure_v3_ms  = Entire circuit with fused v3")
    print("  hybrid_ms   = Prep with v3, IQFT with v2")
    print("\nResults saved with timestamp:", String(runner.timestamp))
