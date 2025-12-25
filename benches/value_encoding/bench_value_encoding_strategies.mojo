"""
Benchmark value encoding using strategy-based execution.

Demonstrates the new ExecutionStrategy variant system by looping
over all strategies instead of calling each method individually.
"""

from butterfly.utils.benchmark_runner import BenchmarkRunner
from butterfly.utils.benchmark_verify import verify_states_equal
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.execution_strategy import (
    ExecutionStrategy,
    Generic,
    SIMD,
    SIMDv2,
    FusedV3,
    GENERIC,
    SIMD_STRATEGY,
    SIMD_V2,
    FUSED_V3,
)
from butterfly.core.gates import H, P
from butterfly.core.types import FloatType
from math import pi
from collections import Dict, List
from time import perf_counter_ns
from benchmark import keep


fn build_value_encoding_circuit(n: Int, value: FloatType) -> QuantumCircuit:
    """Build value encoding circuit."""
    var circuit = QuantumCircuit(n)

    # Normalize value
    var max_val = FloatType(1 << n)
    var normalized = value
    while normalized < 0:
        normalized += max_val
    while normalized >= max_val:
        normalized -= max_val

    # Hadamard on all qubits
    for q in range(n):
        circuit.add(H, q)

    # Phase rotations
    for q in range(n):
        var angle = (
            2.0 * pi * normalized * FloatType(1 << (n - 1 - q)) / max_val
        )
        circuit.add(P(angle), q)

    # Inverse QFT
    for j_inv in range(n - 1, -1, -1):
        var q_target = n - 1 - j_inv
        circuit.add(H, q_target)

        if j_inv > 0:
            for m in range(j_inv):
                var q_control = n - 1 - m
                var theta = -pi / (1 << (j_inv - m))
                circuit.add_controlled(P(theta), q_target, q_control)

    return circuit^


fn get_strategy_name(strategy: ExecutionStrategy) -> String:
    """Get human-readable name for strategy."""
    if strategy.isa[Generic]():
        return "execute"
    elif strategy.isa[SIMD]():
        return "execute_simd"
    elif strategy.isa[SIMDv2]():
        return "execute_simd_v2"
    elif strategy.isa[FusedV3]():
        return "execute_fused_v3"
    else:
        return "unknown"


fn main() raises:
    var runner = BenchmarkRunner("Value Encoding - Strategy-Based")

    # Define all strategies to test
    var strategies = List[ExecutionStrategy]()
    strategies.append(GENERIC)
    strategies.append(SIMD_STRATEGY)
    strategies.append(SIMD_V2)
    strategies.append(FUSED_V3)

    # Configure columns
    var param_cols = List[String]()
    param_cols.append("n")
    param_cols.append("value")
    runner.set_param_columns(param_cols^)

    var bench_cols = List[String]()
    for i in range(len(strategies)):
        bench_cols.append(get_strategy_name(strategies[i]))
    runner.set_bench_columns(bench_cols^)

    print("Benchmarking value encoding with strategy-based execution")
    print("=" * 80)
    print()

    # Test cases
    var test_cases = List[Tuple[Int, FloatType]]()
    test_cases.append((10, 42.0))
    test_cases.append((12, 123.0))
    test_cases.append((15, 456.0))
    test_cases.append((18, 789.0))

    for i in range(len(test_cases)):
        var n = test_cases[i][0]
        var value = test_cases[i][1]

        print("n=" + String(n) + ", value=" + String(value))

        # Create parameter dict
        var params = Dict[String, String]()
        params["n"] = String(n)
        params["value"] = String(value)

        # Build circuit once
        var circuit_template = build_value_encoding_circuit(n, value)

        # VERIFICATION - ensure all strategies produce same result
        print("  Verifying strategies...", end="")
        var ref_circuit = circuit_template.copy()
        var reference_state = ref_circuit.run_with_strategy(GENERIC)

        for j in range(1, len(strategies)):
            var test_circuit = circuit_template.copy()
            var test_state = test_circuit.run_with_strategy(strategies[j])
            var strategy_name = get_strategy_name(strategies[j])
            _ = verify_states_equal(
                reference_state,
                test_state,
                name1="execute",
                name2=strategy_name,
            )
        print(" ✓")

        var iters = 5

        # Benchmark each strategy
        for j in range(len(strategies)):
            var strategy = strategies[j]
            var strategy_name = get_strategy_name(strategy)

            var t0 = Int(perf_counter_ns())
            for _ in range(iters):
                var c = circuit_template.copy()
                var s = c.run_with_strategy(strategy)
                keep(s.re.unsafe_ptr())
            var t1 = Int(perf_counter_ns())
            var time_ms = Float64(t1 - t0) / 1_000_000.0 / iters
            runner.add_result(params, strategy_name, time_ms)

    # Print results table with winner
    runner.print_table(show_winner=True)

    # Save to CSV with folder organization
    var folder_name = "2025_12_24"
    var output_path = (
        "benches/results/" + folder_name + "/value_encoding_strategies"
    )
    runner.save_csv(output_path)

    print("\nStrategies:")
    print("  execute          = Generic (debuggable)")
    print("  execute_simd     = SIMD optimized (runtime dispatch)")
    print("  execute_simd_v2  = SIMD v2 with dispatch")
    print("  execute_fused_v3 = Fusion optimization")
    print("\nResults saved with timestamp:", String(runner.timestamp))
