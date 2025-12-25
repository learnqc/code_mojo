"""
Benchmark comparing execute_fused_v3 vs execute_simd_v2 using BenchmarkRunner.

Tests various circuit sizes with different gate patterns:
- Mixed circuits (local + global gates)
- Global-heavy circuits
- Local-heavy circuits

Includes correctness verification before benchmarking.
"""

from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.gates import H, X, Z, RZ
from butterfly.core.circuit import execute_simd_v2
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.utils.benchmark_runner import BenchmarkRunner
from time import perf_counter_ns
from math import pi
from collections import Dict
from benchmark import keep


fn create_mixed_circuit(n: Int) -> QuantumCircuit:
    """Create a circuit with mixed local and global gates."""
    var circuit = QuantumCircuit(n)

    # Local gates (qubits 0-10)
    for i in range(min(n, 11)):
        circuit.h(i)
        circuit.x(i)
        circuit.z(i)

    # Global gates (qubits 11+)
    for i in range(11, n):
        circuit.h(i)
        circuit.rz(i, pi / 4)

    return circuit^


fn create_global_heavy_circuit(n: Int) -> QuantumCircuit:
    """Create a circuit with mostly global gates."""
    var circuit = QuantumCircuit(n)

    # Mostly global gates
    for i in range(n):
        circuit.h(i)
        circuit.rz(i, pi / 3)
        circuit.x(i)

    return circuit^


fn create_local_heavy_circuit(n: Int) -> QuantumCircuit:
    """Create a circuit with mostly local gates."""
    var circuit = QuantumCircuit(n)

    # Mostly local gates
    for i in range(min(n, 11)):
        for _ in range(5):
            circuit.h(i)
            circuit.x(i)
            circuit.z(i)

    # Few global gates
    for i in range(11, n):
        circuit.h(i)

    return circuit^


fn verify_n15() -> Bool:
    """Verify correctness for n=15."""
    print("  Verifying n=15...")
    var circuit = create_mixed_circuit(15)
    var state_v2 = QuantumState(15)
    var state_v3 = QuantumState(15)

    execute_simd_v2[15](state_v2, circuit)
    execute_fused_v3[32768](state_v3, circuit)

    var max_diff = 0.0
    for i in range(32768):
        var diff = abs(state_v2[i].re - state_v3[i].re) + abs(
            state_v2[i].im - state_v3[i].im
        )
        if diff > max_diff:
            max_diff = diff

    if max_diff > 1e-10:
        print("    ✗ FAILED: max_diff=" + String(max_diff))
        return False
    print("    ✓ PASSED")
    return True


fn verify_n20() -> Bool:
    """Verify correctness for n=20."""
    print("  Verifying n=20...")
    var circuit = create_mixed_circuit(20)
    var state_v2 = QuantumState(20)
    var state_v3 = QuantumState(20)

    execute_simd_v2[20](state_v2, circuit)
    execute_fused_v3[1048576](state_v3, circuit)

    var max_diff = 0.0
    for i in range(1048576):
        var diff = abs(state_v2[i].re - state_v3[i].re) + abs(
            state_v2[i].im - state_v3[i].im
        )
        if diff > max_diff:
            max_diff = diff

    if max_diff > 1e-10:
        print("    ✗ FAILED: max_diff=" + String(max_diff))
        return False
    print("    ✓ PASSED")
    return True


fn run_verification() -> Bool:
    """Run correctness verification."""
    print("\n" + "=" * 80)
    print("CORRECTNESS VERIFICATION")
    print("=" * 80)
    print()

    var all_passed = True

    if not verify_n15():
        all_passed = False

    if not verify_n20():
        all_passed = False

    print()
    if all_passed:
        print("✓ All correctness checks passed!")
    else:
        print("✗ Some correctness checks failed!")
    print()

    return all_passed


fn main() raises:
    # Run verification first
    if not run_verification():
        print("ERROR: Correctness verification failed. Aborting benchmark.")
        return

    # Now run the benchmark
    print("=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    var runner = BenchmarkRunner("Execute Fused V3 vs SIMD V2")

    # Configure columns
    var param_cols = List[String]()
    param_cols.append("n")
    param_cols.append("circuit_type")
    param_cols.append("gates")
    runner.set_param_columns(param_cols^)

    var bench_cols = List[String]()
    bench_cols.append("v2_ms")
    bench_cols.append("v3_ms")
    runner.set_bench_columns(bench_cols^)

    alias iters = 10

    # Benchmark at n=15, 20, 25
    var test_sizes = List[Int]()
    test_sizes.append(15)
    test_sizes.append(20)
    test_sizes.append(25)

    for q_idx in range(3):
        var n = test_sizes[q_idx]
        print("Benchmarking n=" + String(n) + "...")

        # Test all three circuit types
        var circuit_types = List[String]()
        circuit_types.append("mixed")
        circuit_types.append("global-heavy")
        circuit_types.append("local-heavy")

        for c_idx in range(3):
            var circuit_type = circuit_types[c_idx]

            # Create circuit
            var circuit: QuantumCircuit
            if circuit_type == "mixed":
                circuit = create_mixed_circuit(n)
            elif circuit_type == "global-heavy":
                circuit = create_global_heavy_circuit(n)
            else:
                circuit = create_local_heavy_circuit(n)

            # Create params
            var params = Dict[String, String]()
            params["n"] = String(n)
            params["circuit_type"] = circuit_type
            params["gates"] = String(len(circuit.transformations))

            # Benchmark v2 and v3
            if n == 15:
                alias N = 32768
                var t0 = Int(perf_counter_ns())
                for _ in range(iters):
                    var state = QuantumState(n)
                    execute_simd_v2[15](state, circuit)
                    keep(state.re.unsafe_ptr())
                var t1 = Int(perf_counter_ns())
                var time_v2 = Float64(t1 - t0) / 1_000_000.0 / iters
                runner.add_result(params, "v2_ms", time_v2)

                t0 = Int(perf_counter_ns())
                for _ in range(iters):
                    var state = QuantumState(n)
                    execute_fused_v3[N](state, circuit)
                    keep(state.re.unsafe_ptr())
                t1 = Int(perf_counter_ns())
                var time_v3 = Float64(t1 - t0) / 1_000_000.0 / iters
                runner.add_result(params, "v3_ms", time_v3)
            elif n == 20:
                alias N = 1048576
                var t0 = Int(perf_counter_ns())
                for _ in range(iters):
                    var state = QuantumState(n)
                    execute_simd_v2[20](state, circuit)
                    keep(state.re.unsafe_ptr())
                var t1 = Int(perf_counter_ns())
                var time_v2 = Float64(t1 - t0) / 1_000_000.0 / iters
                runner.add_result(params, "v2_ms", time_v2)

                t0 = Int(perf_counter_ns())
                for _ in range(iters):
                    var state = QuantumState(n)
                    execute_fused_v3[N](state, circuit)
                    keep(state.re.unsafe_ptr())
                t1 = Int(perf_counter_ns())
                var time_v3 = Float64(t1 - t0) / 1_000_000.0 / iters
                runner.add_result(params, "v3_ms", time_v3)
            else:  # n == 25
                alias N = 33554432
                var t0 = Int(perf_counter_ns())
                for _ in range(iters):
                    var state = QuantumState(n)
                    execute_simd_v2[25](state, circuit)
                    keep(state.re.unsafe_ptr())
                var t1 = Int(perf_counter_ns())
                var time_v2 = Float64(t1 - t0) / 1_000_000.0 / iters
                runner.add_result(params, "v2_ms", time_v2)

                t0 = Int(perf_counter_ns())
                for _ in range(iters):
                    var state = QuantumState(n)
                    execute_fused_v3[N](state, circuit)
                    keep(state.re.unsafe_ptr())
                t1 = Int(perf_counter_ns())
                var time_v3 = Float64(t1 - t0) / 1_000_000.0 / iters
                runner.add_result(params, "v3_ms", time_v3)

    # Print results
    runner.print_table(show_winner=True)

    # Save to CSV
    var output_path = "benches/results/2025_12_24/v3_vs_v2"
    runner.save_csv(output_path)

    # Generate markdown report automatically
    print("\nGenerating report...")
    from python import Python

    var report_gen = Python.import_module("benches.generate_report")
    var csv_path = output_path + "_" + String(runner.timestamp) + ".csv"
    var report_path = "benches/reports/v3_vs_v2"
    _ = report_gen.generate_report([csv_path], report_path + ".md")

    print("\nLegend:")
    print("  v2_ms = execute_simd_v2 (baseline)")
    print("  v3_ms = execute_fused_v3 (with fusion)")
