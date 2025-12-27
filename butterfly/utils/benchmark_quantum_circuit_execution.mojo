"""
Utility for benchmarking quantum circuit execution functions.

Provides a unified interface to verify and benchmark any functions that
execute quantum circuits and return quantum states.
"""

from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.grid_state import GridQuantumState
from butterfly.core.grid_state_hybrid import HybridGrid
from butterfly.core.types import FloatType
from butterfly.utils.benchmark_verify import verify_states_equal
from collections import List
from time import perf_counter_ns
from benchmark import keep


fn grid_to_quantum_state(state: GridQuantumState) -> QuantumState:
    """Convert GridQuantumState to QuantumState for verification."""
    var row_re = List[FloatType]()
    var row_im = List[FloatType]()
    for r in range(state.num_rows):
        for c in range(state.row_size):
            row_re.append(state.re[state.get_row_offset(r) + c])
            row_im.append(state.im[state.get_row_offset(r) + c])
    return QuantumState(row_re^, row_im^)


fn hybrid_to_quantum_state(state: HybridGrid) -> QuantumState:
    """Convert HybridGrid to QuantumState for verification."""
    var row_re = List[FloatType]()
    var row_im = List[FloatType]()
    for r in range(state.num_rows):
        for c in range(state.row_size):
            row_re.append(state.re[r * state.row_size + c])
            row_im.append(state.im[r * state.row_size + c])
    return QuantumState(row_re^, row_im^)


fn verify_circuit_executions(
    functions: List[fn (QuantumCircuit) -> QuantumState],
    names: List[String],
    circuit: QuantumCircuit,
    tolerance: Float64 = 1e-5,
    verbose: Bool = True,
) raises:
    """Verify all execution functions produce the same result.

    Uses the first function as the reference and compares all others against it.

    Args:
        functions: List of functions (each takes circuit, returns state).
        names: List of names for the functions (for error messages).
        circuit: Circuit to execute.
        tolerance: Maximum allowed difference.
        verbose: Print verification status.

    Raises:
        Error if any function produces different results.
    """
    if len(functions) < 2:
        return  # Nothing to verify

    if verbose:
        print("  Verifying functions...", end="")

    # Run reference (first function)
    var reference = functions[0](circuit)

    # Verify all others match
    for i in range(1, len(functions)):
        var test_state = functions[i](circuit)
        _ = verify_states_equal(
            reference,
            test_state,
            tolerance,
            names[0],
            names[i],
        )

    if verbose:
        print(" ✓")


fn benchmark_circuit_executions(
    functions: List[fn (QuantumCircuit) -> QuantumState],
    circuit: QuantumCircuit,
    iters: Int = 5,
) -> List[Float64]:
    """Benchmark execution functions on the same circuit.

    Args:
        functions: List of functions to benchmark.
        circuit: Circuit to execute.
        iters: Number of iterations per function.

    Returns:
        List of average times in milliseconds.
    """
    var times = List[Float64]()

    for i in range(len(functions)):
        var t0 = Int(perf_counter_ns())
        for _ in range(iters):
            var state = functions[i](circuit)
            keep(state.re.unsafe_ptr())
        var t1 = Int(perf_counter_ns())
        var time_ms = Float64(t1 - t0) / 1_000_000.0 / iters
        times.append(time_ms)

    return times^


fn verify_and_benchmark_circuit_executions(
    functions: List[fn (QuantumCircuit) -> QuantumState],
    names: List[String],
    circuit: QuantumCircuit,
    iters: Int = 5,
    tolerance: Float64 = 1e-5,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Verify and benchmark execution functions in one call.

    This is the main entry point for benchmarking circuit execution functions.
    It first verifies all functions produce the same result, then benchmarks them.

    Args:
        functions: List of functions to verify and benchmark.
        names: List of names for the functions.
        circuit: Circuit to execute.
        iters: Number of iterations per benchmark.
        tolerance: Maximum allowed difference for verification.
        verbose: Print verification status.

    Returns:
        List of average times in milliseconds.

    Raises:
        Error if verification fails.

    Example:
        ```mojo
        # Import your execution functions
        fn execute_generic(c: QuantumCircuit) -> QuantumState:
            var circuit = c.copy()
            return circuit.run_with_strategy(GENERIC)

        fn execute_simd(c: QuantumCircuit) -> QuantumState:
            var circuit = c.copy()
            return circuit.run_with_strategy(SIMD_STRATEGY)

        # Create lists
        var funcs = List[fn(QuantumCircuit) -> QuantumState]()
        funcs.append(execute_generic)
        funcs.append(execute_simd)

        var names = List[String]()
        names.append("Generic")
        names.append("SIMD")

        # Verify and benchmark
        var times = verify_and_benchmark_circuit_executions(
            funcs, names, my_circuit
        )
        ```
    """
    # Verify correctness first
    verify_circuit_executions(functions, names, circuit, tolerance, verbose)

    # Then benchmark
    return benchmark_circuit_executions(functions, circuit, iters)


fn create_quantum_circuit_execution_benchmark(
    functions: List[fn (QuantumCircuit) -> QuantumState],
    names: List[String],
    descriptions: List[String],
    circuit_builder: fn (Int, Float64) -> QuantumCircuit,
    test_cases: List[Tuple[Int, Float64]],
    benchmark_id: String,
    display_name: String,
    output_path: String = "",
    verify: Bool = True,
    iters: Int = 5,
    tolerance: Float64 = 1e-5,
    # Optional: GridQuantumState functions
    grid_functions: List[fn (QuantumCircuit) -> GridQuantumState] = List[
        fn (QuantumCircuit) -> GridQuantumState
    ](),
    grid_names: List[String] = List[String](),
    grid_descriptions: List[String] = List[String](),
    # Optional: HybridGrid functions
    hybrid_functions: List[fn (QuantumCircuit) -> HybridGrid] = List[
        fn (QuantumCircuit) -> HybridGrid
    ](),
    hybrid_names: List[String] = List[String](),
    hybrid_descriptions: List[String] = List[String](),
) raises:
    """Create a complete quantum circuit execution benchmark with verification, timing, and reporting.

    This is the highest-level utility that handles everything:
    - Verification of all functions
    - Benchmarking across test cases
    - Table printing with winner
    - CSV export
    - Function descriptions

    Args:
        functions: List of execution functions to benchmark.
        names: List of names for the functions.
        descriptions: List of descriptions for the functions.
        circuit_builder: Function that builds a circuit given (n, value).
        test_cases: List of (n, value) tuples to test.
        benchmark_id: Identifier for filename/JSON key (e.g., "value_encoding_strategies").
        display_name: Human-readable name for output (e.g., "Value Encoding Strategies").
        output_path: Path to save CSV (default: auto-generated from benchmark_id and date).
        verify: Whether to verify correctness before benchmarking (default: True).
        iters: Number of iterations per benchmark.
        tolerance: Maximum allowed difference for verification.
    """
    from butterfly.utils.benchmark_runner import BenchmarkRunner
    from butterfly.utils.benchmark_utils import get_date_string
    from collections import Dict

    var runner = BenchmarkRunner(display_name)

    # Auto-construct output path if not provided
    var final_output_path = output_path
    if not output_path:
        var date = get_date_string()
        final_output_path = "benches/results/" + date + "/" + benchmark_id

    # Merge all function names and descriptions
    var all_names = List[String]()
    var all_descriptions = List[String]()

    # Add QuantumState functions
    for i in range(len(names)):
        all_names.append(names[i])
        all_descriptions.append(descriptions[i])

    # Add Grid functions
    for i in range(len(grid_names)):
        all_names.append(grid_names[i])
        all_descriptions.append(grid_descriptions[i])

    # Add Hybrid functions
    for i in range(len(hybrid_names)):
        all_names.append(hybrid_names[i])
        all_descriptions.append(hybrid_descriptions[i])

    # Configure columns
    var param_cols = List[String]()
    param_cols.append("n")
    param_cols.append("value")
    runner.set_param_columns(param_cols^)

    var bench_cols = List[String]()
    for i in range(len(all_names)):
        bench_cols.append(all_names[i])
    runner.set_bench_columns(bench_cols^)

    print("Benchmarking " + display_name)
    print("=" * 80)
    print()

    # Run benchmarks for each test case
    for i in range(len(test_cases)):
        var n = test_cases[i][0]
        var value = test_cases[i][1]

        print("n=" + String(n) + ", value=" + String(value))

        # Create parameter dict
        var params = Dict[String, String]()
        params["n"] = String(n)
        params["value"] = String(value)

        # Build circuit
        var circuit = circuit_builder(n, value)

        # Verify all functions produce same result (if enabled)
        if verify and len(functions) > 0:
            # Use first QuantumState function as reference
            var reference = functions[0](circuit)

            # Verify other QuantumState functions
            for j in range(1, len(functions)):
                var test_state = functions[j](circuit)
                _ = verify_states_equal(
                    reference, test_state, tolerance, names[0], names[j]
                )

            # Verify Grid functions (convert to QuantumState first)
            for j in range(len(grid_functions)):
                var grid_state = grid_functions[j](circuit)
                var converted = grid_to_quantum_state(grid_state)
                _ = verify_states_equal(
                    reference, converted, tolerance, names[0], grid_names[j]
                )

            # Verify Hybrid functions (convert to QuantumState first)
            for j in range(len(hybrid_functions)):
                var hybrid_state = hybrid_functions[j](circuit)
                var converted = hybrid_to_quantum_state(hybrid_state)
                _ = verify_states_equal(
                    reference, converted, tolerance, names[0], hybrid_names[j]
                )

            print("  Verifying functions... ✓")

        # Benchmark all functions
        var all_times = List[Float64]()

        # Benchmark QuantumState functions
        for j in range(len(functions)):
            var t0 = Int(perf_counter_ns())
            for _ in range(iters):
                var state = functions[j](circuit)
                keep(state.re.unsafe_ptr())
            var t1 = Int(perf_counter_ns())
            var time_ms = Float64(t1 - t0) / 1_000_000.0 / iters
            all_times.append(time_ms)

        # Benchmark Grid functions
        for j in range(len(grid_functions)):
            var t0 = Int(perf_counter_ns())
            for _ in range(iters):
                var state = grid_functions[j](circuit)
                keep(state.re.unsafe_ptr())
            var t1 = Int(perf_counter_ns())
            var time_ms = Float64(t1 - t0) / 1_000_000.0 / iters
            all_times.append(time_ms)

        # Benchmark Hybrid functions
        for j in range(len(hybrid_functions)):
            var t0 = Int(perf_counter_ns())
            for _ in range(iters):
                var state = hybrid_functions[j](circuit)
                keep(state.re.unsafe_ptr())
            var t1 = Int(perf_counter_ns())
            var time_ms = Float64(t1 - t0) / 1_000_000.0 / iters
            all_times.append(time_ms)

        # Add results to runner
        for j in range(len(all_names)):
            runner.add_result(params, all_names[j], all_times[j])

    # Print results table with winner
    runner.print_table(show_winner=True)

    # Save to CSV
    runner.save_csv(final_output_path)

    # Print function descriptions
    print("\nFunctions:")
    for i in range(len(all_names)):
        print("  " + all_names[i].ljust(20) + " = " + all_descriptions[i])
    print("\nResults saved with timestamp:", String(runner.timestamp))
