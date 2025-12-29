from butterfly.core.state import QuantumState
from utils.variant import Variant
from time import perf_counter_ns, sleep
import benchmark
from benchmark import keep, run, Unit
from butterfly.utils.benchmark_runner import create_runner
from collections import Dict, List


# --- QuantumState Test Functions ---


fn get_state_v1(n: Int) -> QuantumState:
    var state = QuantumState(n)
    # Simulate some work
    sleep(0.05)
    return state^


fn get_state_v2(n: Int) -> QuantumState:
    var state = QuantumState(n)
    # Simulate different work but same result
    sleep(0.07)
    return state^


# --- Custom Verifier (agnostic of runner) ---


fn compare_quantum_states(
    val1: QuantumState, val2: QuantumState, tolerance: Float64
) raises:
    """Specialized comparison for QuantumState."""
    if val1.size() != val2.size():
        raise Error("Verification failed: State sizes differ")

    var diff_sum = 0.0
    for i in range(val1.size()):
        var dr = val1.re[i] - val2.re[i]
        var di = val1.im[i] - val2.im[i]
        diff_sum += dr * dr + di * di

    if diff_sum > tolerance:
        raise Error(
            "Verification failed: QuantumStates differ by " + String(diff_sum)
        )


fn main() raises:
    alias NAME = "quantum_prototype"
    alias DESCRIPTION = "Agnostic Benchmark Verification Prototype"

    # Define columns
    var param_cols = List[String]("n")
    var bench_cols = List[String]("v1", "v2")

    # Create runner
    var runner = create_runner(NAME, DESCRIPTION, param_cols, bench_cols)

    # Hook-Based Verification (QuantumState)
    # The runner has NO knowledge of QuantumState. We inject the hook.
    for n in range(3, 6):
        runner.verify(
            n,
            get_state_v1,
            get_state_v2,
            compare=compare_quantum_states,
            name1="state_v1",
            name2="state_v2",
            tolerance=1e-10,
        )

    # Run benchmark sweep
    for n in range(3, 6):
        var params = Dict[String, String]()
        params["n"] = String(n)

        runner.add_perf_result(params, "v1", get_state_v1, n)
        runner.add_perf_result(params, "v2", get_state_v2, n)

    runner.print_table()

    # optionally save to csv
    # runner.save_csv(NAME)

    # run with the bencmark runner
    # python benches/run_benchmark_suite.py --suite butterfly/utils/benhmark_suite_prototype.json --all
