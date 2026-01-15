from butterfly.core.state import State
from utils.variant import Variant
from time import perf_counter_ns, sleep
import benchmark
from benchmark import keep, run, Unit
from butterfly.utils.benchmark_runner import create_runner
from collections import Dict, List


# --- State Test Functions ---


fn get_state_v1(n: Int) raises -> State:
    var state = State(n)
    # Simulate some work
    sleep(0.05)
    return state^


fn get_state_v2(n: Int) raises -> State:
    var state = State(n)
    # Simulate different work but same result
    sleep(0.07)
    return state^


# --- Custom Verifier (agnostic of runner) ---


fn compare_states(
    val1: State, val2: State, tolerance: Float64
) raises:
    """Specialized comparison for State."""
    if val1.size() != val2.size():
        raise Error("Verification failed: State sizes differ")

    var diff_sum = 0.0
    for i in range(val1.size()):
        var dr = val1.re[i] - val2.re[i]
        var di = val1.im[i] - val2.im[i]
        diff_sum += dr * dr + di * di

    if diff_sum > tolerance:
        raise Error(
            "Verification failed: States differ by " + String(diff_sum)
        )


fn test_main() raises:
    alias NAME = "benchmark_prototype"
    alias DESCRIPTION = "Agnostic Benchmark Verification Prototype"

    # Define columns
    var param_cols = List[String]("n")
    var bench_cols = List[String]("v1", "v2")

    # Create runner
    var runner = create_runner(NAME, DESCRIPTION, param_cols, bench_cols)

    # Hook-Based Verification (State)
    # The runner has NO knowledge of State. We inject the hook.
    for n in range(3, 6):
        runner.verify(
            n,
            get_state_v1,
            get_state_v2,
            compare=compare_states,
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

    # Save CSV only if --autosave flag is present
    from butterfly.utils.benchmark_runner import should_autosave

    runner.save_csv(NAME, autosave=should_autosave())

    # Usage:
    # Development (no CSV): pixi run mojo run -I . butterfly/utils/benchmark_prototype.mojo
    # Production (with CSV): pixi run mojo run -I . butterfly/utils/benchmark_prototype.mojo --autosave
    # Via suite runner: python benches/run_benchmark_suite.py --suite butterfly/utils/benhmark_suite_prototype.json --all
