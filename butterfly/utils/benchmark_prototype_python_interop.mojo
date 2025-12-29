from butterfly.core.state import QuantumState
from butterfly.utils.quantum_interop import get_qiskit_state
from butterfly.utils.benchmark_runner import create_runner
from collections import Dict, List


# --- The 1-Line Python-to-Mojo Wrapper ---
# Now using the domain-specific assembler which leverages generic interop.
fn executor_python(n: Int) raises -> QuantumState:
    return get_qiskit_state(n, 0.0)


fn executor_mojo(n: Int) raises -> QuantumState:
    return QuantumState(n)


fn compare_states(s1: QuantumState, s2: QuantumState, tol: Float64) raises:
    if s1.size() != s2.size():
        raise Error("Size mismatch")
    var diff_sum = 0.0
    for i in range(s1.size()):
        var dr = s1.re[i] - s2.re[i]
        var di = s1.im[i] - s2.im[i]
        diff_sum += dr * dr + di * di
    if diff_sum > tol:
        raise Error("Difference too large: " + String(diff_sum))


fn main() raises:
    var p_cols = List[String]("n")
    var b_cols = List[String]("mojo", "python")
    var runner = create_runner(
        "agnostic_refinement", "Decoupled Interop Demo", p_cols, b_cols
    )

    var n = 4
    # Verification hook ensures bit-perfect parity before benchmarking
    runner.verify(n, executor_mojo, executor_python, compare=compare_states)

    var params = Dict[String, String]()
    params["n"] = String(n)
    runner.add_perf_result(params, "mojo", executor_mojo, n)
    runner.add_perf_result(params, "python", executor_python, n)

    runner.print_table()
