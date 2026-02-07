from collections import Dict, List

from butterfly.core.state import QuantumState
from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.types import FloatType
from butterfly.core.executors import execute
from butterfly.core.transformations_fusion import fuse_circuit_specialized_pairs
from butterfly.utils.context import ExecContext, ExecutionStrategy
from butterfly.utils.benchmark_runner import (
    LabeledFunction,
    create_runner,
    should_autosave,
)
from testing import assert_true


fn build_circuit(n: Int, v: FloatType) -> QuantumCircuit:
    from butterfly.algos.value_encoding_circuit import encode_value_circuit
    return encode_value_circuit(n, v)


fn apply_strategy(
    input: Tuple[Int, FloatType],
    strategy: ExecutionStrategy,
    fused: Bool,
) raises -> QuantumState:
    var n = input[0]
    var v = input[1]
    var circuit = build_circuit(n, v)
    if (
        fused
        and strategy != ExecutionStrategy.GRID_FUSED
        and strategy != ExecutionStrategy.GRID_PARALLEL_FUSED
    ):
        circuit = fuse_circuit_specialized_pairs(circuit)
    var state = QuantumState(n)
    var ctx = ExecContext()
    ctx.execution_strategy = strategy
    execute(state, circuit, ctx)
    return state^


fn apply_scalar(input: Tuple[Int, FloatType]) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.SCALAR, False)


fn apply_simd(input: Tuple[Int, FloatType]) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.SIMD, False)


fn apply_simd_fused(input: Tuple[Int, FloatType]) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.SIMD, True)


fn apply_simd_parallel(input: Tuple[Int, FloatType]) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.SIMD_PARALLEL, False)


fn apply_simd_parallel_fused(
    input: Tuple[Int, FloatType]
) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.SIMD_PARALLEL, True)


fn apply_grid(input: Tuple[Int, FloatType]) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.GRID, False)


fn apply_grid_fused(input: Tuple[Int, FloatType]) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.GRID_FUSED, True)


fn apply_grid_parallel(input: Tuple[Int, FloatType]) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.GRID_PARALLEL, False)


fn apply_grid_parallel_fused(
    input: Tuple[Int, FloatType]
) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.GRID_PARALLEL_FUSED, True)


fn qiskit_available() -> Bool:
    try:
        from python import Python

        _ = Python.import_module("qiskit")
        return True
    except:
        return False


fn apply_qiskit_transpile(input: Tuple[Int, FloatType, Int]) raises -> Int:
    from python import Python

    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, ".")
    var runner = Python.import_module("benches.qiskit_runner")
    runner.transpile_value_encoding(
        Int(input[0]),
        FloatType(input[1]),
        "full",
        Int(input[2]),
    )
    return 0


fn apply_qiskit_run_cached(input: Tuple[Int, FloatType, Int]) raises -> Int:
    from python import Python

    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, ".")
    var runner = Python.import_module("benches.qiskit_runner")
    runner.run_cached_statevector(
        Int(input[0]),
        FloatType(input[1]),
        "full",
        Int(input[2]),
    )
    return 0


fn compare_states(a: QuantumState, b: QuantumState, tolerance: FloatType) raises:
    if a.size() != b.size():
        raise Error("Verification failed: State sizes differ")
    for i in range(a.size()):
        var re_tol = tolerance * max(1.0, abs(a[i].re), abs(b[i].re))
        var im_tol = tolerance * max(1.0, abs(a[i].im), abs(b[i].im))
        assert_true(abs(a[i].re - b[i].re) <= re_tol)
        assert_true(abs(a[i].im - b[i].im) <= im_tol)


fn build_funcs(
    n: Int,
) -> List[LabeledFunction[Tuple[Int, FloatType], QuantumState]]:
    alias Input = Tuple[Int, FloatType]
    alias FnType = LabeledFunction[Input, QuantumState]
    var funcs = List[FnType]()
    if n <= 20:
        funcs.append(FnType("scalar", apply_scalar))
        funcs.append(FnType("simd", apply_simd))
        funcs.append(FnType("simd_fused", apply_simd_fused))
        funcs.append(FnType("simd_parallel", apply_simd_parallel))
        funcs.append(FnType("grid", apply_grid))
        funcs.append(FnType("grid_fused", apply_grid_fused))
        funcs.append(FnType("grid_parallel", apply_grid_parallel))
        funcs.append(FnType("grid_parallel_fused", apply_grid_parallel_fused))
    else:
        funcs.append(FnType("simd_parallel", apply_simd_parallel))
        funcs.append(FnType("grid_parallel", apply_grid_parallel))
        funcs.append(FnType("grid_parallel_fused", apply_grid_parallel_fused))
    return funcs^


fn main() raises:
    alias NAME = "value_encoding_strategies_qiskit"
    alias DESCRIPTION = "Value encoding strategies + Qiskit (prep+iqft)"

    var param_cols = List[String]("n")
    var bench_cols = List[String](
        "scalar",
        "simd",
        "simd_fused",
        "simd_parallel",
        "grid",
        "grid_fused",
        "grid_parallel",
        "grid_parallel_fused",
        "qiskit_r_o3",
    )
    var runner = create_runner(NAME, DESCRIPTION, param_cols, bench_cols, 0)

    var n_values = List[Int]()
    for n in range(3, 26, 2):
        n_values.append(n)

    var iters = 3
    var warmup_iters = 1
    var trials = 5
    var value = FloatType(4.7)
    var qiskit_ok = qiskit_available()

    if not qiskit_ok:
        print("Qiskit not available; skipping Qiskit benchmarks.")

    alias Input = Tuple[Int, FloatType]
    alias QInput = Tuple[Int, FloatType, Int]
    alias QFnType = LabeledFunction[QInput, Int]

    for n in n_values:
        var params = Dict[String, String]()
        params["n"] = String(n)
        var input = (n, value)

        var funcs = build_funcs(n)
        if n <= 20:
            runner.verify(input, funcs, compare_states, tolerance=1e-10)

        runner.add_perf_results[Input, QuantumState](
            params,
            funcs,
            input,
            iters,
            3,
            warmup_iters,
            trials,
        )

        if qiskit_ok:
            var opt_level = 3
            _ = apply_qiskit_transpile((n, value, opt_level))
            var q_funcs = List[QFnType]()
            q_funcs.append(QFnType("qiskit_r_o3", apply_qiskit_run_cached))
            runner.add_perf_results[QInput, Int](
                params,
                q_funcs,
                (n, value, opt_level),
                1,
                3,
                0,
                1,
            )

    runner.print_table()
    runner.save_csv("benches/results/" + NAME, autosave=should_autosave())

    # Development: pixi run mojo run -I . benches/bench_value_encoding_strategies_qiskit.mojo
    # Production:  pixi run mojo run -I . benches/bench_value_encoding_strategies_qiskit.mojo --autosave
