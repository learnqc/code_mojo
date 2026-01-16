from collections import Dict, List

from butterfly.core.state import QuantumState
from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.types import FloatType, pi
from butterfly.core.executors import execute
from butterfly.utils.context import ExecContext, ExecutionStrategy
from butterfly.utils.benchmark_runner import (
    LabeledFunction,
    create_runner,
    should_autosave,
)
from testing import assert_true


fn build_prep_circuit(n: Int, v: FloatType) -> QuantumCircuit:
    var circuit = QuantumCircuit(n)
    for j in range(n):
        circuit.h(j)
    for j in range(n):
        circuit.p(j, 2 * pi / 2 ** (j + 1) * v)
    return circuit^


fn build_full_circuit(n: Int, v: FloatType) -> QuantumCircuit:
    from butterfly.algos.value_encoding_circuit import encode_value_circuit

    return encode_value_circuit(n, v)


fn build_circuit(n: Int, v: FloatType, stage: Int) -> QuantumCircuit:
    if stage == 0:
        return build_prep_circuit(n, v)
    return build_full_circuit(n, v)


fn apply_strategy(
    input: Tuple[Int, FloatType, Int],
    strategy: Int,
) raises -> QuantumState:
    var n = input[0]
    var v = input[1]
    var stage = input[2]
    var circuit = build_circuit(n, v, stage)
    var state = QuantumState(n)
    var ctx = ExecContext()
    ctx.execution_strategy = strategy
    execute(state, circuit, ctx)
    return state^


fn apply_scalar(input: Tuple[Int, FloatType, Int]) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.SCALAR)


fn apply_scalar_parallel(
    input: Tuple[Int, FloatType, Int]
) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.SCALAR_PARALLEL)


fn apply_simd(input: Tuple[Int, FloatType, Int]) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.SIMD)


fn apply_simd_generic(input: Tuple[Int, FloatType, Int]) raises -> QuantumState:
    var n = input[0]
    var v = input[1]
    var stage = input[2]
    var circuit = build_circuit(n, v, stage)
    var state = QuantumState(n)
    var ctx = ExecContext()
    ctx.execution_strategy = ExecutionStrategy.SIMD
    ctx.simd_use_specialized_h = False
    ctx.simd_use_specialized_p = False
    ctx.simd_use_specialized_cp = False
    ctx.simd_use_specialized_x = False
    ctx.simd_use_specialized_cx = False
    ctx.simd_use_specialized_ry = False
    ctx.simd_use_specialized_cry = False
    execute(state, circuit, ctx)
    return state^


fn apply_simd_parallel(
    input: Tuple[Int, FloatType, Int]
) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.SIMD_PARALLEL)


fn apply_grid(input: Tuple[Int, FloatType, Int]) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.GRID)


fn apply_grid_parallel(
    input: Tuple[Int, FloatType, Int]
) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.GRID_PARALLEL)


fn apply_grid_fused(input: Tuple[Int, FloatType, Int]) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.GRID_FUSED)


fn apply_grid_parallel_fused(
    input: Tuple[Int, FloatType, Int]
) raises -> QuantumState:
    return apply_strategy(input, ExecutionStrategy.GRID_PARALLEL_FUSED)


fn compare_states(a: QuantumState, b: QuantumState, tolerance: Float64) raises:
    if a.size() != b.size():
        raise Error("Verification failed: State sizes differ")
    for i in range(a.size()):
        var re_tol = tolerance * max(1.0, abs(a[i].re), abs(b[i].re))
        var im_tol = tolerance * max(1.0, abs(a[i].im), abs(b[i].im))
        assert_true(abs(a[i].re - b[i].re) <= re_tol)
        assert_true(abs(a[i].im - b[i].im) <= im_tol)


fn main() raises:
    alias NAME = "execution_strategies"
    alias DESCRIPTION = "Execution strategies: scalar/simd/grid and parallel variants"

    var param_cols = List[String]("n", "stage")
    var bench_cols = List[String](
        "scalar",
        "scalar_parallel",
        "simd",
        "simd_generic",
        "simd_parallel",
        "grid",
        "grid_fused",
        "grid_parallel",
        "grid_parallel_fused",
    )
    var runner = create_runner(NAME, DESCRIPTION, param_cols, bench_cols, 0)

    var n_values = List[Int](8, 9, 10, 12, 14, 16, 18, 20, 22, 24)
    var iters = 3
    var warmup_iters = 1
    var trials = 5
    var value = FloatType(4.7)

    alias Input = Tuple[Int, FloatType, Int]
    alias FnType = LabeledFunction[Input, QuantumState]
    var funcs_verify = List[FnType]()
    funcs_verify.append(FnType("scalar", apply_scalar))
    funcs_verify.append(FnType("scalar_parallel", apply_scalar_parallel))
    funcs_verify.append(FnType("simd", apply_simd))
    funcs_verify.append(FnType("simd_generic", apply_simd_generic))
    funcs_verify.append(FnType("simd_parallel", apply_simd_parallel))
    funcs_verify.append(FnType("grid", apply_grid))
    funcs_verify.append(FnType("grid_fused", apply_grid_fused))
    funcs_verify.append(FnType("grid_parallel", apply_grid_parallel))
    funcs_verify.append(
        FnType("grid_parallel_fused", apply_grid_parallel_fused)
    )

    var funcs = List[FnType]()
    funcs.append(FnType("simd", apply_simd))
    funcs.append(FnType("simd_parallel", apply_simd_parallel))
    funcs.append(FnType("grid", apply_grid))
    funcs.append(FnType("grid_fused", apply_grid_fused))
    funcs.append(FnType("grid_parallel", apply_grid_parallel))
    funcs.append(FnType("grid_parallel_fused", apply_grid_parallel_fused))

    for n in n_values:
        for stage in range(2):
            var params = Dict[String, String]()
            params["n"] = String(n)
            params["stage"] = "prep" if stage == 0 else "full"

            var input = (n, value, stage)
            if n <= 20:
                runner.verify(
                    input, funcs_verify, compare_states, tolerance=1e-10
                )
                runner.add_perf_results[Input, QuantumState](
                    params,
                    funcs_verify,
                    input,
                    iters,
                    3,
                    warmup_iters,
                    trials,
                )
            else:
                runner.add_perf_results[Input, QuantumState](
                    params,
                    funcs,
                    input,
                    iters,
                    3,
                    warmup_iters,
                    trials,
                )

    runner.print_table()
    runner.save_csv("benches/results/" + NAME, autosave=should_autosave())

    # Development (no CSV): pixi run mojo run -I . benches/bench_execution_strategies.mojo
    # Production (with CSV): pixi run mojo run -I . benches/bench_execution_strategies.mojo --autosave
