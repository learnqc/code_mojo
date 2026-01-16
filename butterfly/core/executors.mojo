from algorithm import parallelize
from butterfly.core.state import QuantumState, simd_width
from butterfly.core.quantum_circuit import (
    QuantumCircuit,
    ClassicalTransform,
    MeasurementTransform,
    QuantumTransformation,
)
from butterfly.core.circuit import (
    ControlKind,
    GateTransformation,
    FusedPairTransformation,
    UnitaryTransformation,
    ControlledUnitaryTransformation,
    SwapTransformation,
    QubitReversalTransformation,
)
from butterfly.core.transformations_scalar import (
    transform_scalar,
    c_transform_scalar,
    mc_transform_scalar,
)
from butterfly.core.transformations_simd import (
    transform_simd,
    c_transform_simd,
    mc_transform_simd,
)
from butterfly.core.transformations_simd_parallel import (
    transform_gate_simd_parallel,
    transform_h_simd_parallel,
    transform_p_simd_parallel,
    transform_x_simd_parallel,
    transform_ry_simd_parallel,
    c_transform_p_simd_parallel,
    c_transform_x_simd_parallel,
    c_transform_ry_simd_parallel,
    mc_transform_simd_parallel,
)
from butterfly.core.transformations_grid import (
    transform_grid,
    c_transform_p_grid,
    transform_row_h_simd,
    transform_row_p_simd,
    transform_row_x_simd,
    transform_row_z_simd,
    transform_row_simd,
    c_transform_row_h_simd,
    c_transform_row_p_simd,
    transform_column_fused_hp_simd_tiled,
    L2_TILE_COLS,
)
from butterfly.core.transformations_unitary import (
    apply_unitary,
    apply_controlled_unitary,
)
from butterfly.core.transformations_fusion import apply_fused_pair
from butterfly.core.fused_kernels_sparse_row import (
    transform_row_fused_hh_simd,
    transform_row_fused_hp_simd,
    transform_row_fused_st_hp_simd,
    transform_row_fused_pp_simd,
    transform_row_fused_shared_c_pp_simd,
)
from butterfly.core.gates import GateKind, X
from butterfly.core.types import FloatType, Gate
from butterfly.utils.context import ExecContext, ExecutionStrategy


fn sort_targets(mut targets: List[Int]):
    var n = len(targets)
    for i in range(n):
        for j in range(i + 1, n):
            if targets[i] > targets[j]:
                var tmp = targets[i]
                targets[i] = targets[j]
                targets[j] = tmp


fn execute_scalar(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    for tr in circuit.transformations:
        if tr.isa[FusedPairTransformation]():
            var pair_tr = tr[FusedPairTransformation].copy()
            apply_fused_pair(state, pair_tr)
        elif tr.isa[GateTransformation]():
            var gate_tr = tr[GateTransformation].copy()
            if gate_tr.target < 0 or gate_tr.target >= state.size():
                raise Error("Target out of bounds: " + String(gate_tr.target))
            for c in gate_tr.controls:
                if c < 0 or c >= state.size():
                    raise Error("Control out of bounds: " + String(c))
            var gate_arg = FloatType(0)
            if gate_tr.gate_info.arg:
                gate_arg = gate_tr.gate_info.arg.value()
            if gate_tr.kind == ControlKind.NO_CONTROL:
                transform_scalar(
                    state,
                    gate_tr.target,
                    gate_tr.gate_info.gate,
                    gate_tr.gate_info.kind,
                    gate_arg,
                    ctx,
                )
            elif gate_tr.kind == ControlKind.SINGLE_CONTROL:
                c_transform_scalar(
                    state,
                    gate_tr.controls[0],
                    gate_tr.target,
                    gate_tr.gate_info.gate,
                    gate_tr.gate_info.kind,
                    gate_arg,
                    ctx,
                )
            else:
                mc_transform_scalar(
                    state,
                    gate_tr.controls,
                    gate_tr.target,
                    gate_tr.gate_info.gate,
                    ctx,
                )
        elif tr.isa[SwapTransformation]():
            var swap_tr = tr[SwapTransformation].copy()
            if swap_tr.a < 0 or swap_tr.a >= state.size():
                raise Error("Target out of bounds: " + String(swap_tr.a))
            if swap_tr.b < 0 or swap_tr.b >= state.size():
                raise Error("Target out of bounds: " + String(swap_tr.b))
            if swap_tr.a != swap_tr.b:
                c_transform_scalar(
                    state, swap_tr.a, swap_tr.b, X, GateKind.X, 0, ctx
                )
                c_transform_scalar(
                    state, swap_tr.b, swap_tr.a, X, GateKind.X, 0, ctx
                )
                c_transform_scalar(
                    state, swap_tr.a, swap_tr.b, X, GateKind.X, 0, ctx
                )
        elif tr.isa[QubitReversalTransformation]():
            var qrev_tr = tr[QubitReversalTransformation].copy()
            var targets = qrev_tr.targets.copy()
            if len(targets) == 0:
                var tmp = state.size()
                var nbits = 0
                while tmp > 1:
                    tmp //= 2
                    nbits += 1
                targets = List[Int](capacity=nbits)
                for i in range(nbits):
                    targets.append(i)
            sort_targets(targets)
            for t in targets:
                if t < 0 or t >= state.size():
                    raise Error("Target out of bounds: " + String(t))
            var half = len(targets) // 2
            for i in range(half):
                var a = targets[i]
                var b = targets[len(targets) - 1 - i]
                if a == b:
                    continue
                c_transform_scalar(state, a, b, X, GateKind.X, 0, ctx)
                c_transform_scalar(state, b, a, X, GateKind.X, 0, ctx)
                c_transform_scalar(state, a, b, X, GateKind.X, 0, ctx)
        elif tr.isa[UnitaryTransformation]():
            var unitary_tr = tr[UnitaryTransformation].copy()
            apply_unitary(state, unitary_tr.u, unitary_tr.target, unitary_tr.m)
        elif tr.isa[ControlledUnitaryTransformation]():
            var c_unitary_tr = tr[ControlledUnitaryTransformation].copy()
            apply_controlled_unitary(
                state,
                c_unitary_tr.u,
                c_unitary_tr.control,
                c_unitary_tr.target,
                c_unitary_tr.m,
            )
        elif tr.isa[MeasurementTransform]():
            var meas_tr = tr[MeasurementTransform].copy()
            meas_tr.apply(
                state,
                meas_tr.targets,
                meas_tr.values,
                meas_tr.seed,
            )
        else:
            var cl_tr = tr[ClassicalTransform].copy()
            cl_tr.apply(state, cl_tr.targets)


fn execute_scalar_parallel(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    threads: Int = 4,
) raises:
    var ctx = ExecContext()
    ctx.threads = threads
    execute_scalar(state, circuit, ctx)


fn execute(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    var exec_ctx = ctx.copy()
    var strategy = exec_ctx.execution_strategy
    if strategy == ExecutionStrategy.SCALAR:
        execute_scalar(state, circuit, exec_ctx)
        return
    if strategy == ExecutionStrategy.SCALAR_PARALLEL:
        if exec_ctx.threads <= 0:
            exec_ctx.threads = 4
        execute_scalar(state, circuit, exec_ctx)
        return
    if strategy == ExecutionStrategy.SIMD:
        execute_simd(state, circuit, exec_ctx)
        return
    if strategy == ExecutionStrategy.SIMD_PARALLEL:
        execute_simd_parallel(state, circuit, exec_ctx)
        return
    if strategy == ExecutionStrategy.GRID:
        exec_ctx.grid_use_parallel = False
        execute_grid(state, circuit, exec_ctx)
        return
    if strategy == ExecutionStrategy.GRID_PARALLEL:
        exec_ctx.grid_use_parallel = True
        execute_grid(state, circuit, exec_ctx)
        return
    if strategy == ExecutionStrategy.GRID_FUSED:
        exec_ctx.grid_use_parallel = False
        execute_grid_fused(state, circuit, exec_ctx)
        return
    if strategy == ExecutionStrategy.GRID_PARALLEL_FUSED:
        exec_ctx.grid_use_parallel = True
        execute_grid_fused(state, circuit, exec_ctx)
        return
    raise Error("Unknown execution strategy: " + String(strategy))


fn execute_simd(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    for tr in circuit.transformations:
        if tr.isa[FusedPairTransformation]():
            var pair_tr = tr[FusedPairTransformation].copy()
            apply_fused_pair(state, pair_tr)
        elif tr.isa[GateTransformation]():
            var gate_tr = tr[GateTransformation].copy()
            if gate_tr.target < 0 or gate_tr.target >= state.size():
                raise Error("Target out of bounds: " + String(gate_tr.target))
            for c in gate_tr.controls:
                if c < 0 or c >= state.size():
                    raise Error("Control out of bounds: " + String(c))
            var gate_arg = FloatType(0)
            if gate_tr.gate_info.arg:
                gate_arg = gate_tr.gate_info.arg.value()
            if gate_tr.kind == ControlKind.NO_CONTROL:
                transform_simd(
                    state,
                    gate_tr.target,
                    gate_tr.gate_info.gate,
                    gate_tr.gate_info.kind,
                    gate_arg,
                    ctx,
                )
            elif gate_tr.kind == ControlKind.SINGLE_CONTROL:
                c_transform_simd(
                    state,
                    gate_tr.controls[0],
                    gate_tr.target,
                    gate_tr.gate_info.gate,
                    gate_tr.gate_info.kind,
                    gate_arg,
                    ctx,
                )
            else:
                mc_transform_simd(
                    state,
                    gate_tr.controls,
                    gate_tr.target,
                    gate_tr.gate_info.gate,
                )
        elif tr.isa[SwapTransformation]():
            var swap_tr = tr[SwapTransformation].copy()
            if swap_tr.a < 0 or swap_tr.a >= state.size():
                raise Error("Target out of bounds: " + String(swap_tr.a))
            if swap_tr.b < 0 or swap_tr.b >= state.size():
                raise Error("Target out of bounds: " + String(swap_tr.b))
            if swap_tr.a != swap_tr.b:
                mc_transform_simd(state, List[Int](swap_tr.a), swap_tr.b, X)
                mc_transform_simd(state, List[Int](swap_tr.b), swap_tr.a, X)
                mc_transform_simd(state, List[Int](swap_tr.a), swap_tr.b, X)
        elif tr.isa[QubitReversalTransformation]():
            var qrev_tr = tr[QubitReversalTransformation].copy()
            var targets = qrev_tr.targets.copy()
            if len(targets) == 0:
                var tmp = state.size()
                var nbits = 0
                while tmp > 1:
                    tmp //= 2
                    nbits += 1
                targets = List[Int](capacity=nbits)
                for i in range(nbits):
                    targets.append(i)
            sort_targets(targets)
            for t in targets:
                if t < 0 or t >= state.size():
                    raise Error("Target out of bounds: " + String(t))
            var half = len(targets) // 2
            for i in range(half):
                var a = targets[i]
                var b = targets[len(targets) - 1 - i]
                if a == b:
                    continue
                mc_transform_simd(state, List[Int](a), b, X)
                mc_transform_simd(state, List[Int](b), a, X)
                mc_transform_simd(state, List[Int](a), b, X)
        elif tr.isa[UnitaryTransformation]():
            var unitary_tr = tr[UnitaryTransformation].copy()
            apply_unitary(state, unitary_tr.u, unitary_tr.target, unitary_tr.m)
        elif tr.isa[ControlledUnitaryTransformation]():
            var c_unitary_tr = tr[ControlledUnitaryTransformation].copy()
            apply_controlled_unitary(
                state,
                c_unitary_tr.u,
                c_unitary_tr.control,
                c_unitary_tr.target,
                c_unitary_tr.m,
            )
        elif tr.isa[MeasurementTransform]():
            var meas_tr = tr[MeasurementTransform].copy()
            meas_tr.apply(
                state,
                meas_tr.targets,
                meas_tr.values,
                meas_tr.seed,
            )
        else:
            var cl_tr = tr[ClassicalTransform].copy()
            cl_tr.apply(state, cl_tr.targets)


fn execute_simd_parallel(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    for tr in circuit.transformations:
        if tr.isa[FusedPairTransformation]():
            var pair_tr = tr[FusedPairTransformation].copy()
            apply_fused_pair(state, pair_tr)
        elif tr.isa[GateTransformation]():
            var gate_tr = tr[GateTransformation].copy()
            if gate_tr.target < 0 or gate_tr.target >= state.size():
                raise Error("Target out of bounds: " + String(gate_tr.target))
            for c in gate_tr.controls:
                if c < 0 or c >= state.size():
                    raise Error("Control out of bounds: " + String(c))
            var gate_arg = FloatType(0)
            if gate_tr.gate_info.arg:
                gate_arg = gate_tr.gate_info.arg.value()
            if gate_tr.kind == ControlKind.NO_CONTROL:
                if (
                    gate_tr.gate_info.kind == GateKind.H
                    and ctx.simd_use_specialized_h
                ):
                    transform_h_simd_parallel(state, gate_tr.target, ctx)
                elif (
                    gate_tr.gate_info.kind == GateKind.P
                    and ctx.simd_use_specialized_p
                ):
                    transform_p_simd_parallel(
                        state,
                        gate_tr.target,
                        Float64(gate_arg),
                        ctx,
                    )
                elif (
                    gate_tr.gate_info.kind == GateKind.X
                    and ctx.simd_use_specialized_x
                ):
                    transform_x_simd_parallel(state, gate_tr.target, ctx)
                elif (
                    gate_tr.gate_info.kind == GateKind.RY
                    and ctx.simd_use_specialized_ry
                ):
                    transform_ry_simd_parallel(
                        state,
                        gate_tr.target,
                        Float64(gate_arg),
                        ctx,
                    )
                else:
                    transform_gate_simd_parallel(
                        state,
                        gate_tr.target,
                        gate_tr.gate_info.gate,
                        ctx,
                    )
            elif gate_tr.kind == ControlKind.SINGLE_CONTROL:
                if (
                    gate_tr.gate_info.kind == GateKind.P
                    and ctx.simd_use_specialized_cp
                ):
                    c_transform_p_simd_parallel(
                        state,
                        gate_tr.controls[0],
                        gate_tr.target,
                        Float64(gate_arg),
                        ctx,
                    )
                elif (
                    gate_tr.gate_info.kind == GateKind.X
                    and ctx.simd_use_specialized_cx
                ):
                    c_transform_x_simd_parallel(
                        state,
                        gate_tr.controls[0],
                        gate_tr.target,
                        ctx,
                    )
                elif (
                    gate_tr.gate_info.kind == GateKind.RY
                    and ctx.simd_use_specialized_cry
                ):
                    c_transform_ry_simd_parallel(
                        state,
                        gate_tr.controls[0],
                        gate_tr.target,
                        Float64(gate_arg),
                        ctx,
                    )
                else:
                    mc_transform_simd_parallel(
                        state,
                        gate_tr.controls,
                        gate_tr.target,
                        gate_tr.gate_info.gate,
                        ctx,
                    )
            else:
                mc_transform_simd_parallel(
                    state,
                    gate_tr.controls,
                    gate_tr.target,
                    gate_tr.gate_info.gate,
                    ctx,
                )
        elif tr.isa[SwapTransformation]():
            var swap_tr = tr[SwapTransformation].copy()
            if swap_tr.a < 0 or swap_tr.a >= state.size():
                raise Error("Target out of bounds: " + String(swap_tr.a))
            if swap_tr.b < 0 or swap_tr.b >= state.size():
                raise Error("Target out of bounds: " + String(swap_tr.b))
            if swap_tr.a != swap_tr.b:
                mc_transform_simd_parallel(
                    state,
                    List[Int](swap_tr.a),
                    swap_tr.b,
                    X,
                )
                mc_transform_simd_parallel(
                    state,
                    List[Int](swap_tr.b),
                    swap_tr.a,
                    X,
                )
                mc_transform_simd_parallel(
                    state,
                    List[Int](swap_tr.a),
                    swap_tr.b,
                    X,
                )
        elif tr.isa[QubitReversalTransformation]():
            var qrev_tr = tr[QubitReversalTransformation].copy()
            var targets = qrev_tr.targets.copy()
            if len(targets) == 0:
                var tmp = state.size()
                var nbits = 0
                while tmp > 1:
                    tmp //= 2
                    nbits += 1
                targets = List[Int](capacity=nbits)
                for i in range(nbits):
                    targets.append(i)
            sort_targets(targets)
            for t in targets:
                if t < 0 or t >= state.size():
                    raise Error("Target out of bounds: " + String(t))
            var half = len(targets) // 2
            for i in range(half):
                var a = targets[i]
                var b = targets[len(targets) - 1 - i]
                if a == b:
                    continue
                mc_transform_simd_parallel(state, List[Int](a), b, X)
                mc_transform_simd_parallel(state, List[Int](b), a, X)
                mc_transform_simd_parallel(state, List[Int](a), b, X)
        elif tr.isa[UnitaryTransformation]():
            var unitary_tr = tr[UnitaryTransformation].copy()
            apply_unitary(state, unitary_tr.u, unitary_tr.target, unitary_tr.m)
        elif tr.isa[ControlledUnitaryTransformation]():
            var c_unitary_tr = tr[ControlledUnitaryTransformation].copy()
            apply_controlled_unitary(
                state,
                c_unitary_tr.u,
                c_unitary_tr.control,
                c_unitary_tr.target,
                c_unitary_tr.m,
            )
        elif tr.isa[MeasurementTransform]():
            var meas_tr = tr[MeasurementTransform].copy()
            meas_tr.apply(
                state,
                meas_tr.targets,
                meas_tr.values,
                meas_tr.seed,
            )
        else:
            var cl_tr = tr[ClassicalTransform].copy()
            cl_tr.apply(state, cl_tr.targets)


fn execute_grid(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    var n = 0
    var tmp = state.size()
    while tmp > 1:
        tmp //= 2
        n += 1
    var col_bits = max(n - 3, 3)
    var num_rows = 1 << (n - col_bits)
    var row_size = 1 << col_bits

    for tr in circuit.transformations:
        if tr.isa[FusedPairTransformation]():
            var pair_tr = tr[FusedPairTransformation].copy()
            if not apply_fused_pair_grid(
                state,
                pair_tr,
                num_rows,
                row_size,
                col_bits,
                ctx.grid_use_parallel,
            ):
                apply_fused_pair(state, pair_tr)
        elif tr.isa[GateTransformation]():
            var gate_tr = tr[GateTransformation].copy()
            if gate_tr.target < 0 or gate_tr.target >= state.size():
                raise Error("Target out of bounds: " + String(gate_tr.target))
            for c in gate_tr.controls:
                if c < 0 or c >= state.size():
                    raise Error("Control out of bounds: " + String(c))
            if gate_tr.kind == ControlKind.NO_CONTROL:
                transform_grid[8](
                    state,
                    col_bits,
                    gate_tr.target,
                    gate_tr.gate_info,
                    ctx,
                )
            elif gate_tr.kind == ControlKind.SINGLE_CONTROL:
                if gate_tr.gate_info.kind == GateKind.P:
                    c_transform_p_grid[8](
                        state,
                        col_bits,
                        gate_tr.controls[0],
                        gate_tr.target,
                        Float64(gate_tr.gate_info.arg.value()),
                        ctx,
                    )
                else:
                    mc_transform_simd_parallel(
                        state,
                        gate_tr.controls,
                        gate_tr.target,
                        gate_tr.gate_info.gate,
                    )
            else:
                mc_transform_simd_parallel(
                    state,
                    gate_tr.controls,
                    gate_tr.target,
                    gate_tr.gate_info.gate,
                )
        elif tr.isa[SwapTransformation]():
            var swap_tr = tr[SwapTransformation].copy()
            if swap_tr.a < 0 or swap_tr.a >= state.size():
                raise Error("Target out of bounds: " + String(swap_tr.a))
            if swap_tr.b < 0 or swap_tr.b >= state.size():
                raise Error("Target out of bounds: " + String(swap_tr.b))
            if swap_tr.a != swap_tr.b:
                mc_transform_simd_parallel(
                    state,
                    List[Int](swap_tr.a),
                    swap_tr.b,
                    X,
                )
                mc_transform_simd_parallel(
                    state,
                    List[Int](swap_tr.b),
                    swap_tr.a,
                    X,
                )
                mc_transform_simd_parallel(
                    state,
                    List[Int](swap_tr.a),
                    swap_tr.b,
                    X,
                )
        elif tr.isa[QubitReversalTransformation]():
            var qrev_tr = tr[QubitReversalTransformation].copy()
            var targets = qrev_tr.targets.copy()
            if len(targets) == 0:
                var tmp = state.size()
                var nbits = 0
                while tmp > 1:
                    tmp //= 2
                    nbits += 1
                targets = List[Int](capacity=nbits)
                for i in range(nbits):
                    targets.append(i)
            sort_targets(targets)
            for t in targets:
                if t < 0 or t >= state.size():
                    raise Error("Target out of bounds: " + String(t))
            var half = len(targets) // 2
            for i in range(half):
                var a = targets[i]
                var b = targets[len(targets) - 1 - i]
                if a == b:
                    continue
                mc_transform_simd_parallel(state, List[Int](a), b, X)
                mc_transform_simd_parallel(state, List[Int](b), a, X)
                mc_transform_simd_parallel(state, List[Int](a), b, X)
        elif tr.isa[UnitaryTransformation]():
            var unitary_tr = tr[UnitaryTransformation].copy()
            apply_unitary(state, unitary_tr.u, unitary_tr.target, unitary_tr.m)
        elif tr.isa[ControlledUnitaryTransformation]():
            var c_unitary_tr = tr[ControlledUnitaryTransformation].copy()
            apply_controlled_unitary(
                state,
                c_unitary_tr.u,
                c_unitary_tr.control,
                c_unitary_tr.target,
                c_unitary_tr.m,
            )
        elif tr.isa[MeasurementTransform]():
            var meas_tr = tr[MeasurementTransform].copy()
            meas_tr.apply(
                state,
                meas_tr.targets,
                meas_tr.values,
                meas_tr.seed,
            )
        else:
            var cl_tr = tr[ClassicalTransform].copy()
            cl_tr.apply(state, cl_tr.targets)


fn apply_fused_pair_grid(
    mut state: QuantumState,
    pair: FusedPairTransformation,
    num_rows: Int,
    row_size: Int,
    col_bits: Int,
    use_parallel: Bool,
) raises -> Bool:
    var t0 = pair.first.copy()
    var t1 = pair.second.copy()

    if (
        t0.kind == ControlKind.MULTI_CONTROL
        or t1.kind == ControlKind.MULTI_CONTROL
    ):
        return False

    # Check if this is a cross-row HP fusion (both targets >= col_bits)
    var t0_is_cross_row = t0.target >= col_bits
    var t1_is_cross_row = t1.target >= col_bits

    # Handle cross-row HP fusion with tiled kernel
    if t0_is_cross_row and t1_is_cross_row:
        if (
            t0.kind != ControlKind.NO_CONTROL
            or t1.kind != ControlKind.NO_CONTROL
        ):
            return False  # Only uncontrolled cross-row fusion for now

        var t0_is_h = t0.gate_info.kind == GateKind.H
        var t0_is_p = t0.gate_info.kind == GateKind.P
        var t1_is_h = t1.gate_info.kind == GateKind.H
        var t1_is_p = t1.gate_info.kind == GateKind.P

        if (t0_is_h and t1_is_p) or (t0_is_p and t1_is_h):
            var target_h: Int
            var target_p: Int
            var theta: FloatType
            if t0_is_h:
                target_h = t0.target - col_bits
                target_p = t1.target - col_bits
                theta = Float64(t1.gate_info.arg.value())
            else:
                target_h = t1.target - col_bits
                target_p = t0.target - col_bits
                theta = Float64(t0.gate_info.arg.value())

            var re_ptr = state.re.unsafe_ptr()
            var im_ptr = state.im.unsafe_ptr()
            alias chunk_size = simd_width
            var tile_size = max(L2_TILE_COLS, chunk_size)
            tile_size = (tile_size // chunk_size) * chunk_size
            var num_tiles = (row_size + tile_size - 1) // tile_size

            @__copy_capture(
                re_ptr, im_ptr, tile_size, target_h, target_p, theta
            )
            @parameter
            fn process_tile_hp(tile_idx: Int):
                var col_start = tile_idx * tile_size
                var col_end = min(col_start + tile_size, row_size)
                col_end = (col_end // chunk_size) * chunk_size
                if col_end <= col_start:
                    return
                transform_column_fused_hp_simd_tiled[chunk_size](
                    re_ptr,
                    im_ptr,
                    num_rows,
                    row_size,
                    col_start,
                    col_end,
                    target_h,
                    target_p,
                    theta,
                )

            if use_parallel:
                parallelize[process_tile_hp](num_tiles)
            else:
                for tile_idx in range(num_tiles):
                    process_tile_hp(tile_idx)
            return True

        return False  # Cross-row but not HP fusion

    # Row-local operations: both targets must be < col_bits
    if t0_is_cross_row or t1_is_cross_row:
        return False
    if t0.kind == ControlKind.SINGLE_CONTROL and t0.controls[0] >= col_bits:
        return False
    if t1.kind == ControlKind.SINGLE_CONTROL and t1.controls[0] >= col_bits:
        return False

    if (
        t0.kind == ControlKind.SINGLE_CONTROL
        and t1.kind == ControlKind.SINGLE_CONTROL
        and t0.gate_info.kind == GateKind.P
        and t1.gate_info.kind == GateKind.P
        and t0.controls[0] == t1.controls[0]
    ):
        var theta0 = Float64(t0.gate_info.arg.value())
        var theta1 = Float64(t1.gate_info.arg.value())

        @parameter
        fn process_row_shared_cp(row: Int):
            transform_row_fused_shared_c_pp_simd[simd_width](
                state,
                row,
                row_size,
                t0.controls[0],
                t0.target,
                t1.target,
                theta0,
                theta1,
            )

        if use_parallel:
            parallelize[process_row_shared_cp](num_rows)
        else:
            for row in range(num_rows):
                process_row_shared_cp(row)
        return True

    if t0.kind != ControlKind.NO_CONTROL or t1.kind != ControlKind.NO_CONTROL:
        return False

    var t0_is_h = t0.gate_info.kind == GateKind.H
    var t0_is_p = t0.gate_info.kind == GateKind.P
    var t1_is_h = t1.gate_info.kind == GateKind.H
    var t1_is_p = t1.gate_info.kind == GateKind.P

    if t0_is_h and t1_is_h:

        @parameter
        fn process_row_hh(row: Int):
            transform_row_fused_hh_simd[simd_width](
                state, row, row_size, t0.target, t1.target
            )

        if use_parallel:
            parallelize[process_row_hh](num_rows)
        else:
            for row in range(num_rows):
                process_row_hh(row)
        return True

    if t0_is_p and t1_is_p:
        var theta0 = Float64(t0.gate_info.arg.value())
        var theta1 = Float64(t1.gate_info.arg.value())

        @parameter
        fn process_row_pp(row: Int):
            transform_row_fused_pp_simd[simd_width](
                state,
                row,
                row_size,
                t0.target,
                t1.target,
                theta0,
                theta1,
            )

        if use_parallel:
            parallelize[process_row_pp](num_rows)
        else:
            for row in range(num_rows):
                process_row_pp(row)
        return True

    if (t0_is_h and t1_is_p) or (t0_is_p and t1_is_h):
        # var th = t0.target
        # var tp = t1.target
        var theta: FloatType
        if t0_is_h:
            th = t0.target
            tp = t1.target
            theta = Float64(t1.gate_info.arg.value())
        else:
            th = t1.target
            tp = t0.target
            theta = Float64(t0.gate_info.arg.value())

        @parameter
        fn process_row_hp(row: Int):
            transform_row_fused_hp_simd[simd_width](
                state, row, row_size, th, tp, theta
            )

        if use_parallel:
            parallelize[process_row_hp](num_rows)
        else:
            for row in range(num_rows):
                process_row_hp(row)
        return True

    return False


struct GridPreparedTransformation(Copyable, Movable):
    var is_controlled: Bool
    var target: Int
    var control: Int
    var gate: Gate
    var gate_kind: Int
    var gate_arg: FloatType
    var theta: Float64
    var is_h_gate: Bool
    var is_x_gate: Bool
    var is_z_gate: Bool
    var is_p_gate: Bool
    var row_control_bit: Int

    fn __init__(out self, gt: GateTransformation, col_bits: Int):
        self.is_controlled = gt.kind == ControlKind.SINGLE_CONTROL
        self.target = gt.target
        self.control = -1
        if self.is_controlled:
            self.control = gt.controls[0]
        self.gate = gt.gate_info.gate
        self.gate_kind = gt.gate_info.kind
        self.gate_arg = FloatType(0)
        if gt.gate_info.arg:
            self.gate_arg = gt.gate_info.arg.value()
        self.theta = 0.0
        self.is_h_gate = gt.gate_info.kind == GateKind.H
        self.is_x_gate = gt.gate_info.kind == GateKind.X
        self.is_z_gate = gt.gate_info.kind == GateKind.Z
        self.is_p_gate = gt.gate_info.kind == GateKind.P
        if self.is_p_gate and gt.gate_info.arg:
            self.theta = Float64(gt.gate_info.arg.value())
        self.row_control_bit = -1
        if self.is_controlled and self.control >= col_bits:
            self.row_control_bit = self.control


struct GridFusionGroup(Copyable, Movable):
    var transformations: List[QuantumTransformation]
    var is_row_local: Bool
    var is_fused: Bool

    fn __init__(out self, is_row_local: Bool = False, is_fused: Bool = False):
        self.transformations = List[QuantumTransformation]()
        self.is_row_local = is_row_local
        self.is_fused = is_fused

    fn __copyinit__(out self, existing: Self):
        self.transformations = List[QuantumTransformation]()
        for i in range(len(existing.transformations)):
            self.transformations.append(existing.transformations[i])
        self.is_row_local = existing.is_row_local
        self.is_fused = existing.is_fused

    fn __moveinit__(out self, deinit existing: Self):
        self.transformations = existing.transformations^
        self.is_row_local = existing.is_row_local
        self.is_fused = existing.is_fused


fn is_row_local_gate(tr: QuantumTransformation, col_bits: Int) -> Bool:
    if tr.isa[GateTransformation]():
        var gt = tr[GateTransformation].copy()
        if gt.kind == ControlKind.MULTI_CONTROL:
            return False
        return gt.target < col_bits
    return False


fn analyze_for_grid_fusion(
    transformations: List[QuantumTransformation],
    col_bits: Int,
) -> List[GridFusionGroup]:
    var groups = List[GridFusionGroup]()
    var i = 0
    var n = len(transformations)

    while i < n:
        var tr = transformations[i]
        if is_row_local_gate(tr, col_bits):
            var g = GridFusionGroup(is_row_local=True, is_fused=False)
            g.transformations.append(tr)
            i += 1
            while i < n and is_row_local_gate(transformations[i], col_bits):
                var next_tr = transformations[i]
                g.transformations.append(next_tr)
                if len(g.transformations) >= 2:
                    g.is_fused = True
                i += 1
            groups.append(g^)
        else:
            var g = GridFusionGroup(is_row_local=False)
            g.transformations.append(tr)
            i += 1
            while i < n and not is_row_local_gate(transformations[i], col_bits):
                g.transformations.append(transformations[i])
                i += 1
            groups.append(g^)

    return groups^


fn execute_row_local_group[
    simd_width: Int
](
    mut state: QuantumState,
    transformations: List[QuantumTransformation],
    num_rows: Int,
    row_size: Int,
    col_bits: Int,
    is_fused: Bool,
    ctx: ExecContext,
) raises:
    var prepared = List[GridPreparedTransformation]()
    for tr in transformations:
        if tr.isa[GateTransformation]():
            prepared.append(
                GridPreparedTransformation(
                    tr[GateTransformation].copy(), col_bits
                )
            )
    var prepared_ptr = prepared.unsafe_ptr()
    var num_prepared = len(prepared)

    @parameter
    fn process_row(row: Int):
        var i = 0
        while i < num_prepared:
            var p = prepared_ptr + i

            if p[].row_control_bit >= 0:
                var row_bit_pos = p[].row_control_bit - col_bits
                if not ((row >> row_bit_pos) & 1):
                    i += 1
                    continue

            if is_fused and i + 1 < num_prepared:
                var fused = False
                for j in range(i + 1, num_prepared):
                    var p2 = prepared_ptr + j
                    var p2_active = True
                    if p2[].row_control_bit >= 0:
                        var row_bit_pos = p2[].row_control_bit - col_bits
                        if not ((row >> row_bit_pos) & 1):
                            p2_active = False

                    if p2_active:
                        var can_fuse = False
                        var treat_uncontrolled = False
                        if (
                            not p[].is_controlled or p[].row_control_bit >= 0
                        ) and (
                            not p2[].is_controlled or p2[].row_control_bit >= 0
                        ):
                            can_fuse = True
                            treat_uncontrolled = True
                        elif (
                            p[].is_controlled
                            and p2[].is_controlled
                            and p[].control == p2[].control
                        ):
                            can_fuse = True

                        if can_fuse:
                            if p[].target != p2[].target:
                                if (
                                    treat_uncontrolled
                                    and p[].is_h_gate
                                    and p2[].is_h_gate
                                ):
                                    transform_row_fused_hh_simd[simd_width](
                                        state,
                                        row,
                                        row_size,
                                        p[].target,
                                        p2[].target,
                                    )
                                    i = j + 1
                                    fused = True
                                elif (
                                    treat_uncontrolled
                                    and p[].is_h_gate
                                    and p2[].is_p_gate
                                ):
                                    transform_row_fused_hp_simd[simd_width](
                                        state,
                                        row,
                                        row_size,
                                        p[].target,
                                        p2[].target,
                                        p2[].theta,
                                    )
                                    i = j + 1
                                    fused = True
                                elif (
                                    treat_uncontrolled
                                    and p[].is_p_gate
                                    and p2[].is_h_gate
                                ):
                                    transform_row_fused_hp_simd[simd_width](
                                        state,
                                        row,
                                        row_size,
                                        p2[].target,
                                        p[].target,
                                        p[].theta,
                                    )
                                    i = j + 1
                                    fused = True
                                elif p[].is_p_gate and p2[].is_p_gate:
                                    if (
                                        p[].is_controlled
                                        and p2[].is_controlled
                                        and p[].control == p2[].control
                                    ):
                                        transform_row_fused_shared_c_pp_simd[
                                            simd_width
                                        ](
                                            state,
                                            row,
                                            row_size,
                                            p[].control,
                                            p[].target,
                                            p2[].target,
                                            p[].theta,
                                            p2[].theta,
                                        )
                                        i = j + 1
                                        fused = True
                                    elif treat_uncontrolled:
                                        transform_row_fused_pp_simd[simd_width](
                                            state,
                                            row,
                                            row_size,
                                            p[].target,
                                            p2[].target,
                                            p[].theta,
                                            p2[].theta,
                                        )
                                        i = j + 1
                                        fused = True
                            else:
                                if (
                                    treat_uncontrolled
                                    and p[].is_h_gate
                                    and p2[].is_p_gate
                                ):
                                    transform_row_fused_st_hp_simd[simd_width](
                                        state,
                                        row,
                                        row_size,
                                        p[].target,
                                        p2[].theta,
                                    )
                                    i = j + 1
                                    fused = True
                        break
                    else:
                        continue

                if fused:
                    continue

            if p[].is_controlled and p[].row_control_bit < 0:
                if p[].is_h_gate:
                    c_transform_row_h_simd[simd_width](
                        state, row, row_size, p[].control, p[].target
                    )
                elif p[].is_p_gate:
                    c_transform_row_p_simd[simd_width](
                        state,
                        row,
                        row_size,
                        p[].control,
                        p[].target,
                        p[].theta,
                    )
                else:
                    c_transform_simd(
                        state,
                        p[].control,
                        p[].target,
                        p[].gate,
                        p[].gate_kind,
                        p[].gate_arg,
                        ctx,
                    )
            else:
                if p[].is_h_gate:
                    transform_row_h_simd[simd_width](
                        state, row, row_size, 1 << p[].target
                    )
                elif p[].is_x_gate:
                    transform_row_x_simd[simd_width](
                        state, row, row_size, p[].target
                    )
                elif p[].is_z_gate:
                    transform_row_z_simd[simd_width](
                        state, row, row_size, p[].target
                    )
                elif p[].is_p_gate:
                    transform_row_p_simd[simd_width](
                        state, row, row_size, p[].target, p[].theta
                    )
                else:
                    transform_row_simd[simd_width](
                        state, row, row_size, p[].target, p[].gate
                    )
            i += 1

    if ctx.grid_use_parallel:
        parallelize[process_row](num_rows)
    else:
        for row in range(num_rows):
            process_row(row)


fn execute_grid_fused(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    var n = 0
    var tmp = state.size()
    while tmp > 1:
        tmp //= 2
        n += 1
    var col_bits = max(n - 3, 3)
    var num_rows = 1 << (n - col_bits)
    var row_size = 1 << col_bits

    var groups = analyze_for_grid_fusion(circuit.transformations, col_bits)

    var groups_ptr = groups.unsafe_ptr()
    for i in range(len(groups)):
        var g_ref = groups_ptr + i
        if g_ref[].is_row_local:
            execute_row_local_group[simd_width](
                state,
                g_ref[].transformations,
                num_rows,
                row_size,
                col_bits,
                g_ref[].is_fused,
                ctx,
            )
        else:
            var sub_circuit = QuantumCircuit(circuit.num_qubits)
            for tr in g_ref[].transformations:
                sub_circuit.transformations.append(tr.copy())
            execute_grid(state, sub_circuit, ctx)


fn test_main() raises:
    from butterfly.algos.value_encoding_circuit import encode_value_circuit

    var n = 3
    var v = 4.7
    var circuit = encode_value_circuit(n, v)

    from butterfly.utils.visualization import print_state

    var state = QuantumState(n)
    execute_scalar(state, circuit)
    print_state(state)
    from butterfly.utils.circuit_print import print_circuit_ascii

    print_circuit_ascii(circuit)
