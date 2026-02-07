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
    ClassicalTransformation,
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
    transform_column_fused_hp_simd_tiled,
    transform_column_fused_hh_simd_tiled,
    transform_column_fused_pp_simd_tiled,
    L2_TILE_COLS,
)
from butterfly.core.transformations_grid_fusion import (
    analyze_for_grid_fusion,
    execute_row_local_group,
)
from butterfly.core.transformations_unitary import (
    apply_unitary,
    apply_controlled_unitary,
)
from butterfly.core.transformations_fusion import apply_fused_pair
from butterfly.core.fused_kernels_sparse_row import (
    transform_row_fused_hh_simd,
    transform_row_fused_hp_simd,
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


fn num_qubits_from_state(state: QuantumState) raises -> Int:
    var size = state.size()
    if size <= 0 or (size & (size - 1)) != 0:
        raise Error("State size must be a power of two")
    var n = 0
    var tmp = size
    while tmp > 1:
        tmp //= 2
        n += 1
    return n


fn validate_qubit_index(idx: Int, nbits: Int, label: String) raises:
    if idx < 0 or idx >= nbits:
        raise Error(label + " out of bounds: " + String(idx))


fn validate_gate_transformation(gt: GateTransformation, nbits: Int) raises:
    validate_qubit_index(gt.target, nbits, "Target")
    var controls_len = len(gt.controls)
    if gt.kind == ControlKind.NO_CONTROL:
        if controls_len != 0:
            raise Error("Gate control kind mismatch: expected 0 controls")
    elif gt.kind == ControlKind.SINGLE_CONTROL:
        if controls_len != 1:
            raise Error("Gate control kind mismatch: expected 1 control")
    else:
        if controls_len < 2:
            raise Error("Gate control kind mismatch: expected >=2 controls")
    for c in gt.controls:
        validate_qubit_index(c, nbits, "Control")


fn validate_unitary(target: Int, m: Int, nbits: Int) raises:
    if m <= 0:
        return
    if target < 0 or target + m > nbits:
        raise Error("Target range out of bounds")


fn validate_controlled_unitary(
    control: Int, target: Int, m: Int, nbits: Int
) raises:
    validate_unitary(target, m, nbits)
    validate_qubit_index(control, nbits, "Control")
    if m <= 0:
        return
    if control >= target and control < target + m:
        raise Error("Control overlaps target range")


fn validate_circuit(state: QuantumState, circuit: QuantumCircuit) raises:
    var nbits = num_qubits_from_state(state)
    if circuit.num_qubits != nbits:
        raise Error(
            "Circuit qubits mismatch: circuit has "
            + String(circuit.num_qubits)
            + ", state has "
            + String(nbits)
        )

    for tr in circuit.transformations:
        if tr.isa[FusedPairTransformation]():
            var pair_tr = tr[FusedPairTransformation].copy()
            validate_gate_transformation(pair_tr.first, nbits)
            validate_gate_transformation(pair_tr.second, nbits)
        elif tr.isa[GateTransformation]():
            var gate_tr = tr[GateTransformation].copy()
            validate_gate_transformation(gate_tr, nbits)
        elif tr.isa[ClassicalTransformation[QuantumState]]():
            var classical_tr = tr[ClassicalTransformation[QuantumState]].copy()
            for t in classical_tr.targets:
                validate_qubit_index(t, nbits, "Target")
        elif tr.isa[SwapTransformation]():
            var swap_tr = tr[SwapTransformation].copy()
            validate_qubit_index(swap_tr.a, nbits, "Target")
            validate_qubit_index(swap_tr.b, nbits, "Target")
        elif tr.isa[QubitReversalTransformation]():
            var qrev_tr = tr[QubitReversalTransformation].copy()
            for t in qrev_tr.targets:
                validate_qubit_index(t, nbits, "Target")
        elif tr.isa[UnitaryTransformation]():
            var unitary_tr = tr[UnitaryTransformation].copy()
            validate_unitary(unitary_tr.target, unitary_tr.m, nbits)
        elif tr.isa[ControlledUnitaryTransformation]():
            var c_unitary_tr = tr[ControlledUnitaryTransformation].copy()
            validate_controlled_unitary(
                c_unitary_tr.control,
                c_unitary_tr.target,
                c_unitary_tr.m,
                nbits,
            )
        elif tr.isa[MeasurementTransform]():
            var meas_tr = tr[MeasurementTransform].copy()
            if len(meas_tr.values) != len(meas_tr.targets):
                raise Error("Measurement values length mismatch")
            for t in meas_tr.targets:
                validate_qubit_index(t, nbits, "Target")
        else:
            raise Error("Unknown transformation type")


struct GridLayout(Copyable, Movable):
    var col_bits: Int
    var num_rows: Int
    var row_size: Int

    fn __init__(
        out self,
        col_bits: Int = -1,
        num_rows: Int = 0,
        row_size: Int = 0,
    ):
        self.col_bits = col_bits
        self.num_rows = num_rows
        self.row_size = row_size


alias GateHandler = fn (
    mut QuantumState, GateTransformation, ExecContext, GridLayout
) raises
alias FusedPairHandler = fn (
    mut QuantumState, FusedPairTransformation, ExecContext, GridLayout
) raises
alias ControlledXHandler = fn (mut QuantumState, Int, Int, ExecContext) raises


fn compute_grid_layout(
    state: QuantumState, ctx: ExecContext
) raises -> GridLayout:
    var n = num_qubits_from_state(state)
    var min_bits = ctx.grid_col_bits_min
    if min_bits < 0:
        min_bits = 0
    var slack = ctx.grid_col_bits_slack
    if slack < 0:
        slack = 0
    var col_bits = max(n - slack, min_bits)
    if col_bits > n:
        col_bits = n
    var num_rows = 1 << (n - col_bits)
    var row_size = 1 << col_bits
    return GridLayout(col_bits, num_rows, row_size)


fn apply_swap_with_cx(
    mut state: QuantumState,
    swap_tr: SwapTransformation,
    ctx: ExecContext,
    cx: ControlledXHandler,
) raises:
    if swap_tr.a == swap_tr.b:
        return
    cx(state, swap_tr.a, swap_tr.b, ctx)
    cx(state, swap_tr.b, swap_tr.a, ctx)
    cx(state, swap_tr.a, swap_tr.b, ctx)


fn apply_qubit_reversal_with_cx(
    mut state: QuantumState,
    targets: List[Int],
    ctx: ExecContext,
    cx: ControlledXHandler,
) raises:
    var local_targets = targets.copy()
    if len(local_targets) == 0:
        var nbits = num_qubits_from_state(state)
        local_targets = List[Int](capacity=nbits)
        for i in range(nbits):
            local_targets.append(i)
    sort_targets(local_targets)
    var half = len(local_targets) // 2
    for i in range(half):
        var a = local_targets[i]
        var b = local_targets[len(local_targets) - 1 - i]
        if a == b:
            continue
        cx(state, a, b, ctx)
        cx(state, b, a, ctx)
        cx(state, a, b, ctx)


fn execute_with_handlers(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext,
    layout: GridLayout,
    on_gate: GateHandler,
    on_fused_pair: FusedPairHandler,
    on_cx: ControlledXHandler,
) raises:
    if ctx.validate_circuit:
        validate_circuit(state, circuit)
    for tr in circuit.transformations:
        if tr.isa[FusedPairTransformation]():
            on_fused_pair(
                state, tr[FusedPairTransformation].copy(), ctx, layout
            )
        elif tr.isa[GateTransformation]():
            on_gate(state, tr[GateTransformation].copy(), ctx, layout)
        elif tr.isa[ClassicalTransform]():
            var classical_tr = tr[ClassicalTransform].copy()
            classical_tr.apply(state, classical_tr.targets)
        elif tr.isa[SwapTransformation]():
            apply_swap_with_cx(
                state, tr[SwapTransformation].copy(), ctx, on_cx
            )
        elif tr.isa[QubitReversalTransformation]():
            var qrev_tr = tr[QubitReversalTransformation].copy()
            apply_qubit_reversal_with_cx(
                state, qrev_tr.targets, ctx, on_cx
            )
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
            raise Error("Unknown transformation type")


fn handle_fused_pair_default(
    mut state: QuantumState,
    pair_tr: FusedPairTransformation,
    ctx: ExecContext,
    layout: GridLayout,
) raises:
    apply_fused_pair(state, pair_tr)


fn handle_gate_scalar(
    mut state: QuantumState,
    gate_tr: GateTransformation,
    ctx: ExecContext,
    layout: GridLayout,
) raises:
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


fn handle_cx_scalar(
    mut state: QuantumState, control: Int, target: Int, ctx: ExecContext
) raises:
    c_transform_scalar(state, control, target, X, GateKind.X, 0, ctx)


fn handle_gate_simd(
    mut state: QuantumState,
    gate_tr: GateTransformation,
    ctx: ExecContext,
    layout: GridLayout,
) raises:
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


fn handle_cx_simd(
    mut state: QuantumState, control: Int, target: Int, ctx: ExecContext
) raises:
    mc_transform_simd(state, List[Int](control), target, X)


fn handle_gate_simd_parallel(
    mut state: QuantumState,
    gate_tr: GateTransformation,
    ctx: ExecContext,
    layout: GridLayout,
) raises:
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
                FloatType(gate_arg),
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
                FloatType(gate_arg),
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
                FloatType(gate_arg),
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
                FloatType(gate_arg),
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


fn handle_cx_simd_parallel(
    mut state: QuantumState, control: Int, target: Int, ctx: ExecContext
) raises:
    mc_transform_simd_parallel(state, List[Int](control), target, X, ctx)


fn handle_gate_grid(
    mut state: QuantumState,
    gate_tr: GateTransformation,
    ctx: ExecContext,
    layout: GridLayout,
) raises:
    if gate_tr.kind == ControlKind.NO_CONTROL:
        transform_grid[8](
            state,
            layout.col_bits,
            gate_tr.target,
            gate_tr.gate_info,
            ctx,
        )
    elif gate_tr.kind == ControlKind.SINGLE_CONTROL:
        if gate_tr.gate_info.kind == GateKind.P:
            c_transform_p_grid[8](
                state,
                layout.col_bits,
                gate_tr.controls[0],
                gate_tr.target,
                FloatType(gate_tr.gate_info.arg.value()),
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


fn handle_fused_pair_grid(
    mut state: QuantumState,
    pair_tr: FusedPairTransformation,
    ctx: ExecContext,
    layout: GridLayout,
) raises:
    if not apply_fused_pair_grid(
        state,
        pair_tr,
        layout.num_rows,
        layout.row_size,
        layout.col_bits,
        ctx.grid_use_parallel,
    ):
        apply_fused_pair(state, pair_tr)


fn handle_cx_grid(
    mut state: QuantumState, control: Int, target: Int, ctx: ExecContext
) raises:
    mc_transform_simd_parallel(state, List[Int](control), target, X)


fn execute_scalar(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    execute_with_handlers(
        state,
        circuit,
        ctx,
        GridLayout(),
        handle_gate_scalar,
        handle_fused_pair_default,
        handle_cx_scalar,
    )


fn execute_scalar_parallel(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    threads: Int = 4,
) raises:
    var ctx = ExecContext()
    ctx.threads = threads
    execute_scalar(state, circuit, ctx)


alias ExecutorFn = fn (
    mut QuantumState, QuantumCircuit, ExecContext
) raises


struct ExecutorEntry(Copyable, Movable):
    var strategy: ExecutionStrategy
    var run: ExecutorFn

    fn __init__(out self, strategy: ExecutionStrategy, run: ExecutorFn):
        self.strategy = strategy
        self.run = run


fn exec_scalar_parallel_ctx(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext,
) raises:
    var local = ctx.copy()
    if local.threads <= 0:
        local.threads = 4
    execute_scalar(state, circuit, local)


fn exec_grid_ctx(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext,
    use_parallel: Bool,
) raises:
    var local = ctx.copy()
    local.grid_use_parallel = use_parallel
    execute_grid(state, circuit, local)


fn exec_grid_fused_ctx(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext,
    use_parallel: Bool,
) raises:
    var local = ctx.copy()
    local.grid_use_parallel = use_parallel
    execute_grid_fused(state, circuit, local)


fn exec_grid_serial(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext,
) raises:
    exec_grid_ctx(state, circuit, ctx, False)


fn exec_grid_parallel(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext,
) raises:
    exec_grid_ctx(state, circuit, ctx, True)


fn exec_grid_fused_serial(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext,
) raises:
    exec_grid_fused_ctx(state, circuit, ctx, False)


fn exec_grid_fused_parallel(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext,
) raises:
    exec_grid_fused_ctx(state, circuit, ctx, True)


fn executor_registry() -> List[ExecutorEntry]:
    var entries = List[ExecutorEntry]()
    entries.append(ExecutorEntry(ExecutionStrategy.SCALAR, execute_scalar))
    entries.append(
        ExecutorEntry(ExecutionStrategy.SCALAR_PARALLEL, exec_scalar_parallel_ctx)
    )
    entries.append(ExecutorEntry(ExecutionStrategy.SIMD, execute_simd))
    entries.append(
        ExecutorEntry(ExecutionStrategy.SIMD_PARALLEL, execute_simd_parallel)
    )
    entries.append(ExecutorEntry(ExecutionStrategy.GRID, exec_grid_serial))
    entries.append(
        ExecutorEntry(ExecutionStrategy.GRID_PARALLEL, exec_grid_parallel)
    )
    entries.append(
        ExecutorEntry(ExecutionStrategy.GRID_FUSED, exec_grid_fused_serial)
    )
    entries.append(
        ExecutorEntry(
            ExecutionStrategy.GRID_PARALLEL_FUSED, exec_grid_fused_parallel
        )
    )
    return entries^


fn get_executor(strategy: ExecutionStrategy) raises -> ExecutorFn:
    var entries = executor_registry()
    for i in range(len(entries)):
        if entries[i].strategy == strategy:
            return entries[i].run
    raise Error("Unknown execution strategy: " + String(strategy.value))


fn execute(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    var exec_ctx = ctx.copy()
    var runner = get_executor(exec_ctx.execution_strategy)
    runner(state, circuit, exec_ctx)


fn execute_simd(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    execute_with_handlers(
        state,
        circuit,
        ctx,
        GridLayout(),
        handle_gate_simd,
        handle_fused_pair_default,
        handle_cx_simd,
    )


fn execute_simd_parallel(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    execute_with_handlers(
        state,
        circuit,
        ctx,
        GridLayout(),
        handle_gate_simd_parallel,
        handle_fused_pair_default,
        handle_cx_simd_parallel,
    )


fn execute_grid(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    var layout = compute_grid_layout(state, ctx)
    execute_with_handlers(
        state,
        circuit,
        ctx,
        layout,
        handle_gate_grid,
        handle_fused_pair_grid,
        handle_cx_grid,
    )


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
                theta = FloatType(t1.gate_info.arg.value())
            else:
                target_h = t1.target - col_bits
                target_p = t0.target - col_bits
                theta = FloatType(t0.gate_info.arg.value())

            var re_ptr = state.re_ptr()
            var im_ptr = state.im_ptr()
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

        # Cross-row HH fusion
        if t0_is_h and t1_is_h:
            var target_1 = t0.target - col_bits
            var target_2 = t1.target - col_bits

            var re_ptr = state.re_ptr()
            var im_ptr = state.im_ptr()
            alias chunk_size = simd_width
            var tile_size = max(L2_TILE_COLS, chunk_size)
            tile_size = (tile_size // chunk_size) * chunk_size
            var num_tiles = (row_size + tile_size - 1) // tile_size

            @__copy_capture(re_ptr, im_ptr, tile_size, target_1, target_2)
            @parameter
            fn process_tile_hh(tile_idx: Int):
                var col_start = tile_idx * tile_size
                var col_end = min(col_start + tile_size, row_size)
                col_end = (col_end // chunk_size) * chunk_size
                if col_end <= col_start:
                    return
                transform_column_fused_hh_simd_tiled[chunk_size](
                    re_ptr,
                    im_ptr,
                    num_rows,
                    row_size,
                    col_start,
                    col_end,
                    target_1,
                    target_2,
                )

            if use_parallel:
                parallelize[process_tile_hh](num_tiles)
            else:
                for tile_idx in range(num_tiles):
                    process_tile_hh(tile_idx)
            return True

        # Cross-row PP fusion
        if t0_is_p and t1_is_p:
            var target_1 = t0.target - col_bits
            var target_2 = t1.target - col_bits
            var theta_1 = FloatType(t0.gate_info.arg.value())
            var theta_2 = FloatType(t1.gate_info.arg.value())

            var re_ptr = state.re_ptr()
            var im_ptr = state.im_ptr()
            alias chunk_size = simd_width
            var tile_size = max(L2_TILE_COLS, chunk_size)
            tile_size = (tile_size // chunk_size) * chunk_size
            var num_tiles = (row_size + tile_size - 1) // tile_size

            @__copy_capture(
                re_ptr, im_ptr, tile_size, target_1, target_2, theta_1, theta_2
            )
            @parameter
            fn process_tile_pp(tile_idx: Int):
                var col_start = tile_idx * tile_size
                var col_end = min(col_start + tile_size, row_size)
                col_end = (col_end // chunk_size) * chunk_size
                if col_end <= col_start:
                    return
                transform_column_fused_pp_simd_tiled[chunk_size](
                    re_ptr,
                    im_ptr,
                    num_rows,
                    row_size,
                    col_start,
                    col_end,
                    target_1,
                    target_2,
                    theta_1,
                    theta_2,
                )

            if use_parallel:
                parallelize[process_tile_pp](num_tiles)
            else:
                for tile_idx in range(num_tiles):
                    process_tile_pp(tile_idx)
            return True

        return False  # Cross-row but not supported fusion

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
        var theta0 = FloatType(t0.gate_info.arg.value())
        var theta1 = FloatType(t1.gate_info.arg.value())

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
        var theta0 = FloatType(t0.gate_info.arg.value())
        var theta1 = FloatType(t1.gate_info.arg.value())

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
            theta = FloatType(t1.gate_info.arg.value())
        else:
            th = t1.target
            tp = t0.target
            theta = FloatType(t0.gate_info.arg.value())

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


fn execute_grid_fused(
    mut state: QuantumState,
    circuit: QuantumCircuit,
    ctx: ExecContext = ExecContext(),
) raises:
    if ctx.validate_circuit:
        validate_circuit(state, circuit)
    var layout = compute_grid_layout(state, ctx)
    var col_bits = layout.col_bits
    var num_rows = layout.num_rows
    var row_size = layout.row_size

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
    var v: FloatType = 4.7
    var circuit = encode_value_circuit(n, v)

    from butterfly.utils.visualization import print_state

    var state = QuantumState(n)
    execute_scalar(state, circuit)
    print_state(state)
    from butterfly.utils.circuit_print import print_circuit_ascii

    print_circuit_ascii(circuit)
