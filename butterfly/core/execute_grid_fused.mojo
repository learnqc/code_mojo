"""
Grid-Fused Executor: Combined parallelization + arithmetic reduction.

Combines grid's optimal parallelization (col_bits=n-3) with fused_v3's
arithmetic reduction patterns for best performance.
"""

from butterfly.core.state import (
    QuantumState,
    transform_simd,
    c_transform_simd_base_v2,
)
from butterfly.core.circuit import (
    QuantumCircuit,
    Transformation,
    GateTransformation,
    SingleControlGateTransformation,
    BitReversalTransformation,
    get_involved_qubits,
    get_target,
)
from butterfly.core.execute_simd_v2_dispatch import (
    execute_transformations_simd_v2,
)
from butterfly.core.execute_as_grid import (
    transform_row,
    transform_row_h_simd,
    transform_row_x_simd,
    transform_row_z_simd,
    transform_row_p_simd,
    transform_row_simd,
    c_transform_row_h_simd,
    c_transform_row_p_simd,
)
from butterfly.algos.fused_gates import compute_kron_product
from butterfly.algos.unitary_kernels import matmul_matrix4x4, Matrix4x4
from butterfly.core.c_transform_fast_v2 import (
    c_transform_h_simd_v2,
    c_transform_p_simd_v2,
)
from butterfly.core.types import Amplitude, Gate
from butterfly.core.gates import is_h, is_x, is_z, is_p, get_phase_angle
from algorithm import parallelize
from butterfly.utils.config import get_workers
from math import log2


@fieldwise_init
struct PreparedTransformation(Copyable, Movable):
    """Unpacked transformation for faster execution in hot loops."""

    var is_gate: Bool
    var is_controlled: Bool
    var target: Int
    var control: Int
    var gate: Gate
    var theta: Float64
    var is_h_gate: Bool
    var is_x_gate: Bool
    var is_z_gate: Bool
    var is_p_gate: Bool

    fn __init__(out self, t: Transformation):
        self.is_gate = False
        self.is_controlled = False
        self.target = -1
        self.control = -1
        self.gate = Gate(uninitialized=True)
        self.theta = 0.0
        self.is_h_gate = False
        self.is_x_gate = False
        self.is_z_gate = False
        self.is_p_gate = False

        if t.isa[GateTransformation]():
            var gt = t[GateTransformation].copy()
            self.is_gate = True
            self.target = gt.target
            self.gate = gt.gate
            self.is_h_gate = is_h(gt.gate)
            self.is_x_gate = is_x(gt.gate)
            self.is_z_gate = is_z(gt.gate)
            self.is_p_gate = is_p(gt.gate)
            if self.is_p_gate:
                self.theta = get_phase_angle(gt.gate)
        elif t.isa[SingleControlGateTransformation]():
            var sct = t[SingleControlGateTransformation].copy()
            self.is_controlled = True
            self.target = sct.target
            self.control = sct.control
            self.gate = sct.gate
            self.is_h_gate = is_h(sct.gate)
            self.is_p_gate = is_p(sct.gate)
            if self.is_p_gate:
                self.theta = get_phase_angle(sct.gate)

    fn __copyinit__(out self, existing: Self):
        self.is_gate = existing.is_gate
        self.is_controlled = existing.is_controlled
        self.target = existing.target
        self.control = existing.control
        self.gate = existing.gate
        self.theta = existing.theta
        self.is_h_gate = existing.is_h_gate
        self.is_x_gate = existing.is_x_gate
        self.is_z_gate = existing.is_z_gate
        self.is_p_gate = existing.is_p_gate

    fn __moveinit__(out self, deinit existing: Self):
        self.is_gate = existing.is_gate
        self.is_controlled = existing.is_controlled
        self.target = existing.target
        self.control = existing.control
        self.gate = existing.gate
        self.theta = existing.theta
        self.is_h_gate = existing.is_h_gate
        self.is_x_gate = existing.is_x_gate
        self.is_z_gate = existing.is_z_gate
        self.is_p_gate = existing.is_p_gate


struct GridFusionGroup(Copyable, Movable):
    """Group of gates to be fused together."""

    var transformations: List[Transformation]
    var is_row_local: Bool  # All targets < col_bits
    var is_radix4: Bool  # 2 gates to fuse

    fn __init__(out self, is_row_local: Bool = False, is_radix4: Bool = False):
        self.transformations = List[Transformation]()
        self.is_row_local = is_row_local
        self.is_radix4 = is_radix4

    fn __copyinit__(out self, existing: Self):
        self.transformations = List[Transformation]()
        for i in range(len(existing.transformations)):
            self.transformations.append(existing.transformations[i])
        self.is_row_local = existing.is_row_local
        self.is_radix4 = existing.is_radix4

    fn __moveinit__(out self, deinit existing: Self):
        self.transformations = existing.transformations^
        self.is_row_local = existing.is_row_local
        self.is_radix4 = existing.is_radix4


fn is_row_local_gate(t: Transformation, col_bits: Int) -> Bool:
    """Check if gate is row-local (all qubits < col_bits)."""
    if t.isa[GateTransformation]():
        return t[GateTransformation].target < col_bits
    elif t.isa[SingleControlGateTransformation]():
        var sct = t[SingleControlGateTransformation].copy()
        return sct.control < col_bits and sct.target < col_bits
    return False


fn analyze_for_grid_fusion(
    transformations: List[Transformation],
    col_bits: Int,
) -> List[GridFusionGroup]:
    """Analyze circuit for grid-fusion opportunities."""
    var groups = List[GridFusionGroup]()
    var i = 0
    var n = len(transformations)

    while i < n:
        var t = transformations[i]

        # Bit reversals get their own group
        if t.isa[BitReversalTransformation]():
            var g = GridFusionGroup()
            g.transformations.append(t)
            groups.append(g^)
            i += 1
            continue

        if is_row_local_gate(t, col_bits):
            # Row-local group
            var g = GridFusionGroup(is_row_local=True)
            g.transformations.append(t)
            i += 1

            # Collect consecutive row-local gates
            while i < n and is_row_local_gate(transformations[i], col_bits):
                if transformations[i].isa[BitReversalTransformation]():
                    break
                var t2 = transformations[i]
                g.transformations.append(t2)
                i += 1

            groups.append(g^)
        else:
            # Cross-row: try radix-4
            if i + 1 < n:
                var next_t = transformations[i + 1]

                if (
                    t.isa[GateTransformation]()
                    and next_t.isa[GateTransformation]()
                ):
                    var gt1 = t[GateTransformation].copy()
                    var gt2 = next_t[GateTransformation].copy()

                    if gt1.target != gt2.target:
                        var g = GridFusionGroup(
                            is_row_local=False, is_radix4=True
                        )
                        g.transformations.append(t)
                        g.transformations.append(next_t)
                        groups.append(g^)
                        i += 2
                        continue

            # Single cross-row gate
            var g = GridFusionGroup(is_row_local=False, is_radix4=False)
            g.transformations.append(t)
            groups.append(g^)
            i += 1

    return groups^


fn execute_grid_fused[
    N: Int,
    SIMD_WIDTH: Int = 8,
](
    mut state: QuantumState, mut circuit: QuantumCircuit, col_bits: Int = -1
) raises:
    """Execute circuit using grid + fusion."""
    var n = circuit.num_qubits
    var actual_col_bits = col_bits if col_bits >= 0 else (n - 3)

    var num_rows = 1 << (n - actual_col_bits)
    var row_size = 1 << actual_col_bits

    var groups = analyze_for_grid_fusion(
        circuit.transformations, actual_col_bits
    )

    var groups_ptr = groups.unsafe_ptr()
    for i in range(len(groups)):
        var g_ref = groups_ptr + i
        if g_ref[].is_row_local:
            execute_row_local_group[SIMD_WIDTH](
                state, g_ref[].transformations, num_rows, row_size
            )
        elif g_ref[].is_radix4:
            _execute_radix4_cross_row[N](state, g_ref[].transformations)

        else:
            # Single cross-row gate
            if len(g_ref[].transformations) > 0:
                var t = g_ref[].transformations[0]
                if t.isa[GateTransformation]():
                    var gt = t[GateTransformation].copy()
                    transform_simd[N](state, gt.target, gt.gate)
                elif t.isa[SingleControlGateTransformation]():
                    var sct = t[SingleControlGateTransformation].copy()
                    from butterfly.core.gates import is_h, is_p, get_phase_angle
                    from butterfly.core.c_transform_fast_v2 import (
                        c_transform_h_simd_v2,
                        c_transform_p_simd_v2,
                    )

                    if is_h(sct.gate):
                        c_transform_h_simd_v2(state, sct.control, sct.target)
                    elif is_p(sct.gate):
                        c_transform_p_simd_v2(
                            state,
                            sct.control,
                            sct.target,
                            get_phase_angle(sct.gate),
                        )
                    else:
                        from butterfly.core.state import (
                            c_transform_simd_base_v2,
                        )

                        c_transform_simd_base_v2[N](
                            state, sct.control, 1 << sct.target, sct.gate
                        )
                else:
                    execute_transformations_simd_v2[N](
                        state, g_ref[].transformations
                    )


fn execute_row_local_group[
    simd_width: Int
](
    mut state: QuantumState,
    transformations: List[Transformation],
    num_rows: Int,
    row_size: Int,
):
    """Execute row-local gates in parallel."""

    var prepared = List[PreparedTransformation]()
    for i in range(len(transformations)):
        prepared.append(PreparedTransformation(transformations[i]))
    var prepared_ptr = prepared.unsafe_ptr()
    var num_prepared = len(prepared)

    @parameter
    fn process_row(row: Int):
        for i in range(num_prepared):
            var p = prepared_ptr + i

            if p[].is_gate:
                if row_size >= simd_width:
                    if p[].is_h_gate:
                        transform_row_h_simd[simd_width](
                            state, row, row_size, p[].target
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
                    elif (1 << p[].target) >= simd_width:
                        transform_row_simd[simd_width](
                            state, row, row_size, p[].target, p[].gate
                        )
                    else:
                        transform_row(
                            state, row, row_size, p[].target, p[].gate
                        )
                else:
                    transform_row(state, row, row_size, p[].target, p[].gate)
            elif p[].is_controlled:
                if p[].is_h_gate:
                    c_transform_row_h_simd[simd_width](
                        state, row, row_size, p[].control, p[].target
                    )
                elif p[].is_p_gate:
                    c_transform_row_p_simd[simd_width](
                        state, row, row_size, p[].control, p[].target, p[].theta
                    )

    var row_workers = get_workers("v_grid_rows")
    if row_workers > 0:
        parallelize[process_row](num_rows, row_workers)
    else:
        parallelize[process_row](num_rows)


fn _execute_radix4_cross_row[
    N: Int
](mut state: QuantumState, transformations: List[Transformation]):
    """Execute 2 cross-row gates fused."""
    if len(transformations) < 2:
        return

    var t1 = transformations[0]
    var t2 = transformations[1]

    if not t1.isa[GateTransformation]() or not t2.isa[GateTransformation]():
        execute_transformations_simd_v2[N](state, transformations)
        return

    var gt1 = t1[GateTransformation].copy()
    var gt2 = t2[GateTransformation].copy()

    var q_high = max(gt1.target, gt2.target)
    var q_low = min(gt1.target, gt2.target)

    # Check if they are on the same qubit
    if q_high == q_low:
        from butterfly.core.gates import matmul_2x2

        var G = matmul_2x2(gt2.gate, gt1.gate)
        transform_simd[N](state, q_high, G)
        return

    # Build fused matrix via Kronecker product
    var g_high: Gate
    var g_low: Gate
    if gt1.target > gt2.target:
        g_high = gt1.gate
        g_low = gt2.gate
    else:
        g_high = gt2.gate
        g_low = gt1.gate

    var M = compute_kron_product(g_high, g_low)

    # Apply
    from butterfly.algos.fused_gates import transform_matrix4

    transform_matrix4(state, q_high, q_low, M)
