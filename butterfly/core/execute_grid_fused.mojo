"""
Grid-Fused Executor: Combined parallelization + arithmetic reduction.

Combines grid's optimal parallelization (col_bits=n-3) with fused_v3's
arithmetic reduction patterns for best performance.
"""

from butterfly.core.state import (
    QuantumState,
    transform_simd,
    c_transform_simd_base_v2,
    simd_width,
)
from butterfly.core.circuit import (
    QuantumCircuit,
    Transformation,
    GateTransformation,
    SingleControlGateTransformation,
    BitReversalTransformation,
    get_involved_qubits,
    get_target,
    get_gate,
    is_controlled,
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
from butterfly.algos.unitary_kernels import Matrix4x4
from butterfly.core.types import Amplitude, Gate
from butterfly.core.gates import is_h, is_x, is_z, is_p, get_phase_angle
from butterfly.core.fused_kernels_sparse import (
    transform_row_fused_hh_simd,
    transform_row_fused_hp_simd,
    transform_row_fused_pp_simd,
    transform_row_fused_shared_c_pp_simd,
    transform_row_fused_st_hp_simd,
)
from algorithm import parallelize
from butterfly.utils.config import get_workers
from butterfly.utils.bit_utils import insert_zero_bit
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
    var row_control_bit: Int  # Qubit index if >= col_bits, else -1

    fn __init__(out self, t: Transformation, col_bits: Int):
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
        self.row_control_bit = -1

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

            # If control is in row index, record it for skipping
            if self.control >= col_bits:
                self.row_control_bit = self.control

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
        self.row_control_bit = existing.row_control_bit

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
        self.row_control_bit = existing.row_control_bit


struct GridFusionGroup(Copyable, Movable):
    """Group of gates to be fused together."""

    var transformations: List[Transformation]
    var is_row_local: Bool  # All targets < col_bits
    var is_fused: Bool
    var fusion_matrices: List[Matrix4x4]
    var fusion_gates: List[Gate]

    fn __init__(
        out self,
        is_row_local: Bool = False,
        is_fused: Bool = False,
    ):
        self.transformations = List[Transformation]()
        self.is_row_local = is_row_local
        self.is_fused = is_fused
        self.fusion_matrices = List[Matrix4x4]()
        self.fusion_gates = List[Gate]()

    fn __copyinit__(out self, existing: Self):
        self.transformations = List[Transformation]()
        for i in range(len(existing.transformations)):
            self.transformations.append(existing.transformations[i])
        self.is_row_local = existing.is_row_local
        self.is_fused = existing.is_fused
        self.fusion_matrices = List[Matrix4x4]()
        for i in range(len(existing.fusion_matrices)):
            self.fusion_matrices.append(existing.fusion_matrices[i])
        self.fusion_gates = List[Gate]()
        for i in range(len(existing.fusion_gates)):
            self.fusion_gates.append(existing.fusion_gates[i])

    fn __moveinit__(out self, deinit existing: Self):
        self.transformations = existing.transformations^
        self.is_row_local = existing.is_row_local
        self.is_fused = existing.is_fused
        self.fusion_matrices = existing.fusion_matrices^
        self.fusion_gates = existing.fusion_gates^


fn is_row_local_gate(t: Transformation, col_bits: Int) -> Bool:
    """Check if gate is row-local (target < col_bits).
    Controlled gates are considered local if the target is in the row and the
    control is either in the row OR in the global row-index bits (which allows skipping).
    """
    if t.isa[GateTransformation]():
        return t[GateTransformation].target < col_bits
    elif t.isa[SingleControlGateTransformation]():
        var sct = t[SingleControlGateTransformation].copy()
        # If target is local, it acts within a row.
        # If control is also local (< col_bits), it's a normal local controlled gate.
        # If control is global (>= col_bits), it's a "row-controlled skip" gate.
        return sct.target < col_bits
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

        # Bit reversals get their own group (marked as non-local)
        if t.isa[BitReversalTransformation]():
            var g = GridFusionGroup(is_row_local=False)
            g.transformations.append(t)
            groups.append(g^)
            i += 1
            continue

        if is_row_local_gate(t, col_bits):
            # Row-local group
            var g = GridFusionGroup(is_row_local=True, is_fused=False)
            g.transformations.append(t)
            i += 1

            # Collect consecutive row-local gates
            while i < n and is_row_local_gate(transformations[i], col_bits):
                if transformations[i].isa[BitReversalTransformation]():
                    break
                var t2 = transformations[i]
                g.transformations.append(t2)
                # If we have 2 or more row-local gates, we can try to fuse
                if not g.is_fused:
                    # Enable fusion if we have at least two transformations
                    if len(g.transformations) >= 2:
                        g.is_fused = True
                i += 1

            # Precompute all possible fusion matrices for this group
            # For a group of size N, we store Matrix(i, j) at index i * N + j
            var num_g = len(g.transformations)
            if num_g > 1:
                # Pre-unpack gates for precomputation
                var preps = List[PreparedTransformation]()
                for k in range(num_g):
                    preps.append(
                        PreparedTransformation(g.transformations[k], col_bits)
                    )
                var preps_ptr = preps.unsafe_ptr()

                # Initialize with placeholders
                for _ in range(num_g * num_g):
                    g.fusion_matrices.append(Matrix4x4(uninitialized=True))
                    g.fusion_gates.append(Gate(uninitialized=True))

                # Compute all pairs
                from butterfly.core.gates import matmul_2x2

                for k1 in range(num_g):
                    for k2 in range(num_g):
                        if k1 < k2:
                            var p1 = preps_ptr + k1
                            var p2 = preps_ptr + k2
                            # Different targets -> 4x4 Kronecker Fusion
                            if p1[].target != p2[].target:
                                var mat = compute_kron_product(
                                    p1[].gate, p2[].gate
                                )
                                if p1[].target < p2[].target:
                                    mat = compute_kron_product(
                                        p2[].gate, p1[].gate
                                    )
                                g.fusion_matrices[k1 * num_g + k2] = mat
                            else:
                                # Same target -> 2x2 Matrix Fusion
                                # G = G2 @ G1 (p2 is applied after p1)
                                g.fusion_gates[k1 * num_g + k2] = matmul_2x2(
                                    p2[].gate, p1[].gate
                                )

            groups.append(g^)
        else:
            # Cross-row gates - group them for better dispatch
            var g = GridFusionGroup(is_row_local=False)
            g.transformations.append(t)
            i += 1
            while i < n and not is_row_local_gate(transformations[i], col_bits):
                if transformations[i].isa[BitReversalTransformation]():
                    break
                g.transformations.append(transformations[i])
                i += 1
            groups.append(g^)

    return groups^


fn execute_grid_fused[
    N: Int,
    SIMD_WIDTH: Int = simd_width,
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
                state,
                g_ref[].transformations,
                g_ref[].fusion_matrices,
                g_ref[].fusion_gates,
                num_rows,
                row_size,
                actual_col_bits,
                g_ref[].is_fused,
            )
        else:
            # Global gates - fall back to standard SIMD dispatcher (Stable & Bit-Perfect)
            execute_transformations_simd_v2[N](state, g_ref[].transformations)


fn transform_row_matrix4_simd[
    simd_width: Int
](
    mut state: QuantumState,
    row: Int,
    row_size: Int,
    t1: Int,
    t2: Int,
    mat: Matrix4x4,
):
    """Apply a 4x4 matrix to two qubits within a row using SIMD."""
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var row_offset = row * row_size

    var high_pos = t1
    var low_pos = t2
    if t2 > t1:
        high_pos = t2
        low_pos = t1

    var stride_low = 1 << low_pos
    var stride_high = 1 << high_pos
    var num_base_pairs = row_size >> 2

    if stride_low >= simd_width:

        @parameter
        fn inner_simd[w: Int](k: Int):
            var i = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(i, high_pos)
            var idx1_local = idx0_local | stride_low
            var idx2_local = idx0_local | stride_high
            var idx3_local = idx1_local | stride_high

            var idx0 = row_offset + idx0_local
            var idx1 = row_offset + idx1_local
            var idx2 = row_offset + idx2_local
            var idx3 = row_offset + idx3_local

            var re0 = ptr_re.load[width=w](idx0)
            var im0 = ptr_im.load[width=w](idx0)
            var re1 = ptr_re.load[width=w](idx1)
            var im1 = ptr_im.load[width=w](idx1)
            var re2 = ptr_re.load[width=w](idx2)
            var im2 = ptr_im.load[width=w](idx2)
            var re3 = ptr_re.load[width=w](idx3)
            var im3 = ptr_im.load[width=w](idx3)

            # Complex Matrix Mul (Correct and Bit-Perfect)
            # Row 0
            var n_re0 = re0 * mat[0][0].re - im0 * mat[0][0].im
            var n_im0 = re0 * mat[0][0].im + im0 * mat[0][0].re
            n_re0 += re1 * mat[0][1].re - im1 * mat[0][1].im
            n_im0 += re1 * mat[0][1].im + im1 * mat[0][1].re
            n_re0 += re2 * mat[0][2].re - im2 * mat[0][2].im
            n_im0 += re2 * mat[0][2].im + im2 * mat[0][2].re
            n_re0 += re3 * mat[0][3].re - im3 * mat[0][3].im
            n_im0 += re3 * mat[0][3].im + im3 * mat[0][3].re

            # Row 1
            var n_re1 = re0 * mat[1][0].re - im0 * mat[1][0].im
            var n_im1 = re0 * mat[1][0].im + im0 * mat[1][0].re
            n_re1 += re1 * mat[1][1].re - im1 * mat[1][1].im
            n_im1 += re1 * mat[1][1].im + im1 * mat[1][1].re
            n_re1 += re2 * mat[1][2].re - im2 * mat[1][2].im
            n_im1 += re2 * mat[1][2].im + im2 * mat[1][2].re
            n_re1 += re3 * mat[1][3].re - im3 * mat[1][3].im
            n_im1 += re3 * mat[1][3].im + im3 * mat[1][3].re

            # Row 2
            var n_re2 = re0 * mat[2][0].re - im0 * mat[2][0].im
            var n_im2 = re0 * mat[2][0].im + im0 * mat[2][0].re
            n_re2 += re1 * mat[2][1].re - im1 * mat[2][1].im
            n_im2 += re1 * mat[2][1].im + im1 * mat[2][1].re
            n_re2 += re2 * mat[2][2].re - im2 * mat[2][2].im
            n_im2 += re2 * mat[2][2].im + im2 * mat[2][2].re
            n_re2 += re3 * mat[2][3].re - im3 * mat[2][3].im
            n_im2 += re3 * mat[2][3].im + im3 * mat[2][3].re

            # Row 3
            var n_re3 = re0 * mat[3][0].re - im0 * mat[3][0].im
            var n_im3 = re0 * mat[3][0].im + im0 * mat[3][0].re
            n_re3 += re1 * mat[3][1].re - im1 * mat[3][1].im
            n_im3 += re1 * mat[3][1].im + im1 * mat[3][1].re
            n_re3 += re2 * mat[3][2].re - im2 * mat[3][2].im
            n_im3 += re2 * mat[3][2].im + im2 * mat[3][2].re
            n_re3 += re3 * mat[3][3].re - im3 * mat[3][3].im
            n_im3 += re3 * mat[3][3].im + im3 * mat[3][3].re

            # Stores
            ptr_re.store(idx0, n_re0)
            ptr_im.store(idx0, n_im0)
            ptr_re.store(idx1, n_re1)
            ptr_im.store(idx1, n_im1)
            ptr_re.store(idx2, n_re2)
            ptr_im.store(idx2, n_im2)
            ptr_re.store(idx3, n_re3)
            ptr_im.store(idx3, n_im3)

        vectorize[inner_simd, simd_width](num_base_pairs)
    else:
        # Scalar fallback
        for k in range(num_base_pairs):
            var i = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(i, high_pos)
            var idx0 = row_offset + idx0_local
            var idx1 = idx0 | stride_low
            var idx2 = idx0 | stride_high
            var idx3 = idx1 | stride_high

            var re0 = ptr_re[idx0]
            var im0 = ptr_im[idx0]
            var re1 = ptr_re[idx1]
            var im1 = ptr_im[idx1]
            var re2 = ptr_re[idx2]
            var im2 = ptr_im[idx2]
            var re3 = ptr_re[idx3]
            var im3 = ptr_im[idx3]

            var n_re0 = re0 * mat[0][0].re - im0 * mat[0][0].im
            var n_im0 = re0 * mat[0][0].im + im0 * mat[0][0].re
            n_re0 += re1 * mat[0][1].re - im1 * mat[0][1].im
            n_im0 += re1 * mat[0][1].im + im1 * mat[0][1].re
            n_re0 += re2 * mat[0][2].re - im2 * mat[0][2].im
            n_im0 += re2 * mat[0][2].im + im2 * mat[0][2].re
            n_re0 += re3 * mat[0][3].re - im3 * mat[0][3].im
            n_im0 += re3 * mat[0][3].im + im3 * mat[0][3].re

            var n_re1 = re0 * mat[1][0].re - im0 * mat[1][0].im
            var n_im1 = re0 * mat[1][0].im + im0 * mat[1][0].re
            n_re1 += re1 * mat[1][1].re - im1 * mat[1][1].im
            n_im1 += re1 * mat[1][1].im + im1 * mat[1][1].re
            n_re1 += re2 * mat[1][2].re - im2 * mat[1][2].im
            n_im1 += re2 * mat[1][2].im + im2 * mat[1][2].re
            n_re1 += re3 * mat[1][3].re - im3 * mat[1][3].im
            n_im1 += re3 * mat[1][3].im + im3 * mat[1][3].re

            var n_re2 = re0 * mat[2][0].re - im0 * mat[2][0].im
            var n_im2 = re0 * mat[2][0].im + im0 * mat[2][0].re
            n_re2 += re1 * mat[2][1].re - im1 * mat[2][1].im
            n_im2 += re1 * mat[2][1].im + im1 * mat[2][1].re
            n_re2 += re2 * mat[2][2].re - im2 * mat[2][2].im
            n_im2 += re2 * mat[2][2].im + im2 * mat[2][2].re
            n_re2 += re3 * mat[2][3].re - im3 * mat[2][3].im
            n_im2 += re3 * mat[2][3].im + im3 * mat[2][3].re

            var n_re3 = re0 * mat[3][0].re - im0 * mat[3][0].im
            var n_im3 = re0 * mat[3][0].im + im0 * mat[3][0].re
            n_re3 += re1 * mat[3][1].re - im1 * mat[3][1].im
            n_im3 += re1 * mat[3][1].im + im1 * mat[3][1].re
            n_re3 += re2 * mat[3][2].re - im2 * mat[3][2].im
            n_im3 += re2 * mat[3][2].im + im2 * mat[3][2].re
            n_re3 += re3 * mat[3][3].re - im3 * mat[3][3].im
            n_im3 += re3 * mat[3][3].im + im3 * mat[3][3].re

            ptr_re[idx0] = n_re0
            ptr_im[idx0] = n_im0
            ptr_re[idx1] = n_re1
            ptr_im[idx1] = n_im1
            ptr_re[idx2] = n_re2
            ptr_im[idx2] = n_im2
            ptr_re[idx3] = n_re3
            ptr_im[idx3] = n_im3


fn execute_row_local_group[
    simd_width: Int
](
    mut state: QuantumState,
    transformations: List[Transformation],
    fusion_matrices: List[Matrix4x4],
    fusion_gates: List[Gate],
    num_rows: Int,
    row_size: Int,
    col_bits: Int,
    is_fused: Bool = False,
):
    """Execute row-local gates in parallel."""

    var prepared = List[PreparedTransformation]()
    for i in range(len(transformations)):
        prepared.append(PreparedTransformation(transformations[i], col_bits))
    var prepared_ptr = prepared.unsafe_ptr()
    var matrices_ptr = fusion_matrices.unsafe_ptr()
    var gates_ptr = fusion_gates.unsafe_ptr()
    var num_prepared = len(prepared)

    @parameter
    fn process_row(row: Int):
        var i = 0
        while i < num_prepared:
            var p = prepared_ptr + i

            # Row-Controlled Skip Logic
            if p[].row_control_bit >= 0:
                var row_bit_pos = p[].row_control_bit - col_bits
                if not ((row >> row_bit_pos) & 1):
                    # Control bit is 0 for this row, skip this gate
                    i += 1
                    continue
                # Control bit is 1, treat as uncontrolled within this row

            if is_fused and i + 1 < num_prepared:
                # Lookahead Fusion: find next active gate to fuse with
                var fused = False
                for j in range(i + 1, num_prepared):
                    var p2 = prepared_ptr + j
                    # Check if p2 is active for this row
                    var p2_active = True
                    if p2[].row_control_bit >= 0:
                        var row_bit_pos = p2[].row_control_bit - col_bits
                        if not ((row >> row_bit_pos) & 1):
                            p2_active = False

                    if p2_active:
                        # Found next active gate. Can we fuse?
                        var can_fuse = False
                        if (
                            not p[].is_controlled or p[].row_control_bit >= 0
                        ) and (
                            not p2[].is_controlled or p2[].row_control_bit >= 0
                        ):
                            can_fuse = True
                        elif (
                            p[].is_controlled
                            and p2[].is_controlled
                            and p[].control == p2[].control
                        ):
                            can_fuse = True

                        if can_fuse:
                            # Case A: Different targets -> 4x4 Fusion
                            if p[].target != p2[].target:
                                # Dispatch to specialized sparse kernels or fallback to dense
                                if p[].is_h_gate and p2[].is_h_gate:
                                    transform_row_fused_hh_simd[simd_width](
                                        state,
                                        row,
                                        row_size,
                                        p[].target,
                                        p2[].target,
                                    )
                                elif p[].is_h_gate and p2[].is_p_gate:
                                    transform_row_fused_hp_simd[simd_width](
                                        state,
                                        row,
                                        row_size,
                                        p[].target,
                                        p2[].target,
                                        p2[].theta,
                                    )
                                elif p[].is_p_gate and p2[].is_h_gate:
                                    # Order matters for HP kernel (H index first)
                                    transform_row_fused_hp_simd[simd_width](
                                        state,
                                        row,
                                        row_size,
                                        p2[].target,
                                        p[].target,
                                        p[].theta,
                                    )
                                elif p[].is_p_gate and p2[].is_p_gate:
                                    if (
                                        p[].control == p2[].control
                                        and p[].is_controlled
                                        and p2[].is_controlled
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
                                    else:
                                        transform_row_fused_pp_simd[simd_width](
                                            state,
                                            row,
                                            row_size,
                                            p[].target,
                                            p2[].target,
                                            p[].theta,
                                            p2[].theta,
                                        )
                                else:
                                    # Fallback to Generic Dense Matrix Fusion
                                    var mat = matrices_ptr[i * num_prepared + j]
                                    transform_row_matrix4_simd[simd_width](
                                        state,
                                        row,
                                        row_size,
                                        p[].target,
                                        p2[].target,
                                        mat,
                                    )
                                i = j + 1
                                fused = True
                            else:
                                # Case B: Same target -> 2x2 Fusion
                                # Detect specialized Same-Target H+P pattern
                                if p[].is_h_gate and p2[].is_p_gate:
                                    transform_row_fused_st_hp_simd[simd_width](
                                        state,
                                        row,
                                        row_size,
                                        p[].target,
                                        p2[].theta,
                                    )
                                else:
                                    # Fallback to Generic 2x2 Matrix Fusion
                                    var fused_gate = gates_ptr[
                                        i * num_prepared + j
                                    ]
                                    transform_row_simd[simd_width](
                                        state,
                                        row,
                                        row_size,
                                        p[].target,
                                        fused_gate,
                                    )
                                i = j + 1
                                fused = True
                        break  # Stop looking after finding first active successor
                    else:
                        # Successor is skipped. Continue lookahead to i+2, etc.
                        continue

                if fused:
                    continue

            # Fallback to single gate execution
            if p[].is_controlled and p[].row_control_bit < 0:
                # Actual row-local controlled gate
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
                    transform_row_simd[simd_width](
                        state, row, row_size, p[].target, p[].gate
                    )
            else:
                # Uncontrolled gate OR row-controlled skip gate (now satisfied)
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
                else:
                    transform_row_simd[simd_width](
                        state, row, row_size, p[].target, p[].gate
                    )
            i += 1

    var workers = get_workers("v_grid_rows")
    if workers > 0:
        parallelize[process_row](num_rows, workers)
    else:
        parallelize[process_row](num_rows)
