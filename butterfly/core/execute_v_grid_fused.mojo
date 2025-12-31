"""
execute_v_grid_fused: Grid execution with gate fusion.

Combines v_grid's row-level parallelism with fused_v3's gate fusion strategy.
Goal: Beat fused_v3 by leveraging better parallelism + fusion.
"""
from algorithm import parallelize
from butterfly.core.state import QuantumState
from butterfly.core.types import Gate, Amplitude
from butterfly.core.circuit import (
    QuantumCircuit,
    Transformation,
    GateTransformation,
    BitReversalTransformation,
    SingleControlGateTransformation,
    is_permutation,
    is_controlled,
    get_target,
    get_gate,
    get_involved_qubits,
)
from butterfly.core.gates import is_h, is_x, is_z, is_p, get_phase_angle
from butterfly.algos.unitary_kernels import Matrix4x4, acc_mul
from butterfly.algos.fused_gates import compute_kron_product
from butterfly.utils.bit_utils import insert_zero_bit
from math import cos, sin, log2


struct TransformationGroup(Copyable, Movable):
    """A group of transformations that can be fused together."""

    var transformations: List[Transformation]
    var is_local: Bool  # If true, all gates operate within rows
    var is_fused: Bool  # If true, can be fused into a single matrix

    fn __init__(
        out self,
        is_local: Bool = False,
        is_fused: Bool = False,
    ):
        self.transformations = List[Transformation]()
        self.is_local = is_local
        self.is_fused = is_fused

    fn __copyinit__(out self, existing: Self):
        self.transformations = List[Transformation]()
        for i in range(len(existing.transformations)):
            self.transformations.append(existing.transformations[i])
        self.is_local = existing.is_local
        self.is_fused = existing.is_fused

    fn __moveinit__(out self, deinit existing: Self):
        self.transformations = existing.transformations^
        self.is_local = existing.is_local
        self.is_fused = existing.is_fused

    fn copy(self) -> Self:
        var new_g = TransformationGroup(self.is_local, self.is_fused)
        for i in range(len(self.transformations)):
            new_g.transformations.append(self.transformations[i])
        return new_g^


struct GridCircuitAnalyzer:
    """Analyzes circuit to identify fusible gate groups for grid execution."""

    var groups: List[TransformationGroup]
    var col_bits: Int

    fn __init__(
        out self,
        transformations: List[Transformation],
        col_bits: Int,
    ):
        self.groups = List[TransformationGroup]()
        self.col_bits = col_bits
        self.analyze(transformations)

    fn is_local(self, t: Transformation) -> Bool:
        """Check if transformation operates within rows (target < col_bits)."""
        if is_controlled(t):
            # Controlled gates are local if both control and target are within rows
            var control = t[SingleControlGateTransformation].control
            var target = t[SingleControlGateTransformation].target
            return control < self.col_bits and target < self.col_bits
        if is_permutation(t):
            return False
        var involved = get_involved_qubits(t)
        for i in range(len(involved)):
            if involved[i] >= self.col_bits:
                return False
        return True

    fn can_fuse(self, t1: Transformation, t2: Transformation) -> Bool:
        """Check if two transformations can be fused."""
        # Both must be local gate transformations
        if not self.is_local(t1) or not self.is_local(t2):
            return False
        if is_controlled(t1) or is_controlled(t2):
            return False
        if is_permutation(t1) or is_permutation(t2):
            return False
        # Must target different qubits
        if get_target(t1) == get_target(t2):
            return False
        return True

    fn analyze(mut self, transformations: List[Transformation]):
        """Partition circuit into fusible groups."""
        var i = 0
        var n = len(transformations)

        while i < n:
            var t = transformations[i]

            # Permutations go in their own group
            if is_permutation(t):
                var g = TransformationGroup()
                g.transformations.append(t)
                self.groups.append(g^)
                i += 1
                continue

            # Try to build a local fused group
            if self.is_local(t):
                var g = TransformationGroup(is_local=True, is_fused=False)
                g.transformations.append(t)
                i += 1

                # Try to fuse with next gate
                while i < n:
                    var next_t = transformations[i]
                    if is_permutation(next_t):
                        break
                    if not self.can_fuse(t, next_t):
                        # Can still add to group if local, just not fused
                        if self.is_local(next_t) and not is_controlled(next_t):
                            g.transformations.append(next_t)
                            i += 1
                        else:
                            break
                    else:
                        # Can fuse!
                        g.is_fused = True
                        g.transformations.append(next_t)
                        i += 1
                        break  # For now, only fuse pairs

                self.groups.append(g^)
            else:
                # Global gate - execute individually
                var g = TransformationGroup(is_local=False, is_fused=False)
                g.transformations.append(t)
                self.groups.append(g^)
                i += 1


fn execute_v_grid_fused(
    mut state: QuantumState, circuit: QuantumCircuit, col_bits: Int
) raises:
    """Execute circuit using grid-based execution with gate fusion.

    Args:
        state: The quantum state to transform
        circuit: The circuit to execute
        col_bits: Number of qubits in each row (grid columns)
    """
    var n = Int(log2(Float64(len(state))))
    var num_rows = 1 << (n - col_bits)
    var row_size = 1 << col_bits

    # Analyze circuit to find fusible groups
    var analyzer = GridCircuitAnalyzer(circuit.transformations, col_bits)

    # Execute each group
    for group_idx in range(len(analyzer.groups)):
        var g = analyzer.groups[group_idx].copy()

        if is_permutation(g.transformations[0]):
            # Bit reversal
            from butterfly.core.state import bit_reverse_state

            bit_reverse_state(state)
        elif g.is_local and g.is_fused and len(g.transformations) == 2:
            # Fused local gates - apply combined matrix to each row
            execute_fused_local_pair(
                state, g.transformations, num_rows, row_size
            )
        elif g.is_local:
            # Local gates - apply sequentially to each row
            execute_local_group(state, g.transformations, num_rows, row_size)
        else:
            # Global gates - fall back to standard v_grid execution
            execute_global_group(state, g.transformations, col_bits)


fn execute_fused_local_pair(
    mut state: QuantumState,
    transformations: List[Transformation],
    num_rows: Int,
    row_size: Int,
):
    """Execute two local gates fused into a single 4x4 matrix operation.

    Since both gates are local (target < col_bits), they operate within each row.
    We can apply the fused 4x4 matrix to each row independently in parallel.
    """
    var t0 = transformations[0]
    var t1 = transformations[1]

    var q0 = get_target(t0)
    var q1 = get_target(t1)
    var gate0 = get_gate(t0)
    var gate1 = get_gate(t1)

    # Compute the 4x4 matrix (Kronecker product)
    var M: Matrix4x4
    var low_pos: Int
    var high_pos: Int

    if q0 > q1:
        M = compute_kron_product(gate0, gate1)
        high_pos = q0
        low_pos = q1
    else:
        M = compute_kron_product(gate1, gate0)
        high_pos = q1
        low_pos = q0

    # Apply the 4x4 matrix to each row in parallel
    @parameter
    fn process_row(row: Int):
        var row_offset = row * row_size
        var ptr_re = state.re.unsafe_ptr()
        var ptr_im = state.im.unsafe_ptr()

        # Within this row, we need to apply the 4x4 matrix
        # The indices within the row are computed using insert_zero_bit
        # but relative to the row, not the entire state
        var stride1 = 1 << low_pos
        var stride2 = 1 << high_pos

        # Number of 4-element groups in this row
        var num_quads = row_size >> 2  # row_size / 4

        for k in range(num_quads):
            # Compute base index within row (with both bits zeroed)
            var local_idx = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(local_idx, high_pos)

            # Convert to global index
            var idx0 = row_offset + idx0_local
            var idx1 = idx0 | stride1
            var idx2 = idx0 | stride2
            var idx3 = idx1 | stride2

            # Load amplitudes
            var re0 = ptr_re[idx0]
            var im0 = ptr_im[idx0]
            var re1 = ptr_re[idx1]
            var im1 = ptr_im[idx1]
            var re2 = ptr_re[idx2]
            var im2 = ptr_im[idx2]
            var re3 = ptr_re[idx3]
            var im3 = ptr_im[idx3]

            # Apply 4x4 matrix M
            # Row 0
            var n_re0 = re0 * M[0][0].re - im0 * M[0][0].im
            var n_im0 = re0 * M[0][0].im + im0 * M[0][0].re
            n_re0 += re1 * M[0][1].re - im1 * M[0][1].im
            n_im0 += re1 * M[0][1].im + im1 * M[0][1].re
            n_re0 += re2 * M[0][2].re - im2 * M[0][2].im
            n_im0 += re2 * M[0][2].im + im2 * M[0][2].re
            n_re0 += re3 * M[0][3].re - im3 * M[0][3].im
            n_im0 += re3 * M[0][3].im + im3 * M[0][3].re

            # Row 1
            var n_re1 = re0 * M[1][0].re - im0 * M[1][0].im
            var n_im1 = re0 * M[1][0].im + im0 * M[1][0].re
            n_re1 += re1 * M[1][1].re - im1 * M[1][1].im
            n_im1 += re1 * M[1][1].im + im1 * M[1][1].re
            n_re1 += re2 * M[1][2].re - im2 * M[1][2].im
            n_im1 += re2 * M[1][2].im + im2 * M[1][2].re
            n_re1 += re3 * M[1][3].re - im3 * M[1][3].im
            n_im1 += re3 * M[1][3].im + im3 * M[1][3].re

            # Row 2
            var n_re2 = re0 * M[2][0].re - im0 * M[2][0].im
            var n_im2 = re0 * M[2][0].im + im0 * M[2][0].re
            n_re2 += re1 * M[2][1].re - im1 * M[2][1].im
            n_im2 += re1 * M[2][1].im + im1 * M[2][1].re
            n_re2 += re2 * M[2][2].re - im2 * M[2][2].im
            n_im2 += re2 * M[2][2].im + im2 * M[2][2].re
            n_re2 += re3 * M[2][3].re - im3 * M[2][3].im
            n_im2 += re3 * M[2][3].im + im3 * M[2][3].re

            # Row 3
            var n_re3 = re0 * M[3][0].re - im0 * M[3][0].im
            var n_im3 = re0 * M[3][0].im + im0 * M[3][0].re
            n_re3 += re1 * M[3][1].re - im1 * M[3][1].im
            n_im3 += re1 * M[3][1].im + im1 * M[3][1].re
            n_re3 += re2 * M[3][2].re - im2 * M[3][2].im
            n_im3 += re2 * M[3][2].im + im2 * M[3][2].re
            n_re3 += re3 * M[3][3].re - im3 * M[3][3].im
            n_im3 += re3 * M[3][3].im + im3 * M[3][3].re

            # Store results
            ptr_re[idx0] = n_re0
            ptr_im[idx0] = n_im0
            ptr_re[idx1] = n_re1
            ptr_im[idx1] = n_im1
            ptr_re[idx2] = n_re2
            ptr_im[idx2] = n_im2
            ptr_re[idx3] = n_re3
            ptr_im[idx3] = n_im3

    parallelize[process_row](num_rows)


fn execute_local_group(
    mut state: QuantumState,
    transformations: List[Transformation],
    num_rows: Int,
    row_size: Int,
):
    """Execute local gates sequentially on each row."""
    from butterfly.core.execute_as_grid import (
        transform_row_simd,
        c_transform_row_h_simd,
        c_transform_row_p_simd,
    )

    @parameter
    fn process_row(row: Int):
        for i in range(len(transformations)):
            var t = transformations[i]

            if is_controlled(t):
                # Controlled gate - use row-local controlled kernels
                var sct = t[SingleControlGateTransformation].copy()
                var control = sct.control
                var target = sct.target
                var gate = sct.gate

                if is_h(gate):
                    c_transform_row_h_simd(
                        state, row, row_size, control, target
                    )
                elif is_p(gate):
                    c_transform_row_p_simd(
                        state,
                        row,
                        row_size,
                        control,
                        target,
                        get_phase_angle(gate),
                    )
            else:
                # Uncontrolled gate
                var target = get_target(t)
                var gate = get_gate(t)
                transform_row_simd[8](state, row, row_size, target, gate)

    parallelize[process_row](num_rows)


fn execute_global_group(
    mut state: QuantumState,
    transformations: List[Transformation],
    col_bits: Int,
) raises:
    """Execute global gates using standard v_grid approach."""
    from butterfly.core.execute_as_grid import execute_as_grid

    # Create temporary circuit with just these transformations
    var temp_circuit = QuantumCircuit(0)  # num_qubits doesn't matter here
    for i in range(len(transformations)):
        temp_circuit.transformations.append(transformations[i])

    execute_as_grid(state, temp_circuit, col_bits)
