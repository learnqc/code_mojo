"""
Grid-Fused V4 Strategy: Zero-Copy Virtual Grid + Radix-4 Fusion.
Optimized for Contiguous SIMD access.
"""
from math import log2
from algorithm import parallelize, vectorize
from butterfly.core.state import QuantumState, simd_width
from butterfly.core.types import FloatType, Type, Gate, Amplitude
from butterfly.core.circuit import (
    QuantumCircuit,
    Transformation,
    GateTransformation,
    SingleControlGateTransformation,
    BitReversalTransformation,
    is_controlled,
    num_controls,
    get_target,
    get_gate,
    get_controls,
    get_involved_qubits,
    get_as_matrix4x4,
)
from butterfly.algos.unitary_kernels import Matrix4x4, acc_mul, matmul_matrix4x4
from butterfly.algos.fused_gates import transform_matrix4, compute_kron_product

alias COL_COL_FUSION = 0
alias ROW_ROW_FUSION = 1
alias ROW_COL_FUSION = 2
alias SINGLE_GATE = 3


struct GridFusedGroup(Copyable, Movable):
    var transformations: List[Transformation]
    var fusion_type: Int

    fn __init__(out self, fusion_type: Int):
        self.transformations = List[Transformation]()
        self.fusion_type = fusion_type

    fn __copyinit__(out self, existing: Self):
        self.transformations = List[Transformation]()
        for i in range(len(existing.transformations)):
            self.transformations.append(existing.transformations[i])
        self.fusion_type = existing.fusion_type

    fn __moveinit__(out self, deinit existing: Self):
        self.transformations = existing.transformations^
        self.fusion_type = existing.fusion_type

    fn copy(self) -> Self:
        var new_g = GridFusedGroup(self.fusion_type)
        for i in range(len(self.transformations)):
            new_g.transformations.append(self.transformations[i])
        return new_g^


struct GridFusedAnalyzer:
    var groups: List[GridFusedGroup]
    var col_bits: Int

    fn __init__(out self, transformations: List[Transformation], col_bits: Int):
        self.groups = List[GridFusedGroup]()
        self.col_bits = col_bits
        self.analyze(transformations)

    fn is_col_qubit(self, q: Int) -> Bool:
        return q < self.col_bits

    fn analyze(mut self, transformations: List[Transformation]):
        var i = 0
        var n = len(transformations)

        while i < n:
            var t0 = transformations[i]

            # For now only fuse non-controlled GateTransformations
            if t0.isa[GateTransformation]() and i + 1 < n:
                var t1 = transformations[i + 1]
                if t1.isa[GateTransformation]():
                    var q0 = t0[GateTransformation].target
                    var q1 = t1[GateTransformation].target

                    if q0 != q1:
                        # Determine fusion type
                        var f_type: Int
                        var is_q0_col = self.is_col_qubit(q0)
                        var is_q1_col = self.is_col_qubit(q1)

                        if is_q0_col and is_q1_col:
                            f_type = COL_COL_FUSION
                        elif not is_q0_col and not is_q1_col:
                            f_type = ROW_ROW_FUSION
                        else:
                            f_type = ROW_COL_FUSION

                        var g = GridFusedGroup(f_type)
                        g.transformations.append(t0)
                        g.transformations.append(t1)
                        self.groups.append(g^)
                        i += 2
                        continue

            # Fallback to single gate
            var g = GridFusedGroup(SINGLE_GATE)
            g.transformations.append(t0)
            self.groups.append(g^)
            i += 1


fn execute_grid_fused_v4(
    mut state: QuantumState, circuit: QuantumCircuit, col_bits: Int
) raises:
    """Execute a circuit using Grid-Fused V4 strategy."""
    var analyzer = GridFusedAnalyzer(circuit.transformations, col_bits)

    for i in range(len(analyzer.groups)):
        var g = analyzer.groups[i].copy()

        if g.fusion_type == SINGLE_GATE:
            var t = g.transformations[0]
            if t.isa[GateTransformation]():
                from butterfly.core.execute_as_grid import execute_as_grid

                var sub_circuit = QuantumCircuit(circuit.num_qubits)
                sub_circuit.transformations.append(t)
                execute_as_grid(state, sub_circuit, col_bits)
            elif t.isa[BitReversalTransformation]():
                from butterfly.core.state import bit_reverse_state

                bit_reverse_state(state)
        elif g.fusion_type == COL_COL_FUSION:
            execute_col_col_fusion(state, g.transformations, col_bits)
        elif g.fusion_type == ROW_ROW_FUSION:
            execute_row_row_fusion(state, g.transformations, col_bits)
        elif g.fusion_type == ROW_COL_FUSION:
            execute_row_col_fusion(state, g.transformations, col_bits)


@always_inline
fn _apply_M4_simd[
    w: Int
](
    base_re: UnsafePointer[FloatType],
    base_im: UnsafePointer[FloatType],
    idx0: SIMD[DType.int64, w],
    idx1: SIMD[DType.int64, w],
    idx2: SIMD[DType.int64, w],
    idx3: SIMD[DType.int64, w],
    M: Matrix4x4,
    is_contiguous: Bool = False,
):
    var ptr_re = UnsafePointer[FloatType](base_re.address)
    var ptr_im = UnsafePointer[FloatType](base_im.address)

    @parameter
    if w > 1:
        if is_contiguous:
            # Contiguous SIMD (Direct Load/Store)
            var i0_idx = Int(idx0[0])
            var i1_idx = Int(idx1[0])
            var i2_idx = Int(idx2[0])
            var i3_idx = Int(idx3[0])
            var v0r = ptr_re.load[width=w](i0_idx)
            var v0i = ptr_im.load[width=w](i0_idx)
            var v1r = ptr_re.load[width=w](i1_idx)
            var v1i = ptr_im.load[width=w](i1_idx)
            var v2r = ptr_re.load[width=w](i2_idx)
            var v2i = ptr_im.load[width=w](i2_idx)
            var v3r = ptr_re.load[width=w](i3_idx)
            var v3i = ptr_im.load[width=w](i3_idx)

            var res0_re = (
                v0r * M[0][0].re
                - v0i * M[0][0].im
                + v1r * M[0][1].re
                - v1i * M[0][1].im
                + v2r * M[0][2].re
                - v2i * M[0][2].im
                + v3r * M[0][3].re
                - v3i * M[0][3].im
            )
            var res0_im = (
                v0r * M[0][0].im
                + v0i * M[0][0].re
                + v1r * M[0][1].im
                + v1i * M[0][1].re
                + v2r * M[0][2].im
                + v2i * M[0][2].re
                + v3r * M[0][3].im
                + v3i * M[0][3].re
            )
            var res1_re = (
                v0r * M[1][0].re
                - v0i * M[1][0].im
                + v1r * M[1][1].re
                - v1i * M[1][1].im
                + v2r * M[1][2].re
                - v2i * M[1][2].im
                + v3r * M[1][3].re
                - v3i * M[1][3].im
            )
            var res1_im = (
                v0r * M[1][0].im
                + v0i * M[1][0].re
                + v1r * M[1][1].im
                + v1i * M[1][1].re
                + v2r * M[1][2].im
                + v2i * M[1][2].re
                + v3r * M[1][3].im
                + v3i * M[1][3].re
            )
            var res2_re = (
                v0r * M[2][0].re
                - v0i * M[2][0].im
                + v1r * M[2][1].re
                - v1i * M[2][1].im
                + v2r * M[2][2].re
                - v2i * M[2][2].im
                + v3r * M[2][3].re
                - v3i * M[2][3].im
            )
            var res2_im = (
                v0r * M[2][0].im
                + v0i * M[2][0].re
                + v1r * M[2][1].im
                + v1i * M[2][1].re
                + v2r * M[2][2].im
                + v2i * M[2][2].re
                + v3r * M[2][3].im
                + v3i * M[2][3].re
            )
            var res3_re = (
                v0r * M[3][0].re
                - v0i * M[3][0].im
                + v1r * M[3][1].re
                - v1i * M[3][1].im
                + v2r * M[3][2].re
                - v2i * M[3][2].im
                + v3r * M[3][3].re
                - v3i * M[3][3].im
            )
            var res3_im = (
                v0r * M[3][0].im
                + v0i * M[3][0].re
                + v1r * M[3][1].im
                + v1i * M[3][1].re
                + v2r * M[3][2].im
                + v2i * M[3][2].re
                + v3r * M[3][3].im
                + v3i * M[3][3].re
            )

            ptr_re.store(i0_idx, res0_re)
            ptr_im.store(i0_idx, res0_im)
            ptr_re.store(i1_idx, res1_re)
            ptr_im.store(i1_idx, res1_im)
            ptr_re.store(i2_idx, res2_re)
            ptr_im.store(i2_idx, res2_im)
            ptr_re.store(i3_idx, res3_re)
            ptr_im.store(i3_idx, res3_im)
        else:
            # Non-contiguous SIMD (Gathers + Manual Scatter)
            var v0r = ptr_re.gather(idx0)
            var v0i = ptr_im.gather(idx0)
            var v1r = ptr_re.gather(idx1)
            var v1i = ptr_im.gather(idx1)
            var v2r = ptr_re.gather(idx2)
            var v2i = ptr_im.gather(idx2)
            var v3r = ptr_re.gather(idx3)
            var v3i = ptr_im.gather(idx3)

            var res0_re = (
                v0r * M[0][0].re
                - v0i * M[0][0].im
                + v1r * M[0][1].re
                - v1i * M[0][1].im
                + v2r * M[0][2].re
                - v2i * M[0][2].im
                + v3r * M[0][3].re
                - v3i * M[0][3].im
            )
            var res0_im = (
                v0r * M[0][0].im
                + v0i * M[0][0].re
                + v1r * M[0][1].im
                + v1i * M[0][1].re
                + v2r * M[0][2].im
                + v2i * M[0][2].re
                + v3r * M[0][3].im
                + v3i * M[0][3].re
            )
            var res1_re = (
                v0r * M[1][0].re
                - v0i * M[1][0].im
                + v1r * M[1][1].re
                - v1i * M[1][1].im
                + v2r * M[1][2].re
                - v2i * M[1][2].im
                + v3r * M[1][3].re
                - v3i * M[1][3].im
            )
            var res1_im = (
                v0r * M[1][0].im
                + v0i * M[1][0].re
                + v1r * M[1][1].im
                + v1i * M[1][1].re
                + v2r * M[1][2].im
                + v2i * M[1][2].re
                + v3r * M[1][3].im
                + v3i * M[1][3].re
            )
            var res2_re = (
                v0r * M[2][0].re
                - v0i * M[2][0].im
                + v1r * M[2][1].re
                - v1i * M[2][1].im
                + v2r * M[2][2].re
                - v2i * M[2][2].im
                + v3r * M[2][3].re
                - v3i * M[2][3].im
            )
            var res2_im = (
                v0r * M[2][0].im
                + v0i * M[2][0].re
                + v1r * M[2][1].im
                + v1i * M[2][1].re
                + v2r * M[2][2].im
                + v2i * M[2][2].re
                + v3r * M[2][3].im
                + v3i * M[2][3].re
            )
            var res3_re = (
                v0r * M[3][0].re
                - v0i * M[3][0].im
                + v1r * M[3][1].re
                - v1i * M[3][1].im
                + v2r * M[3][2].re
                - v2i * M[3][2].im
                + v3r * M[3][3].re
                - v3i * M[3][3].im
            )
            var res3_im = (
                v0r * M[3][0].im
                + v0i * M[3][0].re
                + v1r * M[3][1].im
                + v1i * M[3][1].re
                + v2r * M[3][2].im
                + v2i * M[3][2].re
                + v3r * M[3][3].im
                + v3i * M[3][3].re
            )

            for k in range(w):
                ptr_re[Int(idx0[k])] = res0_re[k]
                ptr_im[Int(idx0[k])] = res0_im[k]
                ptr_re[Int(idx1[k])] = res1_re[k]
                ptr_im[Int(idx1[k])] = res1_im[k]
                ptr_re[Int(idx2[k])] = res2_re[k]
                ptr_im[Int(idx2[k])] = res2_im[k]
                ptr_re[Int(idx3[k])] = res3_re[k]
                ptr_im[Int(idx3[k])] = res3_im[k]
    else:
        # Scalar logic (w=1)
        var i0 = Int(idx0[0])
        var i1 = Int(idx1[0])
        var i2 = Int(idx2[0])
        var i3 = Int(idx3[0])
        var v0r = ptr_re[i0]
        var v0i = ptr_im[i0]
        var v1r = ptr_re[i1]
        var v1i = ptr_im[i1]
        var v2r = ptr_re[i2]
        var v2i = ptr_im[i2]
        var v3r = ptr_re[i3]
        var v3i = ptr_im[i3]

        var res0_re = (
            v0r * M[0][0].re
            - v0i * M[0][0].im
            + v1r * M[0][1].re
            - v1i * M[0][1].im
            + v2r * M[0][2].re
            - v2i * M[0][2].im
            + v3r * M[0][3].re
            - v3i * M[0][3].im
        )
        var res0_im = (
            v0r * M[0][0].im
            + v0i * M[0][0].re
            + v1r * M[0][1].im
            + v1i * M[0][1].re
            + v2r * M[0][2].im
            + v2i * M[0][2].re
            + v3r * M[0][3].im
            + v3i * M[0][3].re
        )
        var res1_re = (
            v0r * M[1][0].re
            - v0i * M[1][0].im
            + v1r * M[1][1].re
            - v1i * M[1][1].im
            + v2r * M[1][2].re
            - v2i * M[1][2].im
            + v3r * M[1][3].re
            - v3i * M[1][3].im
        )
        var res1_im = (
            v0r * M[1][0].im
            + v0i * M[1][0].re
            + v1r * M[1][1].im
            + v1i * M[1][1].re
            + v2r * M[1][2].im
            + v2i * M[1][2].re
            + v3r * M[1][3].im
            + v3i * M[1][3].re
        )
        var res2_re = (
            v0r * M[2][0].re
            - v0i * M[2][0].im
            + v1r * M[2][1].re
            - v1i * M[2][1].im
            + v2r * M[2][2].re
            - v2i * M[2][2].im
            + v3r * M[2][3].re
            - v3i * M[2][3].im
        )
        var res2_im = (
            v0r * M[2][0].im
            + v0i * M[2][0].re
            + v1r * M[2][1].im
            + v1i * M[2][1].re
            + v2r * M[2][2].im
            + v2i * M[2][2].re
            + v3r * M[2][3].im
            + v3i * M[2][3].re
        )
        var res3_re = (
            v0r * M[3][0].re
            - v0i * M[3][0].im
            + v1r * M[3][1].re
            - v1i * M[3][1].im
            + v2r * M[3][2].re
            - v2i * M[3][2].im
            + v3r * M[3][3].re
            - v3i * M[3][3].im
        )
        var res3_im = (
            v0r * M[3][0].im
            + v0i * M[3][0].re
            + v1r * M[3][1].im
            + v1i * M[3][1].re
            + v2r * M[3][2].im
            + v2i * M[3][2].re
            + v3r * M[3][3].im
            + v3i * M[3][3].re
        )

        ptr_re[i0] = res0_re
        ptr_im[i0] = res0_im
        ptr_re[i1] = res1_re
        ptr_im[i1] = res1_im
        ptr_re[i2] = res2_re
        ptr_im[i2] = res2_im
        ptr_re[i3] = res3_re
        ptr_im[i3] = res3_im


fn execute_col_col_fusion(
    mut state: QuantumState,
    transformations: List[Transformation],
    col_bits: Int,
) raises:
    """Intra-row fusion: Both qubits are within columns."""
    var t0 = transformations[0]
    var t1 = transformations[1]

    var q_high = max(get_target(t0), get_target(t1))
    var q_low = min(get_target(t0), get_target(t1))

    var M0 = get_as_matrix4x4(t0, q_high, q_low)
    var M1 = get_as_matrix4x4(t1, q_high, q_low)
    var M = matmul_matrix4x4(M1, M0)

    var n = Int(log2(Float64(len(state))))
    var num_rows = 1 << (n - col_bits)
    var row_size = 1 << col_bits

    var s_high = 1 << q_high
    var s_low = 1 << q_low

    var base_re = state.re.unsafe_ptr()
    var base_im = state.im.unsafe_ptr()

    @parameter
    fn process_row(row: Int):
        var row_offset = row * row_size

        @parameter
        fn vectorized_butterfly[w: Int](i: Int):
            var idx = SIMD[DType.int64, w](0)
            for k in range(w):
                idx[k] = i + k

            idx = (idx & (s_low - 1)) | ((idx & ~(s_low - 1)) << 1)
            idx = (idx & (s_high - 1)) | ((idx & ~(s_high - 1)) << 1)

            var idx0 = row_offset + idx
            var idx1 = idx0 | s_low
            var idx2 = idx0 | s_high
            var idx3 = idx1 | s_high

            _apply_M4_simd[w](
                base_re,
                base_im,
                idx0,
                idx1,
                idx2,
                idx3,
                M,
                is_contiguous=(s_low >= w),
            )

        if s_low >= simd_width:
            vectorize[vectorized_butterfly, simd_width](row_size // 4)
        else:
            vectorize[vectorized_butterfly, 1](row_size // 4)

    parallelize[process_row](num_rows)


fn execute_row_row_fusion(
    mut state: QuantumState,
    transformations: List[Transformation],
    col_bits: Int,
) raises:
    """Inter-row fusion: Both qubits are across rows."""
    var t0 = transformations[0]
    var t1 = transformations[1]

    var q_high = max(get_target(t0), get_target(t1))
    var q_low = min(get_target(t0), get_target(t1))

    var M0 = get_as_matrix4x4(t0, q_high, q_low)
    var M1 = get_as_matrix4x4(t1, q_high, q_low)
    var M = matmul_matrix4x4(M1, M0)

    var n = Int(log2(Float64(len(state))))
    var num_rows = 1 << (n - col_bits)
    var row_size = 1 << col_bits

    var t_high = q_high - col_bits
    var t_low = q_low - col_bits
    var r_stride_high = 1 << t_high
    var r_stride_low = 1 << t_low

    var base_re = state.re.unsafe_ptr()
    var base_im = state.im.unsafe_ptr()

    @parameter
    fn process_column[w: Int](col_simd: Int):
        var col_vec = SIMD[DType.int64, w](0)
        for k in range(w):
            col_vec[k] = col_simd + k

        for i in range(num_rows // 4):
            # Insert bits at t_low and t_high in row_index
            var row_idx_base = i
            var r_idx = row_idx_base
            r_idx = (r_idx & (r_stride_low - 1)) | (
                (r_idx & ~(r_stride_low - 1)) << 1
            )
            r_idx = (r_idx & (r_stride_high - 1)) | (
                (r_idx & ~(r_stride_high - 1)) << 1
            )

            var r_offset = r_idx * row_size
            var idx0 = r_offset + col_vec
            var idx1 = (r_idx | r_stride_low) * row_size + col_vec
            var idx2 = (r_idx | r_stride_high) * row_size + col_vec
            var idx3 = (
                r_idx | r_stride_low | r_stride_high
            ) * row_size + col_vec

            # Row-row columns are ALWAYS contiguous
            _apply_M4_simd[w](
                base_re, base_im, idx0, idx1, idx2, idx3, M, is_contiguous=True
            )

    vectorize[process_column, simd_width](row_size)


fn execute_row_col_fusion(
    mut state: QuantumState,
    transformations: List[Transformation],
    col_bits: Int,
) raises:
    """Combined fusion: One qubit in row, one in col."""
    var t0 = transformations[0]
    var t1 = transformations[1]

    var target0 = get_target(t0)
    var target1 = get_target(t1)

    var q_row: Int
    var q_col: Int
    var g_row: Gate
    var g_col: Gate

    if target0 >= col_bits:
        q_row = target0
        g_row = get_gate(t0)
        q_col = target1
        g_col = get_gate(t1)
    else:
        q_row = target1
        g_row = get_gate(t1)
        q_col = target0
        g_col = get_gate(t0)

    var M = compute_kron_product(g_row, g_col)

    var n = Int(log2(Float64(len(state))))
    var num_rows = 1 << (n - col_bits)
    var row_size = 1 << col_bits

    var r_bit = q_row - col_bits
    var r_stride = 1 << r_bit
    var c_stride = 1 << q_col

    var base_re = state.re.unsafe_ptr()
    var base_im = state.im.unsafe_ptr()

    @parameter
    fn process_grid_unit[w: Int](idx_simd: Int):
        var idx_vec = SIMD[DType.int64, w](0)
        for k in range(w):
            idx_vec[k] = idx_simd + k

        var col = idx_vec % (row_size // 2)
        col = (col & (c_stride - 1)) | ((col & ~(c_stride - 1)) << 1)

        var row = idx_vec // (row_size // 2)
        row = (row & (r_stride - 1)) | ((row & ~(r_stride - 1)) << 1)

        var r_offset = row * row_size
        var idx0 = r_offset + col
        var idx1 = idx0 | c_stride
        var idx2 = (row | r_stride) * row_size + col
        var idx3 = idx2 | c_stride

        _apply_M4_simd[w](
            base_re,
            base_im,
            idx0,
            idx1,
            idx2,
            idx3,
            M,
            is_contiguous=(c_stride >= w),
        )

    if c_stride >= simd_width and (row_size // 2) >= simd_width:
        vectorize[process_grid_unit, simd_width](
            (num_rows // 2) * (row_size // 2)
        )
    else:
        vectorize[process_grid_unit, 1]((num_rows // 2) * (row_size // 2))
