"""
execute_as_grid: Implementation of grid-style execution on standard states.
"""
from algorithm import parallelize
from butterfly.core.state import QuantumState
from butterfly.core.types import Gate
from butterfly.core.circuit import (
    QuantumCircuit,
    Transformation,
    GateTransformation,
    BitReversalTransformation,
    SingleControlGateTransformation,
    MultiControlGateTransformation,
    DiagonalTransformation,
)


from butterfly.core.gates import is_h, is_x, is_z, is_p, get_phase_angle
from butterfly.utils.config import get_workers
from math import cos, sin, iota


@always_inline
fn insert_bit(i: Int, p: Int) -> Int:
    """Insert a '1' at bit position 'p'."""
    var lower_mask = (1 << p) - 1
    var lower = i & lower_mask
    var upper = (i & ~lower_mask) << 1
    return upper | (1 << p) | lower


@always_inline
@always_inline
fn c_transform_row_h_simd[
    simd_width: Int = 8
](mut state: QuantumState, row: Int, row_size: Int, control: Int, target: Int):
    """Controlled Hadamard where both bits are within the row index."""
    var offset = row * row_size
    var re_ptr = state.re.unsafe_ptr() + offset
    var im_ptr = state.im.unsafe_ptr() + offset
    var c_stride = 1 << control
    var t_stride = 1 << target
    alias sq_half = 0.7071067811865476

    # Match SIMD v2 logic: iterate over blocks where control=1
    if target < control:
        # target < control: control bit divides into larger blocks
        var segments_per_block = c_stride // (2 * t_stride)
        for k in range(row_size // (2 * c_stride)):
            for j in range(segments_per_block):
                var sub_start = k * 2 * c_stride + c_stride + j * 2 * t_stride
                for m in range(t_stride):
                    var idx = sub_start + m
                    var r0 = re_ptr[idx]
                    var i0 = im_ptr[idx]
                    var r1 = re_ptr[idx + t_stride]
                    var i1 = im_ptr[idx + t_stride]
                    re_ptr[idx] = (r0 + r1) * sq_half
                    im_ptr[idx] = (i0 + i1) * sq_half
                    re_ptr[idx + t_stride] = (r0 - r1) * sq_half
                    im_ptr[idx + t_stride] = (i0 - i1) * sq_half
    else:
        # target > control: target bit divides into larger blocks
        var segments_per_block = t_stride // (2 * c_stride)
        for k in range(row_size // (2 * t_stride)):
            for p in range(segments_per_block):
                var p_start = k * 2 * t_stride + p * 2 * c_stride + c_stride
                for m in range(c_stride):
                    var idx = p_start + m
                    var r0 = re_ptr[idx]
                    var i0 = im_ptr[idx]
                    var r1 = re_ptr[idx + t_stride]
                    var i1 = im_ptr[idx + t_stride]
                    re_ptr[idx] = (r0 + r1) * sq_half
                    im_ptr[idx] = (i0 + i1) * sq_half
                    re_ptr[idx + t_stride] = (r0 - r1) * sq_half
                    im_ptr[idx + t_stride] = (i0 - i1) * sq_half


@always_inline
fn c_transform_row_p_simd[
    simd_width: Int = 8
](
    mut state: QuantumState,
    row: Int,
    row_size: Int,
    control: Int,
    target: Int,
    theta: Float64,
):
    """Controlled Phase where both bits are within the row index."""
    var offset = row * row_size
    var re_ptr = state.re.unsafe_ptr() + offset
    var im_ptr = state.im.unsafe_ptr() + offset
    var stride = 1 << target
    var c_stride = 1 << control
    var cos_t = cos(theta)
    var sin_t = sin(theta)

    # Match SIMD v2 logic: iterate over blocks where control=1
    if target < control:
        # target < control: control bit divides into larger blocks
        var segments_per_block = c_stride // (2 * stride)
        for k in range(row_size // (2 * c_stride)):
            for j in range(segments_per_block):
                var sub_start = k * 2 * c_stride + c_stride + j * 2 * stride
                for m in range(stride):
                    var idx = (
                        sub_start + m + stride
                    )  # Only target=1 is affected
                    var r1 = re_ptr[idx]
                    var i1 = im_ptr[idx]
                    re_ptr[idx] = r1 * cos_t - i1 * sin_t
                    im_ptr[idx] = r1 * sin_t + i1 * cos_t
    else:
        # target > control: target bit divides into larger blocks
        var segments_per_block = stride // (2 * c_stride)
        for k in range(row_size // (2 * stride)):
            for p in range(segments_per_block):
                var p_start = k * 2 * stride + p * 2 * c_stride + c_stride
                for m in range(c_stride):
                    var idx = p_start + m + stride  # Only target=1 is affected
                    var r1 = re_ptr[idx]
                    var i1 = im_ptr[idx]
                    re_ptr[idx] = r1 * cos_t - i1 * sin_t
                    im_ptr[idx] = r1 * sin_t + i1 * cos_t


@always_inline
fn transform_row_h_simd[
    simd_width: Int = 8
](mut state: QuantumState, row: Int, row_size: Int, target: Int):
    var offset = row * row_size
    var re_ptr = state.re.unsafe_ptr() + offset
    var im_ptr = state.im.unsafe_ptr() + offset
    var stride = 1 << target
    alias sq_half = 0.7071067811865476

    for k in range(0, row_size, 2 * stride):
        if stride >= simd_width:
            for i in range(0, stride, simd_width):
                var idx0 = k + i
                var idx1 = idx0 + stride
                var r0 = re_ptr.load[width=simd_width](idx0)
                var i0 = im_ptr.load[width=simd_width](idx0)
                var r1 = re_ptr.load[width=simd_width](idx1)
                var i1 = im_ptr.load[width=simd_width](idx1)

                re_ptr.store(idx0, (r0 + r1) * sq_half)
                im_ptr.store(idx0, (i0 + i1) * sq_half)
                re_ptr.store(idx1, (r0 - r1) * sq_half)
                im_ptr.store(idx1, (i0 - i1) * sq_half)
        else:
            for i in range(stride):
                var idx0 = k + i
                var idx1 = idx0 + stride
                var r0 = re_ptr[idx0]
                var i0 = im_ptr[idx0]
                var r1 = re_ptr[idx1]
                var i1 = im_ptr[idx1]
                re_ptr[idx0] = (r0 + r1) * sq_half
                im_ptr[idx0] = (i0 + i1) * sq_half
                re_ptr[idx1] = (r0 - r1) * sq_half
                im_ptr[idx1] = (i0 - i1) * sq_half


@always_inline
fn transform_row_x_simd[
    simd_width: Int = 8
](mut state: QuantumState, row: Int, row_size: Int, target: Int):
    var offset = row * row_size
    var re_ptr = state.re.unsafe_ptr() + offset
    var im_ptr = state.im.unsafe_ptr() + offset
    var stride = 1 << target

    for k in range(0, row_size, 2 * stride):
        if stride >= simd_width:
            for i in range(0, stride, simd_width):
                var idx0 = k + i
                var idx1 = idx0 + stride
                var r0 = re_ptr.load[width=simd_width](idx0)
                var i0 = im_ptr.load[width=simd_width](idx0)
                var r1 = re_ptr.load[width=simd_width](idx1)
                var i1 = im_ptr.load[width=simd_width](idx1)

                re_ptr.store(idx0, r1)
                im_ptr.store(idx0, i1)
                re_ptr.store(idx1, r0)
                im_ptr.store(idx1, i0)
        else:
            for i in range(stride):
                var idx0 = k + i
                var idx1 = idx0 + stride
                var r1 = re_ptr[idx1]
                var i1 = im_ptr[idx1]
                var r0 = re_ptr[idx0]
                var i0 = im_ptr[idx0]
                re_ptr[idx0] = r1
                im_ptr[idx0] = i1
                re_ptr[idx1] = r0
                im_ptr[idx1] = i0


@always_inline
fn transform_row_z_simd[
    simd_width: Int = 8
](mut state: QuantumState, row: Int, row_size: Int, target: Int):
    var offset = row * row_size
    var re_ptr = state.re.unsafe_ptr() + offset
    var im_ptr = state.im.unsafe_ptr() + offset
    var stride = 1 << target

    for k in range(0, row_size, 2 * stride):
        if stride >= simd_width:
            for i in range(0, stride, simd_width):
                var idx1 = k + i + stride
                re_ptr.store(idx1, -re_ptr.load[width=simd_width](idx1))
                im_ptr.store(idx1, -im_ptr.load[width=simd_width](idx1))
        else:
            for i in range(stride):
                var idx1 = k + i + stride
                re_ptr[idx1] = -re_ptr[idx1]
                im_ptr[idx1] = -im_ptr[idx1]


@always_inline
fn transform_row_p_simd[
    simd_width: Int = 8
](
    mut state: QuantumState,
    row: Int,
    row_size: Int,
    target: Int,
    theta: Float64,
):
    var offset = row * row_size
    var re_ptr = state.re.unsafe_ptr() + offset
    var im_ptr = state.im.unsafe_ptr() + offset
    var stride = 1 << target
    var cos_t = cos(theta)
    var sin_t = sin(theta)

    for k in range(0, row_size, 2 * stride):
        if stride >= simd_width:
            for i in range(0, stride, simd_width):
                var idx1 = k + i + stride
                var r1 = re_ptr.load[width=simd_width](idx1)
                var i1 = im_ptr.load[width=simd_width](idx1)
                re_ptr.store(idx1, r1 * cos_t - i1 * sin_t)
                im_ptr.store(idx1, r1 * sin_t + i1 * cos_t)
        else:
            for i in range(stride):
                var idx1 = k + i + stride
                var r1 = re_ptr[idx1]
                var i1 = im_ptr[idx1]
                re_ptr[idx1] = r1 * cos_t - i1 * sin_t
                im_ptr[idx1] = r1 * sin_t + i1 * cos_t


fn transform_row(
    mut state: QuantumState, row: Int, row_size: Int, target: Int, gate: Gate
):
    """Transform a single virtual row using scalar operations."""
    var offset = row * row_size
    var re_ptr = state.re.unsafe_ptr() + offset
    var im_ptr = state.im.unsafe_ptr() + offset

    # Extract gate components
    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im

    var stride = 1 << target

    # Process pairs
    for k in range(row_size // (2 * stride)):
        var base = k * 2 * stride
        for i in range(base, base + stride):
            var idx = i
            var idx1 = idx + stride

            # Load
            var re0 = re_ptr[idx]
            var im0 = im_ptr[idx]
            var re1 = re_ptr[idx1]
            var im1 = im_ptr[idx1]

            # Apply gate
            re_ptr[idx] = (
                g00_re * re0 - g00_im * im0 + g01_re * re1 - g01_im * im1
            )
            im_ptr[idx] = (
                g00_re * im0 + g00_im * re0 + g01_re * im1 + g01_im * re1
            )
            re_ptr[idx1] = (
                g10_re * re0 - g10_im * im0 + g11_re * re1 - g11_im * im1
            )
            im_ptr[idx1] = (
                g10_re * im0 + g10_im * re0 + g11_re * im1 + g11_im * re1
            )


fn transform_row_simd[
    simd_width: Int = 8
](mut state: QuantumState, row: Int, row_size: Int, target: Int, gate: Gate):
    """Transform a single virtual row using SIMD operations."""
    var offset = row * row_size
    var re_ptr = state.re.unsafe_ptr() + offset
    var im_ptr = state.im.unsafe_ptr() + offset

    # Extract gate components
    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im

    var stride = 1 << target

    for k in range(0, row_size, 2 * stride):
        if stride >= simd_width:
            # Full block SIMD
            for i in range(0, stride, simd_width):
                var idx0 = k + i
                var idx1 = idx0 + stride
                var r0 = re_ptr.load[width=simd_width](idx0)
                var i0 = im_ptr.load[width=simd_width](idx0)
                var r1 = re_ptr.load[width=simd_width](idx1)
                var i1 = im_ptr.load[width=simd_width](idx1)

                re_ptr.store(
                    idx0,
                    r0 * g00_re - i0 * g00_im + r1 * g01_re - i1 * g01_im,
                )
                im_ptr.store(
                    idx0,
                    r0 * g00_im + i0 * g00_re + r1 * g01_im + i1 * g01_re,
                )
                re_ptr.store(
                    idx1,
                    r0 * g10_re - i0 * g10_im + r1 * g11_re - i1 * g11_im,
                )
                im_ptr.store(
                    idx1,
                    r0 * g10_im + i0 * g10_re + r1 * g11_im + i1 * g11_re,
                )
        else:
            # Fallback for target < simd_width
            for i in range(stride):
                var idx0 = k + i
                var idx1 = idx0 + stride
                var r0 = re_ptr[idx0]
                var i0 = im_ptr[idx0]
                var r1 = re_ptr[idx1]
                var i1 = im_ptr[idx1]

                re_ptr[idx0] = (
                    r0 * g00_re - i0 * g00_im + r1 * g01_re - i1 * g01_im
                )
                im_ptr[idx0] = (
                    r0 * g00_im + i0 * g00_re + r1 * g01_im + i1 * g01_re
                )
                re_ptr[idx1] = (
                    r0 * g10_re - i0 * g10_im + r1 * g11_re - i1 * g11_im
                )
                im_ptr[idx1] = (
                    r0 * g10_im + i0 * g10_re + r1 * g11_im + i1 * g11_re
                )


fn dispatch_global_delegate(
    mut state: QuantumState,
    t: Transformation,
    num_qubits: Int,
):
    """Bridge runtime qubit count to compile-time N dispatch for global gates.

    This ensures that grid-based executors (v_grid, grid_fusion) always use
    the fastest specialized SIMD kernels available for any cross-row operation.
    """
    from butterfly.core.execute_simd_v2_dispatch import (
        dispatch_single_transformation_v2,
    )

    if num_qubits == 30:
        dispatch_single_transformation_v2[1 << 30](state, t)
    elif num_qubits == 29:
        dispatch_single_transformation_v2[1 << 29](state, t)
    elif num_qubits == 28:
        dispatch_single_transformation_v2[1 << 28](state, t)
    elif num_qubits == 27:
        dispatch_single_transformation_v2[1 << 27](state, t)
    elif num_qubits == 26:
        dispatch_single_transformation_v2[1 << 26](state, t)
    elif num_qubits == 25:
        dispatch_single_transformation_v2[1 << 25](state, t)
    elif num_qubits == 24:
        dispatch_single_transformation_v2[1 << 24](state, t)
    elif num_qubits == 23:
        dispatch_single_transformation_v2[1 << 23](state, t)
    elif num_qubits == 22:
        dispatch_single_transformation_v2[1 << 22](state, t)
    elif num_qubits == 21:
        dispatch_single_transformation_v2[1 << 21](state, t)
    elif num_qubits == 20:
        dispatch_single_transformation_v2[1 << 20](state, t)
    elif num_qubits == 19:
        dispatch_single_transformation_v2[1 << 19](state, t)
    elif num_qubits == 18:
        dispatch_single_transformation_v2[1 << 18](state, t)
    elif num_qubits == 17:
        dispatch_single_transformation_v2[1 << 17](state, t)
    elif num_qubits == 16:
        dispatch_single_transformation_v2[1 << 16](state, t)
    elif num_qubits == 15:
        dispatch_single_transformation_v2[1 << 15](state, t)
    elif num_qubits == 14:
        dispatch_single_transformation_v2[1 << 14](state, t)
    elif num_qubits == 13:
        dispatch_single_transformation_v2[1 << 13](state, t)
    elif num_qubits == 12:
        dispatch_single_transformation_v2[1 << 12](state, t)
    elif num_qubits == 11:
        dispatch_single_transformation_v2[1 << 11](state, t)
    elif num_qubits == 10:
        dispatch_single_transformation_v2[1 << 10](state, t)
    else:
        # Fallback to generic execution for small counts
        from butterfly.core.state import (
            transform,
            c_transform,
            mc_transform_interval,
        )
        from butterfly.core.circuit import (
            GateTransformation,
            SingleControlGateTransformation,
            MultiControlGateTransformation,
            BitReversalTransformation,
        )

        if t.isa[GateTransformation]():
            var gt = t[GateTransformation].copy()
            transform(state, gt.target, gt.gate)
        elif t.isa[SingleControlGateTransformation]():
            var sct = t[SingleControlGateTransformation].copy()
            c_transform(state, sct.control, sct.target, sct.gate)
        elif t.isa[MultiControlGateTransformation]():
            var mct = t[MultiControlGateTransformation].copy()
            mc_transform_interval(state, mct.controls, mct.target, mct.gate)
        elif t.isa[BitReversalTransformation]():
            from butterfly.core.state import bit_reverse_state

            bit_reverse_state(state)


fn execute_as_grid(
    mut state: QuantumState, mut circuit: QuantumCircuit, col_bits: Int = -1
) raises:
    """Execute circuit using grid strategy.

    Divides state into virtual rows (size 2^col_bits).
    Gates within rows are executed in parallel across rows.
    Gates across rows are delegated to optimized SIMD kernels.
    """
    var n = circuit.num_qubits
    var actual_col_bits = col_bits if col_bits >= 0 else (n - 3)

    var row_size = 1 << actual_col_bits
    var num_rows = 1 << (n - actual_col_bits)

    # Performance optimization: determine if SIMD is likely to help
    alias with_simd = True

    for i in range(len(circuit.transformations)):
        var t = circuit.transformations[i].copy()

        if t.isa[GateTransformation]():
            var gt = t[GateTransformation].copy()
            var target = gt.target
            var gate = gt.gate

            if target < actual_col_bits:
                # Operation within row - parallelize by row
                @parameter
                fn process_row(row: Int):
                    var stride = 1 << target
                    if with_simd and row_size >= simd_width:
                        if is_h(gate):
                            transform_row_h_simd[simd_width](
                                state, row, row_size, target
                            )
                        elif is_x(gate):
                            transform_row_x_simd[simd_width](
                                state, row, row_size, target
                            )
                        elif is_z(gate):
                            transform_row_z_simd[simd_width](
                                state, row, row_size, target
                            )
                        elif is_p(gate):
                            transform_row_p_simd[simd_width](
                                state,
                                row,
                                row_size,
                                target,
                                get_phase_angle(gate),
                            )
                        elif stride >= simd_width:
                            transform_row_simd[simd_width](
                                state, row, row_size, target, gate
                            )
                        else:
                            transform_row(state, row, row_size, target, gate)
                    else:
                        transform_row(state, row, row_size, target, gate)

                var row_workers = get_workers("v_grid_rows")
                if row_workers > 0:
                    parallelize[process_row](num_rows, row_workers)
                else:
                    parallelize[process_row](num_rows)
            else:
                # Operation across rows - Just delegate to global optimized Dispatcher (The Global Delegation Pattern)
                dispatch_global_delegate(state, t, n)

        elif t.isa[SingleControlGateTransformation]():
            var sct = t[SingleControlGateTransformation].copy()
            var control = sct.control
            var target = sct.target
            var gate = sct.gate

            # Check if both control and target are within rows (local)
            if control < actual_col_bits and target < actual_col_bits:
                # Row-local: use parallel row kernels for better cache locality
                @parameter
                fn process_controlled_row(row: Int):
                    if is_h(gate):
                        c_transform_row_h_simd[simd_width](
                            state, row, row_size, control, target
                        )
                    elif is_p(gate):
                        c_transform_row_p_simd[simd_width](
                            state,
                            row,
                            row_size,
                            control,
                            target,
                            get_phase_angle(gate),
                        )

                var row_workers = get_workers("v_grid_rows")
                if row_workers > 0:
                    parallelize[process_controlled_row](num_rows, row_workers)
                else:
                    parallelize[process_controlled_row](num_rows)
            else:
                # Operation across rows - Delegate to global dispatcher
                dispatch_global_delegate(state, t, n)
        elif t.isa[BitReversalTransformation]():
            from butterfly.core.state import bit_reverse_state

            bit_reverse_state(state)
