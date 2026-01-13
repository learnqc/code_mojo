"""
execute_as_grid: Implementation of grid-style execution on standard states.
"""
from algorithm import parallelize
from butterfly.core.state import QuantumState
from butterfly.core.types import Gate, FloatType, sq_half_re
from butterfly.core.gates import *
from butterfly.utils.context import ExecContext

from butterfly.core.transformations_simd import c_transform_simd
from butterfly.core.transformations_simd_parallel import c_transform_p_simd_parallel


from math import cos, sin, pi

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
                    re_ptr[idx] = (r0 + r1) * sq_half_re
                    im_ptr[idx] = (i0 + i1) * sq_half_re
                    re_ptr[idx + t_stride] = (r0 - r1) * sq_half_re
                    im_ptr[idx + t_stride] = (i0 - i1) * sq_half_re
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
                    re_ptr[idx] = (r0 + r1) * sq_half_re
                    im_ptr[idx] = (i0 + i1) * sq_half_re
                    re_ptr[idx + t_stride] = (r0 - r1) * sq_half_re
                    im_ptr[idx + t_stride] = (i0 - i1) * sq_half_re


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
](mut state: QuantumState, row: Int, row_size: Int, stride: Int):
    var offset = row * row_size
    var re_ptr = state.re.unsafe_ptr() + offset
    var im_ptr = state.im.unsafe_ptr() + offset

    for k in range(0, row_size, 2 * stride):
        if stride >= simd_width:
            for i in range(0, stride, simd_width):
                var idx0 = k + i
                var idx1 = idx0 + stride
                var r0 = re_ptr.load[width=simd_width](idx0)
                var i0 = im_ptr.load[width=simd_width](idx0)
                var r1 = re_ptr.load[width=simd_width](idx1)
                var i1 = im_ptr.load[width=simd_width](idx1)

                re_ptr.store(idx0, (r0 + r1) * sq_half_re)
                im_ptr.store(idx0, (i0 + i1) * sq_half_re)
                re_ptr.store(idx1, (r0 - r1) * sq_half_re)
                im_ptr.store(idx1, (i0 - i1) * sq_half_re)
        else:
            for i in range(stride):
                var idx0 = k + i
                var idx1 = idx0 + stride
                var r0 = re_ptr[idx0]
                var i0 = im_ptr[idx0]
                var r1 = re_ptr[idx1]
                var i1 = im_ptr[idx1]
                re_ptr[idx0] = (r0 + r1) * sq_half_re
                im_ptr[idx0] = (i0 + i1) * sq_half_re
                re_ptr[idx1] = (r0 - r1) * sq_half_re
                im_ptr[idx1] = (i0 - i1) * sq_half_re


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

fn transform_column(mut state: QuantumState, num_rows: Int, row_size: Int, col: Int, gate: Gate, stride: Int):
    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im
    for k in range(num_rows // (2 * stride)):
        for row_idx in range(
            k * 2 * stride, k * 2 * stride + stride
        ):
            var idx0 = row_idx * row_size + col
            var idx1 = (row_idx + stride) * row_size + col
            var r0 = state.re[idx0]
            var i0 = state.im[idx0]
            var r1 = state.re[idx1]
            var i1 = state.im[idx1]
            state.re[idx0] = (
                g00_re * r0
                - g00_im * i0
                + g01_re * r1
                - g01_im * i1
            )
            state.im[idx0] = (
                g00_re * i0
                + g00_im * r0
                + g01_re * i1
                + g01_im * r1
            )
            state.re[idx1] = (
                g10_re * r0
                - g10_im * i0
                + g11_re * r1
                - g11_im * i1
            )
            state.im[idx1] = (
                g10_re * i0
                + g10_im * r0
                + g11_re * i1
                + g11_im * r1
            )

fn transform_column_simd[chunk_size: Int](re_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    im_ptr: UnsafePointer[FloatType, MutAnyOrigin], num_rows: Int, row_size: Int, 
    col_base: Int, gate:Gate, stride: Int):

    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im
    for k in range(num_rows // (2 * stride)):
        for row_idx in range(
            k * 2 * stride, k * 2 * stride + stride
        ):
            var idx0 = row_idx * row_size + col_base
            var idx1 = (
                row_idx + stride
            ) * row_size + col_base
            var r0 = re_ptr.load[width=chunk_size](idx0)
            var i0 = im_ptr.load[width=chunk_size](idx0)
            var r1 = re_ptr.load[width=chunk_size](idx1)
            var i1 = im_ptr.load[width=chunk_size](idx1)

            re_ptr.store(idx0,
                g00_re * r0
                - g00_im * i0
                + g01_re * r1
                - g01_im * i1
            )
            im_ptr.store(idx0,
                g00_re * i0
                + g00_im * r0
                + g01_re * i1
                + g01_im * r1
            )
            re_ptr.store(idx1,
                g10_re * r0
                - g10_im * i0
                + g11_re * r1
                - g11_im * i1
            )
            im_ptr.store(idx1,
                g10_re * i0
                + g10_im * r0
                + g11_re * i1
                + g11_im * r1
            )


fn transform_column_h_simd[chunk_size: Int](re_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    im_ptr: UnsafePointer[FloatType, MutAnyOrigin], num_rows: Int, row_size: Int, col_base: Int, stride: Int):

    for k in range(num_rows // (2 * stride)):
        for row_idx in range(
            k * 2 * stride, k * 2 * stride + stride
        ):
            var idx0 = row_idx * row_size + col_base
            var idx1 = (
                row_idx + stride
            ) * row_size + col_base
            var r0 = re_ptr.load[width=chunk_size](idx0)
            var i0 = im_ptr.load[width=chunk_size](idx0)
            var r1 = re_ptr.load[width=chunk_size](idx1)
            var i1 = im_ptr.load[width=chunk_size](idx1)
            re_ptr.store(idx0, (r0 + r1) * sq_half_re)
            im_ptr.store(idx0, (i0 + i1) * sq_half_re)
            re_ptr.store(idx1, (r0 - r1) * sq_half_re)
            im_ptr.store(idx1, (i0 - i1) * sq_half_re)

fn transform_h_grid[
    simd_width: Int = 8
](mut state: QuantumState, col_bits: Int, target: Int, ctx: ExecContext = ExecContext()) raises:
    transform_grid[simd_width](state, col_bits, target, H_Gate, ctx)

fn transform_p_grid[
    simd_width: Int = 8
](mut state: QuantumState, col_bits: Int, target: Int, theta: FloatType, ctx: ExecContext = ExecContext()) raises:
    transform_grid[simd_width](state, col_bits, target, P_Gate(theta), ctx)

fn transform_grid[simd_width: Int = 8
](
    mut state: QuantumState,
    col_bits: Int,
    target: Int,
    gate_info: GateInfo,
    ctx: ExecContext = ExecContext(),
) raises:
    from math import log2

    var with_simd = simd_width > 0
    var use_parallel = ctx.grid_use_parallel

    var n = Int(log2(Float64(len(state))))
    var num_rows = 1 << (n - col_bits)
    var row_size = 1 << col_bits

    var use_h = gate_info.kind == GateKind.H and ctx.simd_use_specialized_h
    var use_p = gate_info.kind == GateKind.P and ctx.simd_use_specialized_p

    if target < col_bits:
        # Operation within row - parallelize by row
        @parameter
        fn process_row(row: Int):
            if with_simd and row_size >= simd_width:
                if use_h:
                    transform_row_h_simd[simd_width](
                        state, row, row_size, 1 << target)
                elif use_p:
                    transform_row_p_simd[simd_width](
                        state, row, row_size, target, gate_info.arg.value())
                else:
                    transform_row_simd[simd_width](
                        state, row, row_size, target, gate_info.gate)
            else:
                transform_row(state, row, row_size, target, gate_info.gate)

        if use_parallel:
            parallelize[process_row](num_rows)
        else:
            for row in range(num_rows):
                process_row(row)
    else:
        # Operation across rows - parallelize by column
        var t_row = target - col_bits
        var stride = 1 << t_row
        alias chunk_size = simd_width

        if with_simd and row_size >= chunk_size:

            @parameter
            fn process_column_simd(chunk_idx: Int):
                var col_base = chunk_idx * chunk_size
                var re_ptr = state.re.unsafe_ptr()
                var im_ptr = state.im.unsafe_ptr()
                if use_h:
                    transform_column_h_simd[chunk_size](
                        re_ptr, im_ptr, num_rows, row_size, col_base, stride
                    )
                else:
                    # transform_column_p_simd[chunk_size](
                    #     re_ptr, im_ptr, num_rows, row_size, col_base, stride
                    # )
                    transform_column_simd[chunk_size](
                        re_ptr, im_ptr, num_rows, row_size, col_base, gate_info.gate, stride
                    )

            if use_parallel:
                parallelize[process_column_simd](row_size // chunk_size)
            else:
                for chunk_idx in range(row_size // chunk_size):
                    process_column_simd(chunk_idx)
        else:
            @parameter
            fn process_column(col: Int):
                transform_column(state, num_rows, row_size, col, gate_info.gate, stride)

            if use_parallel:
                parallelize[process_column](row_size)
            else:
                for col in range(row_size):
                    process_column(col)

fn c_transform_p_grid[
    simd_width: Int = 8
](mut state: QuantumState, col_bits: Int, control: Int, target: Int, theta: FloatType, ctx: ExecContext = ExecContext()) raises:
    c_transform_grid[simd_width](state, col_bits, control, target, P_Gate(theta), ctx)

fn c_transform_grid[
    simd_width: Int = 8
](
    mut state: QuantumState,
    col_bits: Int,
    control: Int,
    target: Int,
    gate_info: GateInfo,
    ctx: ExecContext = ExecContext(),
) raises:
    from math import log2

    var with_simd = simd_width > 0
    var use_parallel = ctx.grid_use_parallel

    var n = Int(log2(Float64(len(state))))
    var num_rows = 1 << (n - col_bits)
    var row_size = 1 << col_bits

    var use_h = gate_info.kind == GateKind.H and ctx.simd_use_specialized_h
    var use_p = gate_info.kind == GateKind.P and ctx.simd_use_specialized_cp
    var gate_arg = FloatType(0)
    if gate_info.arg:
        gate_arg = gate_info.arg.value()

    # Check if both control and target are within rows (local)
    if control < col_bits and target < col_bits and not use_h and not use_p:
        c_transform_simd(
            state,
            control,
            target,
            gate_info.gate,
            gate_info.kind,
            gate_arg,
            ctx,
        )
        return

    if control < col_bits and target < col_bits:
        # Operation within row - parallelize by row
        @parameter
        fn process_row(row: Int):
            if with_simd and row_size >= simd_width:
                if use_h:
                    c_transform_row_h_simd[simd_width](
                        state, row, row_size, control, 1 << target)
                elif use_p:
                    c_transform_row_p_simd[simd_width](
                        state, row, row_size, control, target, gate_info.arg.value())
                else:
                    pass
                    # c_transform_row_simd[simd_width](
                    #     state, row, row_size, control, target, gate_info.gate)
            else:
                pass
                # c_transform_row(state, row, row_size, control, target, gate_info.gate)

        if use_parallel:
            parallelize[process_row](num_rows)
        else:
            for row in range(num_rows):
                process_row(row)
    else:
        # Operation across rows - parallelize by column
        var t_row = target - col_bits
        var stride = 1 << t_row
        alias chunk_size = simd_width

        if with_simd and row_size >= chunk_size:

            @parameter
            fn process_column_simd(chunk_idx: Int):
                # var col_base = chunk_idx * chunk_size
                # var re_ptr = state.re.unsafe_ptr()
                # var im_ptr = state.im.unsafe_ptr()
                if gate_info.kind == GateKind.P:
                    pass
                    # c_transform_simd(state, control, stride, gate_info.gate)
                else:
                    # transform_column_p_simd[chunk_size](
                    #     re_ptr, im_ptr, num_rows, row_size, col_base, stride
                    # )
                    c_transform_simd(
                        state,
                        control,
                        stride,
                        gate_info.gate,
                        gate_info.kind,
                        gate_arg,
                        ctx,
                    )
                    

            # parallelize[process_column_simd](row_size // chunk_size)
            if use_p and use_parallel:
                c_transform_p_simd_parallel(
                    state, control, target, gate_info.arg.value()
                )
            else:
                c_transform_simd(
                    state,
                    control,
                    target,
                    gate_info.gate,
                    gate_info.kind,
                    gate_arg,
                    ctx,
                )
        else:
            # @parameter
            # fn process_column(col: Int):
            #     transform_column(state, num_rows, row_size, col, gate_info.gate, stride)

            # parallelize[process_column](row_size)

            c_transform_simd(
                state,
                control,
                target,
                gate_info.gate,
                gate_info.kind,
                gate_arg,
                ctx,
            )
