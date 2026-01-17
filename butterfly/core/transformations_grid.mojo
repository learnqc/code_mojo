"""
execute_as_grid: Implementation of grid-style execution on standard states.
"""
from algorithm import parallelize
from butterfly.core.state import QuantumState
from butterfly.core.types import Gate, FloatType, sq_half_re
from butterfly.core.gates import *
from butterfly.utils.context import ExecContext

from butterfly.core.transformations_simd import c_transform_simd
from butterfly.core.transformations_simd_parallel import (
    c_transform_p_simd_parallel,
)


from math import cos, sin, pi

# L2 cache tile size for column operations (number of column elements per tile)
# Targets ~4KB per array per tile (256 * 8 bytes = 2KB re + 2KB im = 4KB)
# This keeps working set in L2 cache during strided row traversal
alias L2_TILE_COLS: Int = 256


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
    theta: FloatType,
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
    theta: FloatType,
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


fn transform_column(
    mut state: QuantumState,
    num_rows: Int,
    row_size: Int,
    col: Int,
    gate: Gate,
    stride: Int,
):
    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im
    for k in range(num_rows // (2 * stride)):
        for row_idx in range(k * 2 * stride, k * 2 * stride + stride):
            var idx0 = row_idx * row_size + col
            var idx1 = (row_idx + stride) * row_size + col
            var r0 = state.re[idx0]
            var i0 = state.im[idx0]
            var r1 = state.re[idx1]
            var i1 = state.im[idx1]
            state.re[idx0] = (
                g00_re * r0 - g00_im * i0 + g01_re * r1 - g01_im * i1
            )
            state.im[idx0] = (
                g00_re * i0 + g00_im * r0 + g01_re * i1 + g01_im * r1
            )
            state.re[idx1] = (
                g10_re * r0 - g10_im * i0 + g11_re * r1 - g11_im * i1
            )
            state.im[idx1] = (
                g10_re * i0 + g10_im * r0 + g11_re * i1 + g11_im * r1
            )


fn transform_column_simd[
    chunk_size: Int
](
    re_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    im_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    num_rows: Int,
    row_size: Int,
    col_base: Int,
    gate: Gate,
    stride: Int,
):
    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im
    for k in range(num_rows // (2 * stride)):
        for row_idx in range(k * 2 * stride, k * 2 * stride + stride):
            var idx0 = row_idx * row_size + col_base
            var idx1 = (row_idx + stride) * row_size + col_base
            var r0 = re_ptr.load[width=chunk_size](idx0)
            var i0 = im_ptr.load[width=chunk_size](idx0)
            var r1 = re_ptr.load[width=chunk_size](idx1)
            var i1 = im_ptr.load[width=chunk_size](idx1)

            re_ptr.store(
                idx0, g00_re * r0 - g00_im * i0 + g01_re * r1 - g01_im * i1
            )
            im_ptr.store(
                idx0, g00_re * i0 + g00_im * r0 + g01_re * i1 + g01_im * r1
            )
            re_ptr.store(
                idx1, g10_re * r0 - g10_im * i0 + g11_re * r1 - g11_im * i1
            )
            im_ptr.store(
                idx1, g10_re * i0 + g10_im * r0 + g11_re * i1 + g11_im * r1
            )


fn transform_column_h_simd[
    chunk_size: Int
](
    re_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    im_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    num_rows: Int,
    row_size: Int,
    col_base: Int,
    stride: Int,
):
    for k in range(num_rows // (2 * stride)):
        for row_idx in range(k * 2 * stride, k * 2 * stride + stride):
            var idx0 = row_idx * row_size + col_base
            var idx1 = (row_idx + stride) * row_size + col_base
            var r0 = re_ptr.load[width=chunk_size](idx0)
            var i0 = im_ptr.load[width=chunk_size](idx0)
            var r1 = re_ptr.load[width=chunk_size](idx1)
            var i1 = im_ptr.load[width=chunk_size](idx1)
            re_ptr.store(idx0, (r0 + r1) * sq_half_re)
            im_ptr.store(idx0, (i0 + i1) * sq_half_re)
            re_ptr.store(idx1, (r0 - r1) * sq_half_re)
            im_ptr.store(idx1, (i0 - i1) * sq_half_re)


fn transform_column_h_simd_tiled[
    chunk_size: Int
](
    re_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    im_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    num_rows: Int,
    row_size: Int,
    col_start: Int,
    col_end: Int,
    stride: Int,
):
    """Tiled Hadamard column operation for better L2 cache locality.

    Processes a tile of columns [col_start, col_end) across all rows.
    By iterating rows in the outer loop and columns in the inner loop,
    we improve cache reuse when row_size is large.
    """
    for k in range(num_rows // (2 * stride)):
        for row_idx in range(k * 2 * stride, k * 2 * stride + stride):
            var base0 = row_idx * row_size
            var base1 = (row_idx + stride) * row_size
            # Process all columns in this tile for this row pair
            for col_base in range(col_start, col_end, chunk_size):
                var idx0 = base0 + col_base
                var idx1 = base1 + col_base
                var r0 = re_ptr.load[width=chunk_size](idx0)
                var i0 = im_ptr.load[width=chunk_size](idx0)
                var r1 = re_ptr.load[width=chunk_size](idx1)
                var i1 = im_ptr.load[width=chunk_size](idx1)
                re_ptr.store(idx0, (r0 + r1) * sq_half_re)
                im_ptr.store(idx0, (i0 + i1) * sq_half_re)
                re_ptr.store(idx1, (r0 - r1) * sq_half_re)
                im_ptr.store(idx1, (i0 - i1) * sq_half_re)


fn transform_column_simd_tiled[
    chunk_size: Int
](
    re_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    im_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    num_rows: Int,
    row_size: Int,
    col_start: Int,
    col_end: Int,
    gate: Gate,
    stride: Int,
):
    """Tiled generic column operation for better L2 cache locality."""
    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im

    for k in range(num_rows // (2 * stride)):
        for row_idx in range(k * 2 * stride, k * 2 * stride + stride):
            var base0 = row_idx * row_size
            var base1 = (row_idx + stride) * row_size
            for col_base in range(col_start, col_end, chunk_size):
                var idx0 = base0 + col_base
                var idx1 = base1 + col_base
                var r0 = re_ptr.load[width=chunk_size](idx0)
                var i0 = im_ptr.load[width=chunk_size](idx0)
                var r1 = re_ptr.load[width=chunk_size](idx1)
                var i1 = im_ptr.load[width=chunk_size](idx1)

                re_ptr.store(
                    idx0,
                    g00_re * r0 - g00_im * i0 + g01_re * r1 - g01_im * i1,
                )
                im_ptr.store(
                    idx0,
                    g00_re * i0 + g00_im * r0 + g01_re * i1 + g01_im * r1,
                )
                re_ptr.store(
                    idx1,
                    g10_re * r0 - g10_im * i0 + g11_re * r1 - g11_im * i1,
                )
                im_ptr.store(
                    idx1,
                    g10_re * i0 + g10_im * r0 + g11_re * i1 + g11_im * r1,
                )


fn transform_column_fused_hp_simd_tiled[
    chunk_size: Int
](
    re_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    im_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    num_rows: Int,
    row_size: Int,
    col_start: Int,
    col_end: Int,
    target_h: Int,  # Cross-row qubit for H (relative to col_bits)
    target_p: Int,  # Cross-row qubit for P (relative to col_bits)
    theta: FloatType,
):
    """Tiled fused H+P column operation for cross-row targets.

    Applies H on target_h and P on target_p in a single traversal.
    Both targets must be in the row-bits region (>= col_bits for the full qubit index).
    """
    var stride_h = 1 << target_h
    var stride_p = 1 << target_p
    var cos_t = cos(theta)
    var sin_t = sin(theta)

    # Determine iteration pattern based on which stride is larger
    if target_h < target_p:
        # H operates on smaller stride, P on larger
        for k in range(num_rows // (2 * stride_p)):
            for row_idx in range(k * 2 * stride_p, k * 2 * stride_p + stride_p):
                # Four row indices: base, base+stride_h, base+stride_p, base+stride_h+stride_p
                var base0 = (
                    row_idx & ~(stride_h - 1)
                ) * row_size  # row with h_bit=0, p_bit=0
                var base_h = (
                    (row_idx & ~(stride_h - 1)) | stride_h
                ) * row_size  # h_bit=1, p_bit=0
                var base_p = (
                    (row_idx | stride_p) & ~(stride_h - 1)
                ) * row_size  # h_bit=0, p_bit=1
                var base_hp = (
                    (row_idx | stride_p) | stride_h
                ) * row_size  # h_bit=1, p_bit=1

                # Skip if row_idx has h_bit set (we handle pairs starting at h_bit=0)
                if (row_idx >> target_h) & 1:
                    continue

                for col_base in range(col_start, col_end, chunk_size):
                    var idx0 = base0 + col_base
                    var idx_h = base_h + col_base
                    var idx_p = base_p + col_base
                    var idx_hp = base_hp + col_base

                    # Load all 4 amplitudes
                    var r00 = re_ptr.load[width=chunk_size](idx0)
                    var i00 = im_ptr.load[width=chunk_size](idx0)
                    var r10 = re_ptr.load[width=chunk_size](idx_h)
                    var i10 = im_ptr.load[width=chunk_size](idx_h)
                    var r01 = re_ptr.load[width=chunk_size](idx_p)
                    var i01 = im_ptr.load[width=chunk_size](idx_p)
                    var r11 = re_ptr.load[width=chunk_size](idx_hp)
                    var i11 = im_ptr.load[width=chunk_size](idx_hp)

                    # Apply H on h_bit: |0⟩→(|0⟩+|1⟩)/√2, |1⟩→(|0⟩-|1⟩)/√2
                    var new_r00 = (r00 + r10) * sq_half_re
                    var new_i00 = (i00 + i10) * sq_half_re
                    var new_r10 = (r00 - r10) * sq_half_re
                    var new_i10 = (i00 - i10) * sq_half_re
                    var new_r01 = (r01 + r11) * sq_half_re
                    var new_i01 = (i01 + i11) * sq_half_re
                    var new_r11 = (r01 - r11) * sq_half_re
                    var new_i11 = (i01 - i11) * sq_half_re

                    # Apply P on p_bit (for p_bit=1 states): multiply by e^{i*theta}
                    var p_r01 = new_r01 * cos_t - new_i01 * sin_t
                    var p_i01 = new_r01 * sin_t + new_i01 * cos_t
                    var p_r11 = new_r11 * cos_t - new_i11 * sin_t
                    var p_i11 = new_r11 * sin_t + new_i11 * cos_t

                    # Store results
                    re_ptr.store(idx0, new_r00)
                    im_ptr.store(idx0, new_i00)
                    re_ptr.store(idx_h, new_r10)
                    im_ptr.store(idx_h, new_i10)
                    re_ptr.store(idx_p, p_r01)
                    im_ptr.store(idx_p, p_i01)
                    re_ptr.store(idx_hp, p_r11)
                    im_ptr.store(idx_hp, p_i11)
    else:
        # P operates on smaller stride, H on larger - just swap the order
        for k in range(num_rows // (2 * stride_h)):
            for row_idx in range(k * 2 * stride_h, k * 2 * stride_h + stride_h):
                if (row_idx >> target_p) & 1:
                    continue

                for col_base in range(col_start, col_end, chunk_size):
                    var base0 = row_idx * row_size
                    var base_p = (row_idx + stride_p) * row_size
                    var base_h = (row_idx + stride_h) * row_size
                    var base_hp = (row_idx + stride_p + stride_h) * row_size

                    var idx0 = base0 + col_base
                    var idx_p = base_p + col_base
                    var idx_h = base_h + col_base
                    var idx_hp = base_hp + col_base

                    var r00 = re_ptr.load[width=chunk_size](idx0)
                    var i00 = im_ptr.load[width=chunk_size](idx0)
                    var r01 = re_ptr.load[width=chunk_size](idx_p)
                    var i01 = im_ptr.load[width=chunk_size](idx_p)
                    var r10 = re_ptr.load[width=chunk_size](idx_h)
                    var i10 = im_ptr.load[width=chunk_size](idx_h)
                    var r11 = re_ptr.load[width=chunk_size](idx_hp)
                    var i11 = im_ptr.load[width=chunk_size](idx_hp)

                    # Apply H on h_bit
                    var new_r00 = (r00 + r10) * sq_half_re
                    var new_i00 = (i00 + i10) * sq_half_re
                    var new_r10 = (r00 - r10) * sq_half_re
                    var new_i10 = (i00 - i10) * sq_half_re
                    var new_r01 = (r01 + r11) * sq_half_re
                    var new_i01 = (i01 + i11) * sq_half_re
                    var new_r11 = (r01 - r11) * sq_half_re
                    var new_i11 = (i01 - i11) * sq_half_re

                    # Apply P on p_bit
                    var p_r01 = new_r01 * cos_t - new_i01 * sin_t
                    var p_i01 = new_r01 * sin_t + new_i01 * cos_t
                    var p_r11 = new_r11 * cos_t - new_i11 * sin_t
                    var p_i11 = new_r11 * sin_t + new_i11 * cos_t

                    re_ptr.store(idx0, new_r00)
                    im_ptr.store(idx0, new_i00)
                    re_ptr.store(idx_p, p_r01)
                    im_ptr.store(idx_p, p_i01)
                    re_ptr.store(idx_h, new_r10)
                    im_ptr.store(idx_h, new_i10)
                    re_ptr.store(idx_hp, p_r11)
                    im_ptr.store(idx_hp, p_i11)


fn transform_column_fused_hh_simd_tiled[
    chunk_size: Int
](
    re_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    im_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    num_rows: Int,
    row_size: Int,
    col_start: Int,
    col_end: Int,
    target_1: Int,  # First H qubit (relative to col_bits)
    target_2: Int,  # Second H qubit (relative to col_bits)
):
    """Tiled fused H+H column operation for cross-row targets."""
    var stride_1 = 1 << target_1
    var stride_2 = 1 << target_2
    var scale: FloatType = 0.5  # (1/√2)² = 0.5

    # Ensure target_1 < target_2 for consistent iteration
    var low = target_1
    var high = target_2
    var stride_low = stride_1
    var stride_high = stride_2
    if target_1 > target_2:
        low = target_2
        high = target_1
        stride_low = stride_2
        stride_high = stride_1

    for k in range(num_rows // (2 * stride_high)):
        for row_idx in range(
            k * 2 * stride_high, k * 2 * stride_high + stride_high
        ):
            # Skip if low bit is set (we handle pairs starting at low_bit=0)
            if (row_idx >> low) & 1:
                continue

            var base00 = row_idx * row_size
            var base01 = (row_idx + stride_low) * row_size
            var base10 = (row_idx + stride_high) * row_size
            var base11 = (row_idx + stride_low + stride_high) * row_size

            for col_base in range(col_start, col_end, chunk_size):
                var idx00 = base00 + col_base
                var idx01 = base01 + col_base
                var idx10 = base10 + col_base
                var idx11 = base11 + col_base

                var r00 = re_ptr.load[width=chunk_size](idx00)
                var i00 = im_ptr.load[width=chunk_size](idx00)
                var r01 = re_ptr.load[width=chunk_size](idx01)
                var i01 = im_ptr.load[width=chunk_size](idx01)
                var r10 = re_ptr.load[width=chunk_size](idx10)
                var i10 = im_ptr.load[width=chunk_size](idx10)
                var r11 = re_ptr.load[width=chunk_size](idx11)
                var i11 = im_ptr.load[width=chunk_size](idx11)

                # Apply HH: each output is combination of all 4 inputs
                # H⊗H: |00⟩→(|00⟩+|01⟩+|10⟩+|11⟩)/2
                var s_re = r00 + r01 + r10 + r11
                var s_im = i00 + i01 + i10 + i11
                var d01_re = r00 - r01 + r10 - r11
                var d01_im = i00 - i01 + i10 - i11
                var d10_re = r00 + r01 - r10 - r11
                var d10_im = i00 + i01 - i10 - i11
                var d11_re = r00 - r01 - r10 + r11
                var d11_im = i00 - i01 - i10 + i11

                re_ptr.store(idx00, s_re * scale)
                im_ptr.store(idx00, s_im * scale)
                re_ptr.store(idx01, d01_re * scale)
                im_ptr.store(idx01, d01_im * scale)
                re_ptr.store(idx10, d10_re * scale)
                im_ptr.store(idx10, d10_im * scale)
                re_ptr.store(idx11, d11_re * scale)
                im_ptr.store(idx11, d11_im * scale)


fn transform_column_fused_pp_simd_tiled[
    chunk_size: Int
](
    re_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    im_ptr: UnsafePointer[FloatType, MutAnyOrigin],
    num_rows: Int,
    row_size: Int,
    col_start: Int,
    col_end: Int,
    target_1: Int,  # First P qubit (relative to col_bits)
    target_2: Int,  # Second P qubit (relative to col_bits)
    theta_1: FloatType,
    theta_2: FloatType,
):
    """Tiled fused P+P column operation for cross-row targets."""
    var stride_1 = 1 << target_1
    var stride_2 = 1 << target_2
    var cos_1 = cos(theta_1)
    var sin_1 = sin(theta_1)
    var cos_2 = cos(theta_2)
    var sin_2 = sin(theta_2)
    # Combined phase for |11⟩ state
    var cos_12 = cos(theta_1 + theta_2)
    var sin_12 = sin(theta_1 + theta_2)

    var low = target_1
    var high = target_2
    var stride_low = stride_1
    var stride_high = stride_2
    var cos_low = cos_1
    var sin_low = sin_1
    var cos_high = cos_2
    var sin_high = sin_2
    if target_1 > target_2:
        low = target_2
        high = target_1
        stride_low = stride_2
        stride_high = stride_1
        cos_low = cos_2
        sin_low = sin_2
        cos_high = cos_1
        sin_high = sin_1

    for k in range(num_rows // (2 * stride_high)):
        for row_idx in range(
            k * 2 * stride_high, k * 2 * stride_high + stride_high
        ):
            if (row_idx >> low) & 1:
                continue

            var base00 = row_idx * row_size
            var base01 = (row_idx + stride_low) * row_size
            var base10 = (row_idx + stride_high) * row_size
            var base11 = (row_idx + stride_low + stride_high) * row_size

            for col_base in range(col_start, col_end, chunk_size):
                var idx00 = base00 + col_base
                var idx01 = base01 + col_base
                var idx10 = base10 + col_base
                var idx11 = base11 + col_base

                # |00⟩: no phase
                # |01⟩: phase by theta_low
                var r01 = re_ptr.load[width=chunk_size](idx01)
                var i01 = im_ptr.load[width=chunk_size](idx01)
                re_ptr.store(idx01, r01 * cos_low - i01 * sin_low)
                im_ptr.store(idx01, r01 * sin_low + i01 * cos_low)

                # |10⟩: phase by theta_high
                var r10 = re_ptr.load[width=chunk_size](idx10)
                var i10 = im_ptr.load[width=chunk_size](idx10)
                re_ptr.store(idx10, r10 * cos_high - i10 * sin_high)
                im_ptr.store(idx10, r10 * sin_high + i10 * cos_high)

                # |11⟩: phase by theta_1 + theta_2
                var r11 = re_ptr.load[width=chunk_size](idx11)
                var i11 = im_ptr.load[width=chunk_size](idx11)
                re_ptr.store(idx11, r11 * cos_12 - i11 * sin_12)
                im_ptr.store(idx11, r11 * sin_12 + i11 * cos_12)


fn transform_h_grid[
    simd_width: Int = 8
](
    mut state: QuantumState,
    col_bits: Int,
    target: Int,
    ctx: ExecContext = ExecContext(),
) raises:
    transform_grid[simd_width](state, col_bits, target, H_Gate, ctx)


fn transform_p_grid[
    simd_width: Int = 8
](
    mut state: QuantumState,
    col_bits: Int,
    target: Int,
    theta: FloatType,
    ctx: ExecContext = ExecContext(),
) raises:
    transform_grid[simd_width](state, col_bits, target, P_Gate(theta), ctx)


fn transform_grid[
    simd_width: Int = 8,
    tile_cols: Int = L2_TILE_COLS,
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

    var n = Int(log2(FloatType(len(state))))
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
                        state, row, row_size, 1 << target
                    )
                elif use_p:
                    transform_row_p_simd[simd_width](
                        state, row, row_size, target, gate_info.arg.value()
                    )
                else:
                    transform_row_simd[simd_width](
                        state, row, row_size, target, gate_info.gate
                    )
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
            var re_ptr = state.re.unsafe_ptr()
            var im_ptr = state.im.unsafe_ptr()

            # Calculate number of L2 tiles (min tile size is chunk_size)
            var tile_size = max(tile_cols, chunk_size)
            # Ensure tile_size is aligned to chunk_size
            tile_size = (tile_size // chunk_size) * chunk_size
            var num_tiles = (row_size + tile_size - 1) // tile_size

            @__copy_capture(re_ptr, im_ptr, tile_size)
            @parameter
            fn process_tile(tile_idx: Int):
                var col_start = tile_idx * tile_size
                var col_end = min(col_start + tile_size, row_size)
                # Ensure col_end is aligned to chunk_size
                col_end = (col_end // chunk_size) * chunk_size
                if col_end <= col_start:
                    return

                if use_h:
                    transform_column_h_simd_tiled[chunk_size](
                        re_ptr,
                        im_ptr,
                        num_rows,
                        row_size,
                        col_start,
                        col_end,
                        stride,
                    )
                else:
                    transform_column_simd_tiled[chunk_size](
                        re_ptr,
                        im_ptr,
                        num_rows,
                        row_size,
                        col_start,
                        col_end,
                        gate_info.gate,
                        stride,
                    )

            if use_parallel:
                parallelize[process_tile](num_tiles)
            else:
                for tile_idx in range(num_tiles):
                    process_tile(tile_idx)
        else:

            @parameter
            fn process_column(col: Int):
                transform_column(
                    state, num_rows, row_size, col, gate_info.gate, stride
                )

            if use_parallel:
                parallelize[process_column](row_size)
            else:
                for col in range(row_size):
                    process_column(col)


fn c_transform_p_grid[
    simd_width: Int = 8
](
    mut state: QuantumState,
    col_bits: Int,
    control: Int,
    target: Int,
    theta: FloatType,
    ctx: ExecContext = ExecContext(),
) raises:
    c_transform_grid[simd_width](
        state, col_bits, control, target, P_Gate(theta), ctx
    )


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

    var n = Int(log2(FloatType(len(state))))
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
                        state, row, row_size, control, 1 << target
                    )
                elif use_p:
                    c_transform_row_p_simd[simd_width](
                        state,
                        row,
                        row_size,
                        control,
                        target,
                        gate_info.arg.value(),
                    )
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
        stride: Int = 1 << t_row
        alias chunk_size = simd_width

        if with_simd and row_size >= chunk_size:

            @__copy_capture(stride)
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
