from memory import UnsafePointer
from butterfly.core.state import QuantumState, simd_width
from butterfly.utils.bit_utils import insert_zero_bit
from butterfly.core.types import FloatType
from algorithm import vectorize
import math


@always_inline
fn transform_row_fused_hh_ptr[
    simd_width: Int
](
    ptr_re_in: UnsafePointer[FloatType],
    ptr_im_in: UnsafePointer[FloatType],
    row: Int,
    row_size: Int,
    t1: Int,
    t2: Int,
):
    """Specialized row-local kernel for H + H fusion."""
    var low_pos = t1
    var high_pos = t2
    if t1 > t2:
        low_pos = t2
        high_pos = t1

    var stride_low = 1 << low_pos
    var stride_high = 1 << high_pos
    var num_base_pairs = row_size >> 2
    var scale: FloatType = 0.5
    var row_off = row * row_size

    if stride_low >= simd_width:

        @parameter
        fn inner_simd[w: Int](k: Int):
            var p_re = UnsafePointer[FloatType](ptr_re_in.address)
            var p_im = UnsafePointer[FloatType](ptr_im_in.address)

            var i = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(i, high_pos)
            var idx0 = row_off + idx0_local

            var r0 = p_re.load[width=w](idx0)
            var i0 = p_im.load[width=w](idx0)
            var r1 = p_re.load[width=w](idx0 + stride_low)
            var i1 = p_im.load[width=w](idx0 + stride_low)
            var r2 = p_re.load[width=w](idx0 + stride_high)
            var i2 = p_im.load[width=w](idx0 + stride_high)
            var r3 = p_re.load[width=w](idx0 + (stride_low + stride_high))
            var i3 = p_im.load[width=w](idx0 + (stride_low + stride_high))

            var s01_re = r0 + r1
            var s01_im = i0 + i1
            var d01_re = r0 - r1
            var d01_im = i0 - i1
            var s23_re = r2 + r3
            var s23_im = i2 + i3
            var d23_re = r2 - r3
            var d23_im = i2 - i3

            p_re.store(idx0, (s01_re + s23_re) * scale)
            p_im.store(idx0, (s01_im + s23_im) * scale)
            p_re.store(idx0 + stride_low, (d01_re + d23_re) * scale)
            p_im.store(idx0 + stride_low, (d01_im + d23_im) * scale)
            p_re.store(idx0 + stride_high, (s01_re - s23_re) * scale)
            p_im.store(idx0 + stride_high, (s01_im - s23_im) * scale)
            p_re.store(
                idx0 + (stride_low + stride_high), (d01_re - d23_re) * scale
            )
            p_im.store(
                idx0 + (stride_low + stride_high), (d01_im - d23_im) * scale
            )

        vectorize[inner_simd, simd_width](num_base_pairs)
    else:
        var p_re = UnsafePointer[FloatType](ptr_re_in.address)
        var p_im = UnsafePointer[FloatType](ptr_im_in.address)
        for k in range(num_base_pairs):
            var i = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(i, high_pos)
            var idx0 = row_off + idx0_local
            var idx1 = idx0 + stride_low
            var idx2 = idx0 + stride_high
            var idx3 = idx1 + stride_high

            var r0 = p_re.load[width=1](idx0)
            var i0 = p_im.load[width=1](idx0)
            var r1 = p_re.load[width=1](idx1)
            var i1 = p_im.load[width=1](idx1)
            var r2 = p_re.load[width=1](idx2)
            var i2 = p_im.load[width=1](idx2)
            var r3 = p_re.load[width=1](idx3)
            var i3 = p_im.load[width=1](idx3)

            var s01_re = r0 + r1
            var s01_im = i0 + i1
            var d01_re = r0 - r1
            var d01_im = i0 - i1
            var s23_re = r2 + r3
            var s23_im = i2 + i3
            var d23_re = r2 - r3
            var d23_im = i2 - i3

            p_re.store(idx0, (s01_re + s23_re) * scale)
            p_im.store(idx0, (s01_im + s23_im) * scale)
            p_re.store(idx1, (d01_re + d23_re) * scale)
            p_im.store(idx1, (d01_im + d23_im) * scale)
            p_re.store(idx2, (s01_re - s23_re) * scale)
            p_im.store(idx2, (s01_im - s23_im) * scale)
            p_re.store(idx3, (d01_re - d23_re) * scale)
            p_im.store(idx3, (d01_im - d23_im) * scale)


@always_inline
fn transform_row_fused_hh_simd[
    simd_width: Int
](mut state: QuantumState, row: Int, row_size: Int, t1: Int, t2: Int):
    transform_row_fused_hh_ptr[simd_width](
        state.re.unsafe_ptr(), state.im.unsafe_ptr(), row, row_size, t1, t2
    )


@always_inline
fn transform_row_fused_hp_ptr[
    simd_width: Int
](
    ptr_re_in: UnsafePointer[FloatType],
    ptr_im_in: UnsafePointer[FloatType],
    row: Int,
    row_size: Int,
    th: Int,
    tp: Int,
    theta: FloatType,
):
    """Specialized row-local kernel for H + P fusion."""
    var row_off = row * row_size
    var low_pos = th if th < tp else tp
    var high_pos = tp if th < tp else th

    var num_base_pairs = row_size >> 2
    var sq_half: FloatType = 0.70710678118654757
    var cos_t: FloatType = math.cos(theta)
    var sin_t: FloatType = math.sin(theta)

    if low_pos >= simd_width:

        @parameter
        fn inner_simd[w: Int](k: Int):
            var p_re = UnsafePointer[FloatType](ptr_re_in.address)
            var p_im = UnsafePointer[FloatType](ptr_im_in.address)
            var i = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(i, high_pos)
            var idx0 = row_off + idx0_local
            var s_th = 1 << th
            var s_tp = 1 << tp

            var idx_0 = idx0
            var idx_h = idx0 + s_th
            var idx_p = idx0 + s_tp
            var idx_hp = idx0 + (s_th + s_tp)

            var r0 = p_re.load[width=w](idx_0)
            var i0 = p_im.load[width=w](idx_0)
            var rh = p_re.load[width=w](idx_h)
            var ih = p_im.load[width=w](idx_h)
            var rp = p_re.load[width=w](idx_p)
            var ip = p_im.load[width=w](idx_p)
            var rhp = p_re.load[width=w](idx_hp)
            var ihp = p_im.load[width=w](idx_hp)

            var n_r0 = (r0 + rh) * sq_half
            var n_i0 = (i0 + ih) * sq_half
            var n_rh = (r0 - rh) * sq_half
            var n_ih = (i0 - ih) * sq_half

            var n_rp = (rp + rhp) * sq_half
            var n_ip = (ip + ihp) * sq_half
            var n_rhp = (rp - rhp) * sq_half
            var n_ihp = (ip - ihp) * sq_half

            p_re.store(idx_0, n_r0)
            p_im.store(idx_0, n_i0)
            p_re.store(idx_h, n_rh)
            p_im.store(idx_h, n_ih)

            p_re.store(idx_p, n_rp * cos_t - n_ip * sin_t)
            p_im.store(idx_p, n_rp * sin_t + n_ip * cos_t)
            p_re.store(idx_hp, n_rhp * cos_t - n_ihp * sin_t)
            p_im.store(idx_hp, n_rhp * sin_t + n_ihp * cos_t)

        vectorize[inner_simd, simd_width](num_base_pairs)
    else:
        var p_re = UnsafePointer[FloatType](ptr_re_in.address)
        var p_im = UnsafePointer[FloatType](ptr_im_in.address)
        for k in range(num_base_pairs):
            var i = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(i, high_pos)
            var idx0 = row_off + idx0_local
            var s_th = 1 << th
            var s_tp = 1 << tp

            var idx_0 = idx0
            var idx_h = idx0 + s_th
            var idx_p = idx0 + s_tp
            var idx_hp = idx0 + (s_th + s_tp)

            var r0 = p_re.load[width=1](idx_0)
            var i0 = p_im.load[width=1](idx_0)
            var rh = p_re.load[width=1](idx_h)
            var ih = p_im.load[width=1](idx_h)
            var rp = p_re.load[width=1](idx_p)
            var ip = p_im.load[width=1](idx_p)
            var rhp = p_re.load[width=1](idx_hp)
            var ihp = p_im.load[width=1](idx_hp)

            var n_r0 = (r0 + rh) * sq_half
            var n_i0 = (i0 + ih) * sq_half
            var n_rh = (r0 - rh) * sq_half
            var n_ih = (i0 - ih) * sq_half
            var n_rp = (rp + rhp) * sq_half
            var n_ip = (ip + ihp) * sq_half
            var n_rhp = (rp - rhp) * sq_half
            var n_ihp = (ip - ihp) * sq_half

            p_re.store(idx_0, n_r0)
            p_im.store(idx_0, n_i0)
            p_re.store(idx_h, n_rh)
            p_im.store(idx_h, n_ih)
            p_re.store(idx_p, n_rp * cos_t - n_ip * sin_t)
            p_im.store(idx_p, n_rp * sin_t + n_ip * cos_t)
            p_re.store(idx_hp, n_rhp * cos_t - n_ihp * sin_t)
            p_im.store(idx_hp, n_rhp * sin_t + n_ihp * cos_t)


@always_inline
fn transform_row_fused_hp_simd[
    simd_width: Int
](
    mut state: QuantumState,
    row: Int,
    row_size: Int,
    th: Int,
    tp: Int,
    theta: FloatType,
):
    transform_row_fused_hp_ptr[simd_width](
        state.re.unsafe_ptr(),
        state.im.unsafe_ptr(),
        row,
        row_size,
        th,
        tp,
        theta,
    )


@always_inline
fn transform_row_fused_st_hp_ptr[
    simd_width: Int
](
    ptr_re_in: UnsafePointer[FloatType],
    ptr_im_in: UnsafePointer[FloatType],
    row: Int,
    row_size: Int,
    target: Int,
    theta: FloatType,
):
    """Row-local kernel for H + P on the same target."""
    var row_off = row * row_size
    var stride = 1 << target
    var num_pairs = row_size >> 1
    var sq_half: FloatType = 0.70710678118654757
    var cos_t: FloatType = math.cos(theta)
    var sin_t: FloatType = math.sin(theta)

    if stride >= simd_width:

        @parameter
        fn inner_simd[w: Int](k: Int):
            var p_re = UnsafePointer[FloatType](ptr_re_in.address)
            var p_im = UnsafePointer[FloatType](ptr_im_in.address)
            var idx0_local = insert_zero_bit(k, target)
            var idx0 = row_off + idx0_local
            var idx1 = idx0 + stride

            var r0 = p_re.load[width=w](idx0)
            var i0 = p_im.load[width=w](idx0)
            var r1 = p_re.load[width=w](idx1)
            var i1 = p_im.load[width=w](idx1)

            var n_r0 = (r0 + r1) * sq_half
            var n_i0 = (i0 + i1) * sq_half
            var n_r1 = (r0 - r1) * sq_half
            var n_i1 = (i0 - i1) * sq_half

            p_re.store(idx0, n_r0)
            p_im.store(idx0, n_i0)
            p_re.store(idx1, n_r1 * cos_t - n_i1 * sin_t)
            p_im.store(idx1, n_r1 * sin_t + n_i1 * cos_t)

        vectorize[inner_simd, simd_width](num_pairs)
    else:
        var p_re = UnsafePointer[FloatType](ptr_re_in.address)
        var p_im = UnsafePointer[FloatType](ptr_im_in.address)
        for k in range(num_pairs):
            var idx0_local = insert_zero_bit(k, target)
            var idx0 = row_off + idx0_local
            var idx1 = idx0 + stride
            var r0 = p_re.load[width=1](idx0)
            var i0 = p_im.load[width=1](idx0)
            var r1 = p_re.load[width=1](idx1)
            var i1 = p_im.load[width=1](idx1)

            var n_r0 = (r0 + r1) * sq_half
            var n_i0 = (i0 + i1) * sq_half
            var n_r1 = (r0 - r1) * sq_half
            var n_i1 = (i0 - i1) * sq_half

            p_re.store(idx0, n_r0)
            p_im.store(idx0, n_i0)
            p_re.store(idx1, n_r1 * cos_t - n_i1 * sin_t)
            p_im.store(idx1, n_r1 * sin_t + n_i1 * cos_t)


@always_inline
fn transform_row_fused_st_hp_simd[
    simd_width: Int
](
    mut state: QuantumState,
    row: Int,
    row_size: Int,
    target: Int,
    theta: FloatType,
):
    transform_row_fused_st_hp_ptr[simd_width](
        state.re.unsafe_ptr(),
        state.im.unsafe_ptr(),
        row,
        row_size,
        target,
        theta,
    )

@always_inline
fn transform_row_fused_pp_ptr[
    simd_width: Int
](
    ptr_re_in: UnsafePointer[FloatType],
    ptr_im_in: UnsafePointer[FloatType],
    row: Int,
    row_size: Int,
    t1: Int,
    t2: Int,
    theta1: FloatType,
    theta2: FloatType,
):
    """Specialized row-local kernel for P + P fusion."""
    var row_off = row * row_size
    var h_pos = t1
    var l_pos = t2
    var th1 = theta1
    var th2 = theta2
    if t2 > t1:
        h_pos = t2
        l_pos = t1
        th1 = theta2
        th2 = theta1

    var stride_low = 1 << l_pos
    var stride_high = 1 << h_pos
    var num_base_pairs = row_size >> 2
    var c1: FloatType = math.cos(th1)
    var s1: FloatType = math.sin(th1)
    var c2: FloatType = math.cos(th2)
    var s2: FloatType = math.sin(th2)
    var c12: FloatType = math.cos(th1 + th2)
    var s12: FloatType = math.sin(th1 + th2)

    if l_pos >= simd_width:

        @parameter
        fn inner_simd[w: Int](k: Int):
            var p_re = UnsafePointer[FloatType](ptr_re_in.address)
            var p_im = UnsafePointer[FloatType](ptr_im_in.address)
            var i = insert_zero_bit(k, l_pos)
            var idx0_local = insert_zero_bit(i, h_pos)
            var idx0 = row_off + idx0_local
            var idx1 = idx0 + stride_low
            var idx2 = idx0 + stride_high
            var idx3 = idx1 + stride_high

            var r1 = p_re.load[width=w](idx1)
            var i1 = p_im.load[width=w](idx1)
            p_re.store(idx1, r1 * c2 - i1 * s2)
            p_im.store(idx1, r1 * s2 + i1 * c2)

            var r2 = p_re.load[width=w](idx2)
            var i2 = p_im.load[width=w](idx2)
            p_re.store(idx2, r2 * c1 - i2 * s1)
            p_im.store(idx2, r2 * s1 + i2 * c1)

            var r3 = p_re.load[width=w](idx3)
            var i3 = p_im.load[width=w](idx3)
            p_re.store(idx3, r3 * c12 - i3 * s12)
            p_im.store(idx3, r3 * s12 + i3 * c12)

        vectorize[inner_simd, simd_width](num_base_pairs)
    else:
        var p_re = UnsafePointer[FloatType](ptr_re_in.address)
        var p_im = UnsafePointer[FloatType](ptr_im_in.address)
        for k in range(num_base_pairs):
            var i = insert_zero_bit(k, l_pos)
            var idx0_local = insert_zero_bit(i, h_pos)
            var idx0 = row_off + idx0_local
            var idx1 = idx0 + stride_low
            var idx2 = idx0 + stride_high
            var idx3 = idx1 + stride_high

            var r1 = p_re.load[width=1](idx1)
            var i1 = p_im.load[width=1](idx1)
            p_re.store(idx1, r1 * c2 - i1 * s2)
            p_im.store(idx1, r1 * s2 + i1 * c2)

            var r2 = p_re.load[width=1](idx2)
            var i2 = p_im.load[width=1](idx2)
            p_re.store(idx2, r2 * c1 - i2 * s1)
            p_im.store(idx2, r2 * s1 + i2 * c1)

            var r3 = p_re.load[width=1](idx3)
            var i3 = p_im.load[width=1](idx3)
            p_re.store(idx3, r3 * c12 - i3 * s12)
            p_im.store(idx3, r3 * s12 + i3 * c12)


@always_inline
fn transform_row_fused_pp_simd[
    simd_width: Int
](
    mut state: QuantumState,
    row: Int,
    row_size: Int,
    t1: Int,
    t2: Int,
    theta1: FloatType,
    theta2: FloatType,
):
    transform_row_fused_pp_ptr[simd_width](
        state.re.unsafe_ptr(),
        state.im.unsafe_ptr(),
        row,
        row_size,
        t1,
        t2,
        theta1,
        theta2,
    )


@always_inline
fn transform_row_fused_shared_c_pp_ptr[
    simd_width: Int
](
    ptr_re_in: UnsafePointer[FloatType],
    ptr_im_in: UnsafePointer[FloatType],
    row: Int,
    row_size: Int,
    control: Int,
    t1: Int,
    t2: Int,
    theta1: FloatType,
    theta2: FloatType,
):
    """Row-local kernel for CP + CP sharing a control bit."""
    var row_off = row * row_size
    var b1 = control
    var b2 = t1
    var b3 = t2
    if b1 > b2:
        var tmp = b1
        b1 = b2
        b2 = tmp
    if b2 > b3:
        var tmp = b2
        b2 = b3
        b3 = tmp
    if b1 > b2:
        var tmp = b1
        b1 = b2
        b2 = tmp

    var s_control = 1 << control
    var s1 = 1 << t1
    var s2 = 1 << t2
    var cos1: FloatType = math.cos(theta1)
    var sin1: FloatType = math.sin(theta1)
    var cos2: FloatType = math.cos(theta2)
    var sin2: FloatType = math.sin(theta2)
    var cos12: FloatType = math.cos(theta1 + theta2)
    var sin12: FloatType = math.sin(theta1 + theta2)
    var num_base_groups = row_size >> 3

    if b1 >= simd_width:

        @parameter
        fn inner_simd[w: Int](k: Int):
            var p_re = UnsafePointer[FloatType](ptr_re_in.address)
            var p_im = UnsafePointer[FloatType](ptr_im_in.address)
            var idx0_local = insert_zero_bit(
                insert_zero_bit(insert_zero_bit(k, b1), b2), b3
            )
            var idx0 = row_off + idx0_local
            var c_base = idx0 + s_control

            var i1 = c_base + s1
            var r1 = p_re.load[width=w](i1)
            var im1 = p_im.load[width=w](i1)
            p_re.store(i1, r1 * cos1 - im1 * sin1)
            p_im.store(i1, r1 * sin1 + im1 * cos1)

            var i2 = c_base + s2
            var r2 = p_re.load[width=w](i2)
            var im2 = p_im.load[width=w](i2)
            p_re.store(i2, r2 * cos2 - im2 * sin2)
            p_im.store(i2, r2 * sin2 + im2 * cos2)

            var i12 = c_base + s1 + s2
            var r12 = p_re.load[width=w](i12)
            var im12 = p_im.load[width=w](i12)
            p_re.store(i12, r12 * cos12 - im12 * sin12)
            p_im.store(i12, r12 * sin12 + im12 * cos12)

        vectorize[inner_simd, simd_width](num_base_groups)
    else:
        var p_re = UnsafePointer[FloatType](ptr_re_in.address)
        var p_im = UnsafePointer[FloatType](ptr_im_in.address)
        for k in range(num_base_groups):
            var idx0 = row_off + insert_zero_bit(
                insert_zero_bit(insert_zero_bit(k, b1), b2), b3
            )
            var c_base = idx0 + s_control
            var inst_i1 = c_base + s1
            var r1 = p_re.load[width=1](inst_i1)
            var im1 = p_im.load[width=1](inst_i1)
            p_re.store(inst_i1, r1 * cos1 - im1 * sin1)
            p_im.store(inst_i1, r1 * sin1 + im1 * cos1)

            var inst_i2 = c_base + s2
            var r2 = p_re.load[width=1](inst_i2)
            var im2 = p_im.load[width=1](inst_i2)
            p_re.store(inst_i2, r2 * cos2 - im2 * sin2)
            p_im.store(inst_i2, r2 * sin2 + im2 * cos2)

            var inst_i12 = c_base + s1 + s2
            var r12 = p_re.load[width=1](inst_i12)
            var im12 = p_im.load[width=1](inst_i12)
            p_re.store(inst_i12, r12 * cos12 - im12 * sin12)
            p_im.store(inst_i12, r12 * sin12 + im12 * cos12)


@always_inline
fn transform_row_fused_shared_c_pp_simd[
    simd_width: Int
](
    mut state: QuantumState,
    row: Int,
    row_size: Int,
    control: Int,
    t1: Int,
    t2: Int,
    theta1: FloatType,
    theta2: FloatType,
):
    transform_row_fused_shared_c_pp_ptr[simd_width](
        state.re.unsafe_ptr(),
        state.im.unsafe_ptr(),
        row,
        row_size,
        control,
        t1,
        t2,
        theta1,
        theta2,
    )
