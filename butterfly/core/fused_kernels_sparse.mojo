from butterfly.core.state import QuantumState, simd_width
from butterfly.utils.bit_utils import insert_zero_bit
from algorithm import vectorize
from math import cos, sin


@always_inline
fn transform_row_fused_hh_simd[
    simd_width: Int
](mut state: QuantumState, row: Int, row_size: Int, t1: Int, t2: Int,):
    """Specialized sparse kernel for H + H fusion."""
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
    alias scale = 0.5

    if stride_low >= simd_width:

        @parameter
        fn inner_simd[w: Int](k: Int):
            var i = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(i, high_pos)
            var idx0 = row_offset + idx0_local
            var idx1 = idx0 + stride_low
            var idx2 = idx0 + stride_high
            var idx3 = idx1 + stride_high

            var r0 = ptr_re.load[width=w](idx0)
            var i0 = ptr_im.load[width=w](idx0)
            var r1 = ptr_re.load[width=w](idx1)
            var i1 = ptr_im.load[width=w](idx1)
            var r2 = ptr_re.load[width=w](idx2)
            var i2 = ptr_im.load[width=w](idx2)
            var r3 = ptr_re.load[width=w](idx3)
            var i3 = ptr_im.load[width=w](idx3)

            # Hadamard sum/diff lattice (Sparse)
            var s01_re = r0 + r1
            var s01_im = i0 + i1
            var d01_re = r0 - r1
            var d01_im = i0 - i1
            var s23_re = r2 + r3
            var s23_im = i2 + i3
            var d23_re = r2 - r3
            var d23_im = i2 - i3

            ptr_re.store(idx0, (s01_re + s23_re) * scale)
            ptr_im.store(idx0, (s01_im + s23_im) * scale)
            ptr_re.store(idx1, (d01_re + d23_re) * scale)
            ptr_im.store(idx1, (d01_im + d23_im) * scale)
            ptr_re.store(idx2, (s01_re - s23_re) * scale)
            ptr_im.store(idx2, (s01_im - s23_im) * scale)
            ptr_re.store(idx3, (d01_re - d23_re) * scale)
            ptr_im.store(idx3, (d01_im - d23_im) * scale)

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

            var r0 = ptr_re[idx0]
            var i0 = ptr_im[idx0]
            var r1 = ptr_re[idx1]
            var i1 = ptr_im[idx1]
            var r2 = ptr_re[idx2]
            var i2 = ptr_im[idx2]
            var r3 = ptr_re[idx3]
            var i3 = ptr_im[idx3]

            var s01_re = r0 + r1
            var s01_im = i0 + i1
            var d01_re = r0 - r1
            var d01_im = i0 - i1
            var s23_re = r2 + r3
            var s23_im = i2 + i3
            var d23_re = r2 - r3
            var d23_im = i2 - i3

            ptr_re[idx0] = (s01_re + s23_re) * scale
            ptr_im[idx0] = (s01_im + s23_im) * scale
            ptr_re[idx1] = (d01_re + d23_re) * scale
            ptr_im[idx1] = (d01_im + d23_im) * scale
            ptr_re[idx2] = (s01_re - s23_re) * scale
            ptr_im[idx2] = (s01_im - s23_im) * scale
            ptr_re[idx3] = (d01_re - d23_re) * scale
            ptr_im[idx3] = (d01_im - d23_im) * scale


@always_inline
fn transform_row_fused_hp_simd[
    simd_width: Int
](
    mut state: QuantumState,
    row: Int,
    row_size: Int,
    th: Int,
    tp: Int,
    theta: Float64,
):
    """Specialized sparse kernel for H + P fusion."""
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var row_offset = row * row_size

    var stride_h = 1 << th
    var stride_p = 1 << tp

    var low_pos = th if th < tp else tp
    var high_pos = tp if th < tp else th

    var num_base_pairs = row_size >> 2
    alias sq_half = 0.7071067811865476
    var cos_t = cos(theta)
    var sin_t = sin(theta)

    if low_pos >= simd_width:

        @parameter
        fn inner_simd[w: Int](k: Int):
            var i = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(i, high_pos)
            var idx0 = row_offset + idx0_local
            var idx_h = idx0 + stride_h
            var idx_p = idx0 + stride_p
            var idx_hp = idx0 + stride_h + stride_p

            var r0 = ptr_re.load[width=w](idx0)
            var i0 = ptr_im.load[width=w](idx0)
            var rh = ptr_re.load[width=w](idx_h)
            var ih = ptr_im.load[width=w](idx_h)
            var rp = ptr_re.load[width=w](idx_p)
            var ip = ptr_im.load[width=w](idx_p)
            var rhp = ptr_re.load[width=w](idx_hp)
            var ihp = ptr_im.load[width=w](idx_hp)

            # H application logic (Sparse)
            var n_r0 = (r0 + rh) * sq_half
            var n_i0 = (i0 + ih) * sq_half
            var n_rh = (r0 - rh) * sq_half
            var n_ih = (i0 - ih) * sq_half

            var n_rp = (rp + rhp) * sq_half
            var n_ip = (ip + ihp) * sq_half
            var n_rhp = (rp - rhp) * sq_half
            var n_ihp = (ip - ihp) * sq_half

            # P application logic (on p bits)
            # n_rp and n_rhp are the ones where target_p = 1
            ptr_re.store(idx0, n_r0)
            ptr_im.store(idx0, n_i0)
            ptr_re.store(idx_h, n_rh)
            ptr_im.store(idx_h, n_ih)

            ptr_re.store(idx_p, n_rp * cos_t - n_ip * sin_t)
            ptr_im.store(idx_p, n_rp * sin_t + n_ip * cos_t)
            ptr_re.store(idx_hp, n_rhp * cos_t - n_ihp * sin_t)
            ptr_im.store(idx_hp, n_rhp * sin_t + n_ihp * cos_t)

        vectorize[inner_simd, simd_width](num_base_pairs)
    else:
        # Scalar fallback
        for k in range(num_base_pairs):
            var i = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(i, high_pos)
            var idx0 = row_offset + idx0_local
            var idx_h = idx0 + stride_h
            var idx_p = idx0 + stride_p
            var idx_hp = idx0 + stride_h + stride_p

            var r0 = ptr_re[idx0]
            var i0 = ptr_im[idx0]
            var rh = ptr_re[idx_h]
            var ih = ptr_im[idx_h]
            var rp = ptr_re[idx_p]
            var ip = ptr_im[idx_p]
            var rhp = ptr_re[idx_hp]
            var ihp = ptr_im[idx_hp]

            var n_r0 = (r0 + rh) * sq_half
            var n_i0 = (i0 + ih) * sq_half
            var n_rh = (r0 - rh) * sq_half
            var n_ih = (i0 - ih) * sq_half
            var n_rp = (rp + rhp) * sq_half
            var n_ip = (ip + ihp) * sq_half
            var n_rhp = (rp - rhp) * sq_half
            var n_ihp = (ip - ihp) * sq_half

            ptr_re[idx0] = n_r0
            ptr_im[idx0] = n_i0
            ptr_re[idx_h] = n_rh
            ptr_im[idx_h] = n_ih
            ptr_re[idx_p] = n_rp * cos_t - n_ip * sin_t
            ptr_im[idx_p] = n_rp * sin_t + n_ip * cos_t
            ptr_re[idx_hp] = n_rhp * cos_t - n_ihp * sin_t
            ptr_im[idx_hp] = n_rhp * sin_t + n_ihp * cos_t


@always_inline
fn transform_row_fused_st_hp_simd[
    simd_width: Int
](
    mut state: QuantumState,
    row: Int,
    row_size: Int,
    target: Int,
    theta: Float64,
):
    """Specialized sparse kernel for H + P fusion on the SAME target."""
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var row_offset = row * row_size
    var stride = 1 << target
    var num_pairs = row_size >> 1
    alias sq_half = 0.7071067811865476
    var cos_t = cos(theta)
    var sin_t = sin(theta)

    if stride >= simd_width:

        @parameter
        fn inner_simd[w: Int](k: Int):
            var idx0_local = insert_zero_bit(k, target)
            var idx0 = row_offset + idx0_local
            var idx1 = idx0 + stride

            var r0 = ptr_re.load[width=w](idx0)
            var i0 = ptr_im.load[width=w](idx0)
            var r1 = ptr_re.load[width=w](idx1)
            var i1 = ptr_im.load[width=w](idx1)

            var n_r0 = (r0 + r1) * sq_half
            var n_i0 = (i0 + i1) * sq_half
            var n_r1 = (r0 - r1) * sq_half
            var n_i1 = (i0 - i1) * sq_half

            ptr_re.store(idx0, n_r0)
            ptr_im.store(idx0, n_i0)
            ptr_re.store(idx1, n_r1 * cos_t - n_i1 * sin_t)
            ptr_im.store(idx1, n_r1 * sin_t + n_i1 * cos_t)

        vectorize[inner_simd, simd_width](num_pairs)
    else:
        for k in range(num_pairs):
            var idx0_local = insert_zero_bit(k, target)
            var idx0 = row_offset + idx0_local
            var idx1 = idx0 | stride
            var r0 = ptr_re[idx0]
            var i0 = ptr_im[idx0]
            var r1 = ptr_re[idx1]
            var i1 = ptr_im[idx1]

            var n_r0 = (r0 + r1) * sq_half
            var n_i0 = (i0 + i1) * sq_half
            var n_r1 = (r0 - r1) * sq_half
            var n_i1 = (i0 - i1) * sq_half

            ptr_re[idx0] = n_r0
            ptr_im[idx0] = n_i0
            ptr_re[idx1] = n_r1 * cos_t - n_i1 * sin_t
            ptr_im[idx1] = n_r1 * sin_t + n_i1 * cos_t


@always_inline
fn transform_row_fused_pp_simd[
    simd_width: Int
](
    mut state: QuantumState,
    row: Int,
    row_size: Int,
    t1: Int,
    t2: Int,
    theta1: Float64,
    theta2: Float64,
):
    """Specialized sparse kernel for P + P fusion."""
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var row_offset = row * row_size

    var high_pos = t1
    var low_pos = t2
    var th1 = theta1
    var th2 = theta2
    if t2 > t1:
        high_pos = t2
        low_pos = t1
        th1 = theta2
        th2 = theta1

    var stride_low = 1 << low_pos
    var stride_high = 1 << high_pos
    var num_base_pairs = row_size >> 2

    # Precompute phase components
    var c1 = cos(th1)
    var s1 = sin(th1)
    var c2 = cos(th2)
    var s2 = sin(th2)
    var c12 = cos(th1 + th2)
    var s12 = sin(th1 + th2)

    if stride_low >= simd_width:

        @parameter
        fn inner_simd[w: Int](k: Int):
            var i = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(i, high_pos)
            var idx0 = row_offset + idx0_local
            var idx1 = idx0 + stride_low
            var idx2 = idx0 + stride_high
            var idx3 = idx1 + stride_high

            # idx0: Identity

            # idx1: Shift by theta2
            var r1 = ptr_re.load[width=w](idx1)
            var i1 = ptr_im.load[width=w](idx1)
            ptr_re.store(idx1, r1 * c2 - i1 * s2)
            ptr_im.store(idx1, r1 * s2 + i1 * c2)

            # idx2: Shift by theta1
            var r2 = ptr_re.load[width=w](idx2)
            var i2 = ptr_im.load[width=w](idx2)
            ptr_re.store(idx2, r2 * c1 - i2 * s1)
            ptr_im.store(idx2, r2 * s1 + i2 * c1)

            # idx3: Shift by theta1 + theta2
            var r3 = ptr_re.load[width=w](idx3)
            var i3 = ptr_im.load[width=w](idx3)
            ptr_re.store(idx3, r3 * c12 - i3 * s12)
            ptr_im.store(idx3, r3 * s12 + i3 * c12)

        vectorize[inner_simd, simd_width](num_base_pairs)
    else:
        for k in range(num_base_pairs):
            var i = insert_zero_bit(k, low_pos)
            var idx0_local = insert_zero_bit(i, high_pos)
            var idx0 = row_offset + idx0_local
            var idx1 = idx0 | stride_low
            var idx2 = idx0 | stride_high
            var idx3 = idx1 | stride_high

            var r1 = ptr_re[idx1]
            var i1 = ptr_im[idx1]
            ptr_re[idx1] = r1 * c2 - i1 * s2
            ptr_im[idx1] = r1 * s2 + i1 * c2

            var r2 = ptr_re[idx2]
            var i2 = ptr_im[idx2]
            ptr_re[idx2] = r2 * c1 - i2 * s1
            ptr_im[idx2] = r2 * s1 + i2 * c1

            var r3 = ptr_re[idx3]
            var i3 = ptr_im[idx3]
            ptr_re[idx3] = r3 * c12 - i3 * s12
            ptr_im[idx3] = r3 * s12 + i3 * c12


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
    theta1: Float64,
    theta2: Float64,
):
    """Specialized sparse kernel for CP + CP fusion sharing a control bit."""
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var row_offset = row * row_size

    # Prepare indices for nested insert_zero_bit (must be in increasing order)
    var b1 = control
    var b2 = t1
    var b3 = t2
    # Bubble sort
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

    var cos1 = cos(theta1)
    var sin1 = sin(theta1)
    var cos2 = cos(theta2)
    var sin2 = sin(theta2)
    var cos12 = cos(theta1 + theta2)
    var sin12 = sin(theta1 + theta2)

    var num_base_groups = row_size >> 3

    if b1 >= simd_width:

        @parameter
        fn inner_simd[w: Int](k: Int):
            var idx0_local = insert_zero_bit(
                insert_zero_bit(insert_zero_bit(k, b1), b2), b3
            )
            var idx0 = row_offset + idx0_local

            # States where control bit is 1
            var c_base = idx0 + s_control

            # Case 1: Control=1, T1=1, T2=0
            var i1 = c_base + s1
            var r1 = ptr_re.load[width=w](i1)
            var im1 = ptr_im.load[width=w](i1)
            ptr_re.store(i1, r1 * cos1 - im1 * sin1)
            ptr_im.store(i1, r1 * sin1 + im1 * cos1)

            # Case 2: Control=1, T1=0, T2=1
            var i2 = c_base + s2
            var r2 = ptr_re.load[width=w](i2)
            var im2 = ptr_im.load[width=w](i2)
            ptr_re.store(i2, r2 * cos2 - im2 * sin2)
            ptr_im.store(i2, r2 * sin2 + im2 * cos2)

            # Case 3: Control=1, T1=1, T2=1
            var i12 = c_base + s1 + s2
            var r12 = ptr_re.load[width=w](i12)
            var im12 = ptr_im.load[width=w](i12)
            ptr_re.store(i12, r12 * cos12 - im12 * sin12)
            ptr_im.store(i12, r12 * sin12 + im12 * cos12)

        vectorize[inner_simd, simd_width](num_base_groups)
    else:
        for k in range(num_base_groups):
            var idx0 = row_offset + insert_zero_bit(
                insert_zero_bit(insert_zero_bit(k, b1), b2), b3
            )
            var c_base = idx0 | s_control

            var inst_i1 = c_base | s1
            var r1 = ptr_re[inst_i1]
            var im1 = ptr_im[inst_i1]
            ptr_re[inst_i1] = r1 * cos1 - im1 * sin1
            ptr_im[inst_i1] = r1 * sin1 + im1 * cos1

            var inst_i2 = c_base | s2
            var r2 = ptr_re[inst_i2]
            var im2 = ptr_im[inst_i2]
            ptr_re[inst_i2] = r2 * cos2 - im2 * sin2
            ptr_im[inst_i2] = r2 * sin2 + im2 * cos2

            var inst_i12 = c_base | s1 | s2
            var r12 = ptr_re[inst_i12]
            var im12 = ptr_im[inst_i12]
            ptr_re[inst_i12] = r12 * cos12 - im12 * sin12
            ptr_im[inst_i12] = r12 * sin12 + im12 * cos12
