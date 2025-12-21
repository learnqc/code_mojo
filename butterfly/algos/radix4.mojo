from butterfly.core.state import QuantumState
from butterfly.core.types import Type, FloatType, float_bytes
from algorithm import parallelize
from sys.info import simd_width_of
from memory import UnsafePointer


fn radix4_dif_simd(
    mut state: QuantumState,
    factors_re: List[FloatType],
    factors_im: List[FloatType],
    stride: Int,
):
    """
    Performs one Radix-4 DIF stage.
    Equivalent to 2 Radix-2 stages.
    """
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()
    var factors_len = len(factors_re)
    var n = state.size()

    radix4_kernel_impl(
        ptr_re, ptr_im, ptr_fac_re, ptr_fac_im, n, stride, factors_len
    )


fn radix4_kernel_impl(
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    ptr_fac_re: UnsafePointer[FloatType],
    ptr_fac_im: UnsafePointer[FloatType],
    n: Int,
    stride: Int,
    factors_len: Int,
):
    var p_re = UnsafePointer[FloatType](ptr_re.address)
    var p_im = UnsafePointer[FloatType](ptr_im.address)
    var p_fac_re = UnsafePointer[FloatType](ptr_fac_re.address)
    var p_fac_im = UnsafePointer[FloatType](ptr_fac_im.address)

    var global_factor_stride = factors_len // (stride * 2)

    alias target_block_bytes = 16384  # 16KB
    var items_per_block = target_block_bytes // float_bytes
    alias simd_width = simd_width_of[Type]()
    items_per_block = (items_per_block // simd_width) * simd_width

    var block_width = 4 * stride
    var num_blocks = n // block_width
    var total_work = num_blocks * stride

    @parameter
    fn work_kernel(idx: Int):
        var start = idx * items_per_block
        var end = start + items_per_block
        if end > total_work:
            end = total_work

        var i = start
        while i < end:
            var group = i // stride
            var k = i % stride

            var remaining_in_group = stride - k
            var remaining_in_task = end - i
            var step_len = min(remaining_in_group, remaining_in_task)

            var group_offset = group * block_width

            var p0 = group_offset + k
            var p1 = p0 + stride
            var p2 = p1 + stride
            var p3 = p2 + stride

            for j in range(0, step_len, simd_width):
                var vl = simd_width
                if j + vl > step_len:
                    vl = step_len - j

                if vl < simd_width:
                    # Scalar Fallback for tail
                    for s in range(vl):
                        var idx0 = p0 + s
                        var idx1 = p1 + s
                        var idx2 = p2 + s
                        var idx3 = p3 + s
                        var curr_k = k + s

                        # Load Scalar
                        var a_re = p_re[idx0]
                        var a_im = p_im[idx0]
                        var b_re = p_re[idx1]
                        var b_im = p_im[idx1]
                        var c_re = p_re[idx2]
                        var c_im = p_im[idx2]
                        var d_re = p_re[idx3]
                        var d_im = p_im[idx3]

                        # Butterfly (Scalar)
                        var t1_re = a_re + c_re
                        var t1_im = a_im + c_im
                        var t2_re = a_re - c_re
                        var t2_im = a_im - c_im
                        var t3_re = b_re + d_re
                        var t3_im = b_im + d_im
                        var t4_re = b_re - d_re
                        var t4_im = b_im - d_im

                        var z0_re = t1_re + t3_re
                        var z0_im = t1_im + t3_im
                        var z2_re = t1_re - t3_re
                        var z2_im = t1_im - t3_im
                        var z1_re = t2_re + t4_im
                        var z1_im = t2_im - t4_re
                        var z3_re = t2_re - t4_im
                        var z3_im = t2_im + t4_re

                        # Twiddles (Scalar)
                        var k_adj = curr_k * global_factor_stride

                        # w1
                        var idx_w1 = k_adj
                        # wrapping
                        var w1_re: FloatType
                        var w1_im: FloatType
                        if idx_w1 >= factors_len:
                            var w_re_raw = p_fac_re[idx_w1 - factors_len]
                            var w_im_raw = p_fac_im[idx_w1 - factors_len]
                            w1_re = -w_re_raw
                            w1_im = -w_im_raw
                        else:
                            w1_re = p_fac_re[idx_w1]
                            w1_im = p_fac_im[idx_w1]

                        # w2
                        var idx_w2 = k_adj * 2
                        var w2_re: FloatType
                        var w2_im: FloatType
                        if idx_w2 >= factors_len:
                            var w_re_raw = p_fac_re[idx_w2 - factors_len]
                            var w_im_raw = p_fac_im[idx_w2 - factors_len]
                            w2_re = -w_re_raw
                            w2_im = -w_im_raw
                        else:
                            w2_re = p_fac_re[idx_w2]
                            w2_im = p_fac_im[idx_w2]

                        # w3
                        var idx_w3 = k_adj * 3
                        # Might wrap once as discussed (3N/4 max)
                        var w3_re: FloatType
                        var w3_im: FloatType
                        if idx_w3 >= factors_len:
                            var w_re_raw = p_fac_re[idx_w3 - factors_len]
                            var w_im_raw = p_fac_im[idx_w3 - factors_len]
                            w3_re = -w_re_raw
                            w3_im = -w_im_raw
                        else:
                            w3_re = p_fac_re[idx_w3]
                            w3_im = p_fac_im[idx_w3]

                        # Apply
                        var out1_re = z2_re * w2_re - z2_im * w2_im
                        var out1_im = z2_re * w2_im + z2_im * w2_re

                        var out2_re = z1_re * w1_re - z1_im * w1_im
                        var out2_im = z1_re * w1_im + z1_im * w1_re

                        var out3_re = z3_re * w3_re - z3_im * w3_im
                        var out3_im = z3_re * w3_im + z3_im * w3_re

                        # Store
                        # Store
                        p_re[idx0] = z0_re
                        p_im[idx0] = z0_im
                        p_re[idx1] = out1_re
                        p_im[idx1] = out1_im
                        p_re[idx2] = out2_re
                        p_im[idx2] = out2_im
                        p_re[idx3] = out3_re
                        p_im[idx3] = out3_im
                else:
                    # Full SIMD
                    var idx0 = p0 + j
                    var idx1 = p1 + j
                    var idx2 = p2 + j
                    var idx3 = p3 + j
                    var curr_k = k + j

                    # --- Compute Step Inline ---
                    # Load Inputs (a, b, c, d)
                    var a_re = p_re.load[width=simd_width](idx0)
                    var a_im = p_im.load[width=simd_width](idx0)
                    var b_re = p_re.load[width=simd_width](idx1)
                    var b_im = p_im.load[width=simd_width](idx1)
                    var c_re = p_re.load[width=simd_width](idx2)
                    var c_im = p_im.load[width=simd_width](idx2)
                    var d_re = p_re.load[width=simd_width](idx3)
                    var d_im = p_im.load[width=simd_width](idx3)

                    # Butterfly
                    var t1_re = a_re + c_re
                    var t1_im = a_im + c_im
                    var t2_re = a_re - c_re
                    var t2_im = a_im - c_im
                    var t3_re = b_re + d_re
                    var t3_im = b_im + d_im
                    var t4_re = b_re - d_re
                    var t4_im = b_im - d_im

                    var z0_re = t1_re + t3_re
                    var z0_im = t1_im + t3_im
                    var z2_re = t1_re - t3_re
                    var z2_im = t1_im - t3_im
                    var z1_re = t2_re + t4_im
                    var z1_im = t2_im - t4_re
                    var z3_re = t2_re - t4_im
                    var z3_im = t2_im + t4_re

                    # Twiddles
                    var k_vec = SIMD[DType.int64, simd_width]()
                    for i in range(simd_width):
                        k_vec[i] = Int64(curr_k + i)

                    var k_adj = k_vec * global_factor_stride

                    # Inline gather_twiddle for w1, w2, w3
                    # Comparisons changed to < to avoid >= constraint

                    # --- w1 ---
                    var idx_w1 = k_adj
                    var limit_simd = SIMD[DType.int64, simd_width](factors_len)

                    # is_in_range means idx < limit. (i.e. NOT neg)
                    var is_in_range_1 = idx_w1.lt(limit_simd)

                    # if in_range: idx. else: idx - limit
                    var read_idx_1 = safe_select[simd_width, DType.int64](
                        is_in_range_1, idx_w1, idx_w1 - limit_simd
                    )

                    var w_re_raw_1 = p_fac_re.gather(read_idx_1)
                    var w_im_raw_1 = p_fac_im.gather(read_idx_1)

                    # if in_range: w. else: -w
                    var w1_re = safe_select[simd_width, Type](
                        is_in_range_1, w_re_raw_1, -w_re_raw_1
                    )
                    var w1_im = safe_select[simd_width, Type](
                        is_in_range_1, w_im_raw_1, -w_im_raw_1
                    )

                    # --- w2 ---
                    var idx_w2 = k_adj * 2
                    var is_in_range_2 = idx_w2.lt(limit_simd)
                    var read_idx_2 = safe_select[simd_width, DType.int64](
                        is_in_range_2, idx_w2, idx_w2 - limit_simd
                    )

                    var w_re_raw_2 = p_fac_re.gather(read_idx_2)
                    var w_im_raw_2 = p_fac_im.gather(read_idx_2)

                    var w2_re = safe_select[simd_width, Type](
                        is_in_range_2, w_re_raw_2, -w_re_raw_2
                    )
                    var w2_im = safe_select[simd_width, Type](
                        is_in_range_2, w_im_raw_2, -w_im_raw_2
                    )

                    # --- w3 ---
                    var idx_w3 = k_adj * 3
                    # Wrapping might happen recursively?
                    # Max 3*k_adj. If N=large, 3*k could exceed 2*factors_len?
                    # factors_len = N/2.
                    # k < N/4. 3k < 3N/4.
                    # 3N/4 = 1.5 * N/2.
                    # So it exceeds factors_len ONCE. (1.5 < 2).
                    # So single subtraction is enough.
                    var is_in_range_3 = idx_w3.lt(limit_simd)
                    var read_idx_3 = safe_select[simd_width, DType.int64](
                        is_in_range_3, idx_w3, idx_w3 - limit_simd
                    )

                    var w_re_raw_3 = p_fac_re.gather(read_idx_3)
                    var w_im_raw_3 = p_fac_im.gather(read_idx_3)

                    var w3_re = safe_select[simd_width, Type](
                        is_in_range_3, w_re_raw_3, -w_re_raw_3
                    )
                    var w3_im = safe_select[simd_width, Type](
                        is_in_range_3, w_im_raw_3, -w_im_raw_3
                    )

                    # Apply
                    var out1_re = z2_re * w2_re - z2_im * w2_im
                    var out1_im = z2_re * w2_im + z2_im * w2_re

                    var out2_re = z1_re * w1_re - z1_im * w1_im
                    var out2_im = z1_re * w1_im + z1_im * w1_re

                    var out3_re = z3_re * w3_re - z3_im * w3_im
                    var out3_im = z3_re * w3_im + z3_im * w3_re

                    # Store SIMD
                    p_re.store(idx0, z0_re)
                    p_im.store(idx0, z0_im)
                    p_re.store(idx1, out1_re)
                    p_im.store(idx1, out1_im)
                    p_re.store(idx2, out2_re)
                    p_im.store(idx2, out2_im)
                    p_re.store(idx3, out3_re)
                    p_im.store(idx3, out3_im)

            i += step_len

    parallelize[work_kernel](
        (total_work + items_per_block - 1) // items_per_block
    )


@always_inline
fn safe_select[
    simd_w: Int, T: DType
](
    cond: SIMD[DType.bool, simd_w], t: SIMD[T, simd_w], f: SIMD[T, simd_w]
) -> SIMD[T, simd_w]:
    @parameter
    if simd_w == 1:
        # Scalar
        return t if cond[0] else f
    else:
        return cond.select(t, f)
