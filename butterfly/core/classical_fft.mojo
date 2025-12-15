from butterfly.core.state import *
from butterfly.algos.vec_swaps import swap_state_to_distance_8_simd
from butterfly.algos.tail_stages import (
    fused_stride2_stride1_swapped,
    stride4_swapped_simd,
)
from memory import UnsafePointer
from algorithm import parallelize
from buffer import NDBuffer
from utils.fast_div import FastDiv


fn generate_factors(
    n: Int, inverse: Bool = False
) -> Tuple[List[FloatType], List[FloatType]]:
    var angle = 2.0 * pi / FloatType(n)
    if not inverse:
        angle = -angle

    # Twiddle factor for this stage
    var wm_re = cos(angle)
    var wm_im = sin(angle)

    var w_re = FloatType(1.0)
    var w_im = FloatType(0.0)

    alias sqh = sqrt(0.5).cast[Type]()

    factors_re = List[FloatType](length=n // 2, fill=0.0)
    factors_im = List[FloatType](length=n // 2, fill=0.0)

    for j in range(n // 8):
        factors_re[j] = w_re  # cos(j * angle)
        factors_im[j] = w_im  # sin(j * angle)

        sum = (w_re + w_im) * sqh
        diff = (w_re - w_im) * sqh

        factors_re[j + n // 8] = sum  # cos((j + n // 8) * angle)
        factors_im[j + n // 8] = -diff  # sin((j + n // 8) * angle)

        factors_re[j + n // 4] = w_im  # cos((j + n // 4) * angle)
        factors_im[j + n // 4] = -w_re  # sin((j + n // 4) * angle)

        factors_re[j + 3 * (n // 8)] = -diff  # cos((j + 3 * (n // 8)) * angle)
        factors_im[j + 3 * (n // 8)] = -sum  # sin((j + 3 * (n // 8)) * angle)

        var w_im_new = w_re * wm_im + w_im * wm_re
        var w_re_new = w_re * wm_re - w_im * wm_im

        w_re = w_re_new
        w_im = w_im_new

    return factors_re^, factors_im^


fn fft_dit(mut state: QuantumState, inverse: Bool = False):
    """Apply FFT to the quantum state using Cooley-Tukey algorithm.

    This is an in-place FFT that operates on the state's complex amplitudes.
    Uses the radix-2 decimation-in-time algorithm.

    Args:
        state: QuantumState to transform (modified in-place)
        inverse: If True, performs inverse FFT
    """
    var n = state.size()

    # Bit-reversal permutation
    bit_reverse_state(state)

    ref factors_re, factors_im = generate_factors(n)

    # Cooley-Tukey FFT
    var m = 2
    while m <= n:
        f = n // m
        # Process all groups at this stage
        for k in range(0, n, m):
            # Butterfly operations within group
            for j in range(m // 2):
                w_re = factors_re[f * j]
                w_im = factors_im[f * j]
                var t_re = (
                    w_re * state.re[k + j + m // 2]
                    - w_im * state.im[k + j + m // 2]
                )
                var t_im = (
                    w_re * state.im[k + j + m // 2]
                    + w_im * state.re[k + j + m // 2]
                )

                var u_re = state.re[k + j]
                var u_im = state.im[k + j]

                state.re[k + j] = u_re + t_re
                state.im[k + j] = u_im + t_im
                state.re[k + j + m // 2] = u_re - t_re
                state.im[k + j + m // 2] = u_im - t_im

        m *= 2

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale


fn fft_dit_simd(mut state: QuantumState, inverse: Bool = False):
    """Vectorized DIT FFT."""
    var n = state.size()
    bit_reverse_state(state)
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = UnsafePointer(state.re.unsafe_ptr())
    var ptr_im = UnsafePointer(state.im.unsafe_ptr())

    var m = 2
    while m <= n:
        var f = n // m
        var m_half = m // 2

        @parameter
        fn process_simd[width: Int](j: Int):
            # For large strides we can't vectorize factor loading easily unless stride=1
            # But here 'j' is the inner loop index.
            # We want to process 'width' items at j, j+1, ... j+width-1
            # Indices:
            # k is fixed for this block.
            # u_idx = k + j
            # t_idx = k + j + m_half

            # Factors:
            # fac_idx = f * j -> strided!
            # If f == 1 (last stage), stride is 1 (contiguous)
            # If f > 1, we must gather or compute.

            # Since f is power of 2, we can just compute indices?
            # Or use gather.

            # Wait, `process_simd` is called by `vectorize`.
            # `vectorize` passes `j` as scalar index.
            # We assume we are inside the `k` loop.

            # For simple vectorization, let's just use gather/scatter or scalar loads for factors if stride > 1
            pass

        # Strategy:
        # If m_half >= simd_width, we can vectorize the j loop.
        # factor index is f*j. If f==1, it's contiguous. If f>1, it's strided.

        for k in range(0, n, m):
            if m_half >= simd_width:  # Vectorize

                @parameter
                fn inner_simd[width: Int](j: Int):
                    var offset = k + j
                    var offset_t = offset + m_half

                    var u_re = ptr_re.load[width=width](offset)
                    var u_im = ptr_im.load[width=width](offset)
                    var t_re_in = ptr_re.load[width=width](offset_t)
                    var t_im_in = ptr_im.load[width=width](offset_t)

                    # Compute factors
                    var w_re: SIMD[Type, width]
                    var w_im: SIMD[Type, width]

                    if f == 1:
                        w_re = factors_re.unsafe_ptr().load[width=width](f * j)
                        w_im = factors_im.unsafe_ptr().load[width=width](f * j)
                    else:
                        # Manual gather for strided access
                        w_re = SIMD[Type, width]()
                        w_im = SIMD[Type, width]()
                        for s in range(width):
                            w_re[s] = factors_re.unsafe_ptr().load(f * (j + s))
                            w_im[s] = factors_im.unsafe_ptr().load(f * (j + s))

                    # Complex mul: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
                    # t = w * t_in
                    var t_re = w_re * t_re_in - w_im * t_im_in
                    var t_im = w_re * t_im_in + w_im * t_re_in

                    ptr_re.store[width=width](offset, u_re + t_re)
                    ptr_im.store[width=width](offset, u_im + t_im)
                    ptr_re.store[width=width](offset_t, u_re - t_re)
                    ptr_im.store[width=width](offset_t, u_im - t_im)

                vectorize[inner_simd, simd_width](m_half)
            else:
                # Scalar fallback for small blocks
                for j in range(m_half):
                    var w_re = factors_re[f * j]
                    var w_im = factors_im[f * j]

                    var u_re = state.re[k + j]
                    var u_im = state.im[k + j]
                    var t_re_in = state.re[k + j + m // 2]
                    var t_im_in = state.im[k + j + m // 2]

                    var t_re = w_re * t_re_in - w_im * t_im_in
                    var t_im = w_re * t_im_in + w_im * t_re_in

                    state.re[k + j] = u_re + t_re
                    state.im[k + j] = u_im + t_im
                    state.re[k + j + m // 2] = u_re - t_re
                    state.im[k + j + m // 2] = u_im - t_im

        m *= 2

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    # Vectorize normalization
    var ptr_re_norm = UnsafePointer(state.re.unsafe_ptr())
    var ptr_im_norm = UnsafePointer(state.im.unsafe_ptr())

    @parameter
    fn norm_simd[width: Int](i: Int):
        ptr_re_norm.store[width=width](
            i, ptr_re_norm.load[width=width](i) * scale
        )
        ptr_im_norm.store[width=width](
            i, ptr_im_norm.load[width=width](i) * scale
        )

    vectorize[norm_simd, simd_width](n)


fn fft_dif(mut state: QuantumState, inverse: Bool = False):
    """NumPy-style FFT using decimation-in-frequency algorithm."""
    var n = state.size()
    var log_n = Int(log2(Float64(n)))

    # Factors generation (reused or recomputed)
    # We need to inline this properly or share it.
    # For now, we reuse the logic but we need access to factors.
    # The original fft_dif computed factors inside? No, it used generate_factors equivalent logic implicitly?
    # Actually the previous file had a manual copy of factor generation inside generate_factors AND inside fft_dif factors loop.
    # Let's just use generate_factors for cleaner code, assuming cost is negligible or we hoist it.

    ref factors_re, factors_im = generate_factors(n)

    var block_size = n

    for _ in range(log_n):
        var half_block = block_size // 2
        var _ = n // block_size  # Stride for factors?
        # In DIF, we need w^(k * N/block_size)?
        # Original code used: factors_re[n // block_size * j]
        # So stride is n // block_size.

        var stride = n // block_size

        for block_start in range(0, n, block_size):
            for j in range(half_block):
                var top_idx = block_start + j
                var bot_idx = top_idx + half_block

                var top_re = state.re[top_idx]
                var top_im = state.im[top_idx]
                var bot_re = state.re[bot_idx]
                var bot_im = state.im[bot_idx]

                var sum_re = top_re + bot_re
                var sum_im = top_im + bot_im
                var diff_re = top_re - bot_re
                var diff_im = top_im - bot_im

                var w_re = factors_re[stride * j]
                var w_im = factors_im[stride * j]

                var t_re = w_re * diff_re - w_im * diff_im
                var t_im = w_re * diff_im + w_im * diff_re

                # In DIF, we multiply AFTER butterfly on the bottom part?
                # Standard DIF:
                # a = u + v
                # b = (u - v) * w
                # Yes, matching logic above.

                state.re[top_idx] = sum_re
                state.im[top_idx] = sum_im
                state.re[bot_idx] = t_re
                state.im[bot_idx] = t_im

        block_size = half_block

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale


fn fft_dif_simd(mut state: QuantumState, inverse: Bool = False):
    """Vectorized DIF FFT."""
    var n = state.size()
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = UnsafePointer(state.re.unsafe_ptr())
    var ptr_im = UnsafePointer(state.im.unsafe_ptr())

    var block_size = n
    for _ in range(log_n):
        var half_block = block_size // 2
        var stride = n // block_size

        for block_start in range(0, n, block_size):
            if half_block >= simd_width:

                @parameter
                fn inner_simd[width: Int](j: Int):
                    var top_idx = block_start + j
                    var bot_idx = top_idx + half_block

                    var top_re = ptr_re.load[width=width](top_idx)
                    var top_im = ptr_im.load[width=width](top_idx)
                    var bot_re = ptr_re.load[width=width](bot_idx)
                    var bot_im = ptr_im.load[width=width](bot_idx)

                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im

                    # Factors
                    var w_re: SIMD[Type, width]
                    var w_im: SIMD[Type, width]

                    if stride == 1:
                        w_re = factors_re.unsafe_ptr().load[width=width](j)
                        w_im = factors_im.unsafe_ptr().load[width=width](j)
                    else:
                        # Manual gather for strided access
                        w_re = SIMD[Type, width]()
                        w_im = SIMD[Type, width]()
                        for k in range(width):
                            w_re[k] = factors_re.unsafe_ptr().load(
                                stride * (j + k)
                            )
                            w_im[k] = factors_im.unsafe_ptr().load(
                                stride * (j + k)
                            )

                    # t = diff * w
                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    ptr_re.store[width=width](top_idx, sum_re)
                    ptr_im.store[width=width](top_idx, sum_im)
                    ptr_re.store[width=width](bot_idx, t_re)
                    ptr_im.store[width=width](bot_idx, t_im)

                vectorize[inner_simd, simd_width](half_block)
            else:
                # Scalar fallback
                for j in range(half_block):
                    var w_re = factors_re[stride * j]
                    var w_im = factors_im[stride * j]

                    var top_idx = block_start + j
                    var bot_idx = top_idx + half_block

                    var top_re = state.re[top_idx]
                    var top_im = state.im[top_idx]
                    var bot_re = state.re[bot_idx]
                    var bot_im = state.im[bot_idx]

                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im

                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    state.re[top_idx] = sum_re
                    state.im[top_idx] = sum_im
                    state.re[bot_idx] = t_re
                    state.im[bot_idx] = t_im

        block_size = half_block

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    var ptr_re_norm = UnsafePointer(state.re.unsafe_ptr())
    var ptr_im_norm = UnsafePointer(state.im.unsafe_ptr())

    @parameter
    fn norm_simd[width: Int](i: Int):
        ptr_re_norm.store[width=width](
            i, ptr_re_norm.load[width=width](i) * scale
        )
        ptr_im_norm.store[width=width](
            i, ptr_im_norm.load[width=width](i) * scale
        )

    vectorize[norm_simd, simd_width](n)


fn fft_dit_parallel(mut state: QuantumState, inverse: Bool = False):
    """Parallelized DIT FFT."""
    var n = state.size()
    bit_reverse_state(state)
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var m = 2
    while m <= n:
        var f = n // m
        var m_half = m // 2

        @parameter
        fn stage_worker(idx: Int):
            # idx goes from 0 to n/2 - 1
            var group = idx // m_half
            var j = idx % m_half
            var k = group * m

            var w_re = ptr_fac_re[f * j]
            var w_im = ptr_fac_im[f * j]

            var u_re = ptr_re[k + j]
            var u_im = ptr_im[k + j]
            var t_re_in = ptr_re[k + j + m_half]
            var t_im_in = ptr_im[k + j + m_half]

            var t_re = w_re * t_re_in - w_im * t_im_in
            var t_im = w_re * t_im_in + w_im * t_re_in

            ptr_re[k + j] = u_re + t_re
            ptr_im[k + j] = u_im + t_im
            ptr_re[k + j + m_half] = u_re - t_re
            ptr_im[k + j + m_half] = u_im - t_im

        parallelize[stage_worker](n // 2, n // 2)
        m *= 2

    var scale = FloatType(1.0) / sqrt(FloatType(n))

    @parameter
    fn norm_worker(i: Int):
        ptr_re[i] *= scale
        ptr_im[i] *= scale

    parallelize[norm_worker](n)


fn fft_dif_parallel(mut state: QuantumState, inverse: Bool = False):
    """Parallelized DIF FFT."""
    var n = state.size()
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var stride = n // 2
    for _ in range(log_n):

        @parameter
        fn stage_worker(idx: Int):
            # idx: 0 to n/2 - 1
            var block_idx = idx // stride
            var j = idx % stride
            var block_start = block_idx * 2 * stride

            var top_idx = block_start + j
            var bot_idx = top_idx + stride

            var top_re = ptr_re[top_idx]
            var top_im = ptr_im[top_idx]
            var bot_re = ptr_re[bot_idx]
            var bot_im = ptr_im[bot_idx]

            var sum_re = top_re + bot_re
            var sum_im = top_im + bot_im
            var diff_re = top_re - bot_re
            var diff_im = top_im - bot_im

            var w_re = ptr_fac_re[n // 2 // stride * j]
            var w_im = ptr_fac_im[n // 2 // stride * j]

            var t_re = diff_re * w_re - diff_im * w_im
            var t_im = diff_re * w_im + diff_im * w_re

            ptr_re[top_idx] = sum_re
            ptr_im[top_idx] = sum_im
            ptr_re[bot_idx] = t_re
            ptr_im[bot_idx] = t_im

        parallelize[stage_worker](n // 2, 8)
        stride = stride // 2

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    # State was replaced in bit_reverse_state, so we need fresh pointers
    var final_ptr_re = state.re.unsafe_ptr()
    var final_ptr_im = state.im.unsafe_ptr()

    @parameter
    fn norm_worker(i: Int):
        final_ptr_re[i] *= scale
        final_ptr_im[i] *= scale

    parallelize[norm_worker](n)


fn fft_dif_parallel_simd(mut state: QuantumState, inverse: Bool = False):
    """Parallelized + Vectorized DIF FFT.

    Uses parallel execution for outer chunks and SIMD gather/scatter for inner butterfly operations.
    Best for large n (>= 17).
    """
    var n = state.size()
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var stride = n // 2
    for _ in range(log_n):
        # We process 'n // 2' butterflies in total.
        # We can chunk this work.
        # Each chunk will process `simd_width` butterflies at once for vectorization.

        var fd = FastDiv[DType.uint32](stride)
        alias TargetType = Scalar[FastDiv[DType.uint32].uint_type]

        if (stride // 2) >= simd_width:

            @parameter
            fn stage_worker_simd(vec_idx: Int):
                var idx_base = vec_idx * simd_width

                # Construct vector of indices: [idx, idx+1, ..., idx+width-1]
                var idxs = SIMD[DType.int64, simd_width]()
                for i in range(simd_width):
                    idxs[i] = Int64(idx_base + i)

                # Map to butterfly indices
                # block_idx = idx // stride
                # j = idx % stride

                var group_ids = idxs // stride
                var js = idxs % stride

                var block_starts = group_ids * (2 * stride)

                var top_idxs = block_starts + js
                var bot_idxs = top_idxs + stride

                # Gather data
                var top_re = ptr_re.gather(top_idxs)
                var top_im = ptr_im.gather(top_idxs)
                var bot_re = ptr_re.gather(bot_idxs)
                var bot_im = ptr_im.gather(bot_idxs)

                var sum_re = top_re + bot_re
                var sum_im = top_im + bot_im
                var diff_re = top_re - bot_re
                var diff_im = top_im - bot_im

                # Factors gather
                # factor_idx = (n // 2 // stride) * j
                var w_idxs = js * (n // 2 // stride)
                var w_re = ptr_fac_re.gather(w_idxs)
                var w_im = ptr_fac_im.gather(w_idxs)

                # Butterfly
                var t_re = diff_re * w_re - diff_im * w_im
                var t_im = diff_re * w_im + diff_im * w_re

                # Scatter results
                ptr_re.scatter(top_idxs, sum_re)
                ptr_im.scatter(top_idxs, sum_im)
                ptr_re.scatter(bot_idxs, t_re)
                ptr_im.scatter(bot_idxs, t_im)

            parallelize[stage_worker_simd]((n // 2) // simd_width, 8)

        else:
            # Fallback for very small strides (unlikely if n is large, but possible at last stage)
            for idx in range(n // 2):
                # var block_idx = idx // stride
                # var j = idx % stride
                block_idx, j = fd.__divmod__(idx)

                var block_start = block_idx * 2 * stride

                var top_idx = block_start + j
                var bot_idx = top_idx + stride

                var top_re = ptr_re[top_idx]
                var top_im = ptr_im[top_idx]
                var bot_re = ptr_re[bot_idx]
                var bot_im = ptr_im[bot_idx]

                var sum_re = top_re + bot_re
                var sum_im = top_im + bot_im
                var diff_re = top_re - bot_re
                var diff_im = top_im - bot_im

                var w_re = ptr_fac_re[n // 2 // stride * j]
                var w_im = ptr_fac_im[n // 2 // stride * j]

                var t_re = diff_re * w_re - diff_im * w_im
                var t_im = diff_re * w_im + diff_im * w_re

                ptr_re[top_idx] = sum_re
                ptr_im[top_idx] = sum_im
                ptr_re[bot_idx] = t_re
                ptr_im[bot_idx] = t_im

        stride = stride // 2

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    var final_ptr_re = state.re.unsafe_ptr()
    var final_ptr_im = state.im.unsafe_ptr()

    @parameter
    fn norm_worker(i: Int):
        final_ptr_re[i] *= scale
        final_ptr_im[i] *= scale

    parallelize[norm_worker](n, 8)


fn fft_dif_parallel_simd_ndbuffer(
    mut state: QuantumState, inverse: Bool = False
):
    """Parallelized + Vectorized DIF FFT using NDBuffer.

    Dispatches to static implementation based on n for NDBuffer support.
    """
    var n = state.size()

    # Dispatch to purely static implementation for NDBuffer size
    @parameter
    for i in range(1, 25):  # Support up to 2^24
        if n == (1 << i):
            fft_dif_parallel_simd_ndbuffer_static[1 << i](state, inverse)
            return


fn fft_dif_parallel_simd_ndbuffer_static[
    n: Int
](mut state: QuantumState, inverse: Bool = False):
    """Static implementation using NDBuffer."""
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    # Wrap pointers in NDBuffer with static size
    # Assuming list data is contiguous and NDBuffer can wrap it.
    # Note: explicit shape required for NDBuffer

    # We use UnsafePointer to init NDBuffer from list data?
    # Or implicitly via List reference?
    # NDBuffer constructor taking List exists.

    # Since 'n' is known, we can declare NDBuffer with shape n.
    # NDBuffer[type, rank, origin, shape](data)

    # Note: QuantumState.re/im are List[FloatType].
    # But NDBuffer usually takes a pointer or reference.
    # We will use UnsafePointer to generic memory.

    # var ptr_re = state.re.unsafe_ptr()
    # var ptr_im = state.im.unsafe_ptr()

    # NDBuffer signature: NDBuffer[DType, Rank, Origin, Shape](UnsafePointer)
    # We omit origin (default) and provide shape.
    # Wait, in test_vectorize lines are: NDBuffer[dtype, 1, _, N](ptr)
    # We use that syntax.

    # Reusing factors via UnsafePointer because factors are static sized?
    # generate_factors returns List.
    # factor size is n/2.

    # var ptr_fac_re = factors_re.unsafe_ptr()
    # var ptr_fac_im = factors_im.unsafe_ptr()
    # Factors buffer
    # var buf_fac_re = NDBuffer[DType.float64, 1, 0, n // 2](ptr_fac_re)
    # Using UnsafePointer for factors to keep code similar to previous, or convert them too?
    # Let's convert them to NDBuffer too for consistency.

    var buf_re = NDBuffer[Type, 1, _, n](state.re)
    var buf_im = NDBuffer[Type, 1, _, n](state.im)
    var buf_fac_re = NDBuffer[Type, 1, _, n // 2](factors_re)
    var buf_fac_im = NDBuffer[Type, 1, _, n // 2](factors_im)

    var stride = n // 2
    for _ in range(log_n):
        # var fd = FastDiv[DType.uint32](stride)
        # alias TargetType = Scalar[FastDiv[DType.uint32].uint_type]

        if (n // 2) >= simd_width:

            @parameter
            fn stage_worker_simd(vec_idx: Int):
                var idx_base = vec_idx * simd_width

                var idxs = SIMD[DType.int64, simd_width]()
                for i in range(simd_width):
                    idxs[i] = Int64(idx_base + i)

                var group_ids = idxs // stride
                var js = idxs % stride

                var block_starts = group_ids * (2 * stride)

                var top_idxs = block_starts + js
                var bot_idxs = top_idxs + stride

                # Gather usng NDBuffer
                # NDBuffer.load[width] uses scalar index.
                # NDBuffer.gather uses SIMD index.

                # Check if we can use contiguous load (stride >= simd_width)
                # In this case, top_idxs are contiguous [base, base+1, ...].
                # bot_idxs are also contiguous [base+stride, ...].

                # Note: This optimization assumes aligned access patterns which FFT provides when stride is large.

                var top_re: SIMD[Type, simd_width]
                var top_im: SIMD[Type, simd_width]
                var bot_re: SIMD[Type, simd_width]
                var bot_im: SIMD[Type, simd_width]
                var w_re: SIMD[Type, simd_width]
                var w_im: SIMD[Type, simd_width]

                if stride >= simd_width:
                    # Contiguous path using load/store (scalar index)
                    # top_idxs[0] casts to Int automatically? top_idxs is SIMD[int64].
                    var base_top = Int(top_idxs[0])
                    var base_bot = Int(bot_idxs[0])
                    var base_w = Int((js * (n // 2 // stride))[0])

                    top_re = buf_re.load[width=simd_width](base_top)
                    top_im = buf_im.load[width=simd_width](base_top)
                    bot_re = buf_re.load[width=simd_width](base_bot)
                    bot_im = buf_im.load[width=simd_width](base_bot)

                    w_re = buf_fac_re.load[width=simd_width](base_w)
                    w_im = buf_fac_im.load[width=simd_width](base_w)

                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im

                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    buf_re.store[width=simd_width](base_top, sum_re)
                    buf_im.store[width=simd_width](base_top, sum_im)
                    buf_re.store[width=simd_width](base_bot, t_re)
                    buf_im.store[width=simd_width](base_bot, t_im)

                else:
                    # Strided path - Manual gather/scatter
                    top_re = SIMD[Type, simd_width]()
                    top_im = SIMD[Type, simd_width]()
                    bot_re = SIMD[Type, simd_width]()
                    bot_im = SIMD[Type, simd_width]()
                    w_re = SIMD[Type, simd_width]()
                    w_im = SIMD[Type, simd_width]()

                    var w_idx_vec = js * (n // 2 // stride)

                    for k in range(simd_width):
                        top_re[k] = buf_re[Int(top_idxs[k])]
                        top_im[k] = buf_im[Int(top_idxs[k])]
                        bot_re[k] = buf_re[Int(bot_idxs[k])]
                        bot_im[k] = buf_im[Int(bot_idxs[k])]
                        w_re[k] = buf_fac_re[Int(w_idx_vec[k])]
                        w_im[k] = buf_fac_im[Int(w_idx_vec[k])]

                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im

                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    for k in range(simd_width):
                        buf_re[Int(top_idxs[k])] = sum_re[k]
                        buf_im[Int(top_idxs[k])] = sum_im[k]
                        buf_re[Int(bot_idxs[k])] = t_re[k]
                        buf_im[Int(bot_idxs[k])] = t_im[k]

            parallelize[stage_worker_simd]((n // 2) // simd_width, 8)

        else:
            # Fallback scalar for small strides
            for idx in range(n // 2):
                # ... logic same as before ...
                # var block_idx, j = fd.__divmod__(TargetType(idx))

                var block_idx = idx // stride
                var j = idx % stride

                var block_start = block_idx * 2 * stride

                var top_idx = block_start + j
                var bot_idx = top_idx + stride

                var top_re = buf_re[top_idx]
                var top_im = buf_im[top_idx]
                var bot_re = buf_re[bot_idx]
                var bot_im = buf_im[bot_idx]

                var sum_re = top_re + bot_re
                var sum_im = top_im + bot_im
                var diff_re = top_re - bot_re
                var diff_im = top_im - bot_im

                var w_re = buf_fac_re[n // 2 // stride * j]
                var w_im = buf_fac_im[n // 2 // stride * j]

                var t_re = diff_re * w_re - diff_im * w_im
                var t_im = diff_re * w_im + diff_im * w_re

                buf_re[top_idx] = sum_re
                buf_im[top_idx] = sum_im
                buf_re[bot_idx] = t_re
                buf_im[bot_idx] = t_im

        stride = stride // 2

    bit_reverse_state(state)

    # Normalization with NDBuffer?
    # State data swapped? bit_reverse_state modifies state data in place usually.
    # Assuming pointers valid if in-place swap.
    # But `bit_reverse_state` might reallocate?
    # Let's check `bit_reverse_state`.
    # It creates `s_re` list and then... assigned back?
    # `state.re = s_re`. Yes, pointer invalidated!

    # Re-acquire pointer and buffer!
    # var final_ptr_re = state.re.unsafe_ptr()
    # var final_ptr_im = state.im.unsafe_ptr()
    var final_buf_re = NDBuffer[Type, 1, _, n](state.re)
    var final_buf_im = NDBuffer[Type, 1, _, n](state.im)

    var scale = FloatType(1.0) / sqrt(FloatType(n))

    @parameter
    fn norm_worker(i: Int):
        final_buf_re[i] *= scale
        final_buf_im[i] *= scale

    parallelize[norm_worker](n, 8)


fn fft_dif_parallel_ndbuffer(mut state: QuantumState, inverse: Bool = False):
    """Parallelized DIF FFT using NDBuffer.

    Dispatches to static implementation based on n for NDBuffer support.
    """
    var n = state.size()

    @parameter
    for i in range(1, 25):  # Support up to 2^24
        if n == (1 << i):
            fft_dif_parallel_ndbuffer_static[1 << i](state, inverse)
            return


fn fft_dif_parallel_ndbuffer_static[
    n: Int
](mut state: QuantumState, inverse: Bool = False):
    """Static implementation using NDBuffer (Scalar Parallel)."""
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var buf_re = NDBuffer[Type, 1, _, n](state.re)
    var buf_im = NDBuffer[Type, 1, _, n](state.im)
    var buf_fac_re = NDBuffer[Type, 1, _, n // 2](factors_re)
    var buf_fac_im = NDBuffer[Type, 1, _, n // 2](factors_im)

    var stride = n // 2
    for _ in range(log_n):
        var fd = FastDiv[DType.uint32](stride)
        alias TargetType = Scalar[FastDiv[DType.uint32].uint_type]

        @parameter
        fn stage_worker(idx: Int):
            # idx: 0 to n/2 - 1
            # var block_idx = idx // stride
            # var j = idx % stride

            var res = fd.__divmod__(TargetType(idx))
            var block_idx = Int(res[0])
            var j = Int(res[1])

            var block_start = block_idx * 2 * stride
            var top_idx = block_start + j
            var bot_idx = top_idx + stride

            var top_re = buf_re[top_idx]
            var top_im = buf_im[top_idx]
            var bot_re = buf_re[bot_idx]
            var bot_im = buf_im[bot_idx]

            var sum_re = top_re + bot_re
            var sum_im = top_im + bot_im
            var diff_re = top_re - bot_re
            var diff_im = top_im - bot_im

            var w_re = buf_fac_re[n // 2 // stride * j]
            var w_im = buf_fac_im[n // 2 // stride * j]

            var t_re = diff_re * w_re - diff_im * w_im
            var t_im = diff_re * w_im + diff_im * w_re

            buf_re[top_idx] = sum_re
            buf_im[top_idx] = sum_im
            buf_re[bot_idx] = t_re
            buf_im[bot_idx] = t_im

        parallelize[stage_worker](n // 2, 8)
        stride = stride // 2

    bit_reverse_state(state)

    var final_buf_re = NDBuffer[Type, 1, _, n](state.re)
    var final_buf_im = NDBuffer[Type, 1, _, n](state.im)

    var scale = FloatType(1.0) / sqrt(FloatType(n))

    @parameter
    fn norm_worker(i: Int):
        final_buf_re[i] *= scale
        final_buf_im[i] *= scale

    parallelize[norm_worker](n, 8)


fn fft_dif_parallel_fastdiv(mut state: QuantumState, inverse: Bool = False):
    """Parallelized DIF FFT using FastDiv (UnsafePointer)."""
    var n = state.size()
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var stride = n // 2
    for _ in range(log_n):
        var fd = FastDiv[DType.uint32](stride)
        alias TargetType = Scalar[FastDiv[DType.uint32].uint_type]

        @parameter
        fn stage_worker(idx: Int):
            var res = fd.__divmod__(TargetType(idx))
            var block_idx = Int(res[0])
            var j = Int(res[1])
            var block_start = block_idx * 2 * stride

            var top_idx = block_start + j
            var bot_idx = top_idx + stride

            var top_re = ptr_re[top_idx]
            var top_im = ptr_im[top_idx]
            var bot_re = ptr_re[bot_idx]
            var bot_im = ptr_im[bot_idx]

            var sum_re = top_re + bot_re
            var sum_im = top_im + bot_im
            var diff_re = top_re - bot_re
            var diff_im = top_im - bot_im

            var w_re = ptr_fac_re[n // 2 // stride * j]
            var w_im = ptr_fac_im[n // 2 // stride * j]

            var t_re = diff_re * w_re - diff_im * w_im
            var t_im = diff_re * w_im + diff_im * w_re

            ptr_re[top_idx] = sum_re
            ptr_im[top_idx] = sum_im
            ptr_re[bot_idx] = t_re
            ptr_im[bot_idx] = t_im

        parallelize[stage_worker](n // 2, 8)
        stride = stride // 2

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    var final_ptr_re = state.re.unsafe_ptr()
    var final_ptr_im = state.im.unsafe_ptr()

    @parameter
    fn norm_worker(i: Int):
        final_ptr_re[i] *= scale
        final_ptr_im[i] *= scale

    parallelize[norm_worker](n, 8)


fn fft_dif_parallel_ndbuffer_std(
    mut state: QuantumState, inverse: Bool = False
):
    """Parallelized DIF FFT using NDBuffer (Standard Division)."""
    var n = state.size()

    @parameter
    for i in range(1, 25):
        if n == (1 << i):
            fft_dif_parallel_ndbuffer_static_std[1 << i](state, inverse)
            return


fn fft_dif_parallel_ndbuffer_static_std[
    n: Int
](mut state: QuantumState, inverse: Bool = False):
    """Static implementation using NDBuffer (Standard Division)."""
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var buf_re = NDBuffer[Type, 1, _, n](state.re)
    var buf_im = NDBuffer[Type, 1, _, n](state.im)
    var buf_fac_re = NDBuffer[Type, 1, _, n // 2](factors_re)
    var buf_fac_im = NDBuffer[Type, 1, _, n // 2](factors_im)

    var stride = n // 2
    for _ in range(log_n):

        @parameter
        fn stage_worker(idx: Int):
            # Standard division/modulo
            var block_idx = idx // stride
            var j = idx % stride

            var block_start = block_idx * 2 * stride
            var top_idx = block_start + j
            var bot_idx = top_idx + stride

            var top_re = buf_re[top_idx]
            var top_im = buf_im[top_idx]
            var bot_re = buf_re[bot_idx]
            var bot_im = buf_im[bot_idx]

            var sum_re = top_re + bot_re
            var sum_im = top_im + bot_im
            var diff_re = top_re - bot_re
            var diff_im = top_im - bot_im

            var w_re = buf_fac_re[n // 2 // stride * j]
            var w_im = buf_fac_im[n // 2 // stride * j]

            var t_re = diff_re * w_re - diff_im * w_im
            var t_im = diff_re * w_im + diff_im * w_re

            buf_re[top_idx] = sum_re
            buf_im[top_idx] = sum_im
            buf_re[bot_idx] = t_re
            buf_im[bot_idx] = t_im

        parallelize[stage_worker](n // 2, 8)
        stride = stride // 2

    bit_reverse_state(state)

    var final_buf_re = NDBuffer[Type, 1, _, n](state.re)
    var final_buf_im = NDBuffer[Type, 1, _, n](state.im)

    var scale = FloatType(1.0) / sqrt(FloatType(n))

    @parameter
    fn norm_worker(i: Int):
        final_buf_re[i] *= scale
        final_buf_im[i] *= scale

    parallelize[norm_worker](n, 8)


fn fft_dif_parallel_simd_fastdiv(
    mut state: QuantumState, inverse: Bool = False
):
    """Parallelized + Vectorized DIF FFT using FastDiv for scalarized control logic.
    """
    var n = state.size()
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var stride = n // 2
    for _ in range(log_n):
        var fd = FastDiv[DType.uint32](stride)
        alias TargetType = Scalar[FastDiv[DType.uint32].uint_type]

        if (
            stride // 2
        ) >= simd_width:  # Optimization valid when vector fits in stride/2
            # Stride is at least 2*simd_width.
            # Stride is multiple of simd_width (powers of 2).
            # Indices are aligned to simd_width.
            # Thus a vector never crosses a stride boundary.

            @parameter
            fn stage_worker_simd(vec_idx: Int):
                var idx_base = vec_idx * simd_width

                # Scalar FastDiv
                var res = fd.__divmod__(TargetType(idx_base))
                var block_idx = Int(res[0])
                var j_base = Int(res[1])

                # Scalar calcs
                var block_start = block_idx * 2 * stride
                var top_idx_base = block_start + j_base
                var bot_idx_base = top_idx_base + stride

                # Vector indices construction (iota)
                # idxs = [top_idx_base, top_idx_base+1...]
                var top_idxs = SIMD[DType.int64, simd_width]()
                var bot_idxs = SIMD[DType.int64, simd_width]()
                for i in range(simd_width):
                    top_idxs[i] = Int64(top_idx_base + i)
                    bot_idxs[i] = Int64(bot_idx_base + i)

                # Factors index
                # w_base = (n // 2 // stride) * j_base
                # But w_idx also increments!
                # w_idx(j) = C * j.
                # w_vec = C * (j_base + iota) = C*j_base + C*iota.
                # Since factor stride (n//2//stride) is scalar, we can just do:
                # w_idxs = (j_vector) * (n // 2 // stride)

                var factor_stride = n // 2 // stride
                var j_vec = SIMD[DType.int64, simd_width]()
                for i in range(simd_width):
                    j_vec[i] = Int64(j_base + i)

                var w_idxs = j_vec * factor_stride

                # Gather data
                var top_re = ptr_re.gather(top_idxs)
                var top_im = ptr_im.gather(top_idxs)
                var bot_re = ptr_re.gather(bot_idxs)
                var bot_im = ptr_im.gather(bot_idxs)

                var sum_re = top_re + bot_re
                var sum_im = top_im + bot_im
                var diff_re = top_re - bot_re
                var diff_im = top_im - bot_im

                var w_re = ptr_fac_re.gather(w_idxs)
                var w_im = ptr_fac_im.gather(w_idxs)

                var t_re = diff_re * w_re - diff_im * w_im
                var t_im = diff_re * w_im + diff_im * w_re

                ptr_re.scatter(top_idxs, sum_re)
                ptr_im.scatter(top_idxs, sum_im)
                ptr_re.scatter(bot_idxs, t_re)
                ptr_im.scatter(bot_idxs, t_im)

            parallelize[stage_worker_simd]((n // 2) // simd_width, 8)

        else:
            # Scalar fallback for small strides
            for idx in range(n // 2):
                var res = fd.__divmod__(TargetType(idx))
                var block_idx = Int(res[0])
                var j = Int(res[1])

                var block_start = block_idx * 2 * stride
                var top_idx = block_start + j
                var bot_idx = top_idx + stride

                var top_re = ptr_re[top_idx]
                var top_im = ptr_im[top_idx]
                var bot_re = ptr_re[bot_idx]
                var bot_im = ptr_im[bot_idx]

                var sum_re = top_re + bot_re
                var sum_im = top_im + bot_im
                var diff_re = top_re - bot_re
                var diff_im = top_im - bot_im

                var w_idx = (n // 2 // stride) * j
                var w_re = ptr_fac_re[w_idx]
                var w_im = ptr_fac_im[w_idx]

                var t_re = diff_re * w_re - diff_im * w_im
                var t_im = diff_re * w_im + diff_im * w_re

                ptr_re[top_idx] = sum_re
                ptr_im[top_idx] = sum_im
                ptr_re[bot_idx] = t_re
                ptr_im[bot_idx] = t_im

        stride = stride // 2

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    var final_ptr_re = state.re.unsafe_ptr()
    var final_ptr_im = state.im.unsafe_ptr()

    @parameter
    fn norm_worker(i: Int):
        final_ptr_re[i] *= scale
        final_ptr_im[i] *= scale

    parallelize[norm_worker](n)


fn fft_dif_parallel_simd_phast(mut state: QuantumState, inverse: Bool = False):
    """Parallelized + Vectorized DIF FFT using sequential twiddle buffering (PhastFT style).
    OPTIMIZATION: Prevents strided gathers for twiddles by pre-packing the current stage's
    twiddle sequence into a contiguous buffer. Since the same twiddle sequence repeats
    for every block in the stage, this also improves cache locality for later stages.
    """
    var n = state.size()
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var stride = n // 2

    # Pre-allocate a scratch buffer for twiddles using List
    var tw_max = n // 2
    var tw_buf_re = List[FloatType](capacity=tw_max)
    var tw_buf_im = List[FloatType](capacity=tw_max)

    # Fast initialization (unsafe setting length)
    # Since we are writing to it before reading, we can just "set" the size if List supported it.
    # But for now, use resize (assuming it's faster than loop)
    # Or pointer arithmetic to set size? No private member access.
    # We will assume resize is decent, or just use unsafe_ptr to write without resizing (unsafe but we can just use capacity).
    # Wait, if we use capacity but size is 0, list might free/realloc?
    # List is struct.
    # Actually, we can just use the unsafe pointer from the start and IGNORE the size property
    # as long as we don't call append/methods that rely on size.
    # We allocated capacity. The memory is there.

    # Pre-allocate buffer (always allocate to be safe, or alloc inside if needed? List alloc is fast enough if reused/outside)
    # We moved alloc outside? Yes.

    var ptr_tw_buf_re = tw_buf_re.unsafe_ptr()
    var ptr_tw_buf_im = tw_buf_im.unsafe_ptr()

    for _ in range(log_n):
        var fd = FastDiv[DType.uint32](stride)
        alias TargetType = Scalar[FastDiv[DType.uint32].uint_type]

        var factor_stride = n // 2 // stride

        if (stride // 2) >= simd_width:
            # Tiling Optimization for Large Strides (> L2 Cache)
            # Threshold chosen based on N=22 benchmarks (Stride 1M was slow, Tiling fixed it).
            alias TILED_THRESHOLD = 262144
            alias TILE_SIZE = 4096  # 4KB chunks

            # Use Zero-Copy (Direct Gather) if strides are small enough (avoid buffer overhead)
            # Factor Stride 1 (Stage 0), 2 (Stage 1), 4 (Stage 2) are good candidates.
            if factor_stride <= 4 and stride >= TILED_THRESHOLD:
                # Tiled Zero-Copy Mode (Generic factor_stride)
                var num_items = n // 2
                var num_tiles = num_items // TILE_SIZE

                @parameter
                fn worker_tiled(tile_idx: Int):
                    var base_idx = tile_idx * TILE_SIZE
                    for k in range(0, TILE_SIZE, simd_width):
                        var idx_base = base_idx + k
                        var res = fd.__divmod__(TargetType(idx_base))
                        var block_idx = Int(res[0])
                        var j_base = Int(res[1])
                        var block_start = block_idx * 2 * stride
                        var top_idx_base = block_start + j_base
                        var bot_idx_base = top_idx_base + stride

                        var top_idxs = SIMD[DType.int64, simd_width]()
                        var bot_idxs = SIMD[DType.int64, simd_width]()
                        var fac_idxs = SIMD[DType.int64, simd_width]()

                        for i in range(simd_width):
                            top_idxs[i] = Int64(top_idx_base + i)
                            bot_idxs[i] = Int64(bot_idx_base + i)
                            fac_idxs[i] = Int64((j_base + i) * factor_stride)

                        var w_re = ptr_fac_re.gather(fac_idxs)
                        var w_im = ptr_fac_im.gather(fac_idxs)

                        var top_re = ptr_re.gather(top_idxs)
                        var top_im = ptr_im.gather(top_idxs)
                        var bot_re = ptr_re.gather(bot_idxs)
                        var bot_im = ptr_im.gather(bot_idxs)

                        var sum_re = top_re + bot_re
                        var sum_im = top_im + bot_im
                        var diff_re = top_re - bot_re
                        var diff_im = top_im - bot_im

                        var t_re = diff_re * w_re - diff_im * w_im
                        var t_im = diff_re * w_im + diff_im * w_re

                        ptr_re.scatter(top_idxs, sum_re)
                        ptr_im.scatter(top_idxs, sum_im)
                        ptr_re.scatter(bot_idxs, t_re)
                        ptr_im.scatter(bot_idxs, t_im)

                parallelize[worker_tiled](num_tiles)

                # Handle remainder if any (unlikely with power of 2 sizes, but good practice)
                # Power of 2 sizes will always be divisible by 4096 (2^12) if n >= 13.

            elif factor_stride == 1:
                # Standard Zero-Copy: Use factors directly (Origin: ptr_fac_re)
                @parameter
                fn worker_fac(vec_idx: Int):
                    var idx_base = vec_idx * simd_width
                    var res = fd.__divmod__(TargetType(idx_base))
                    var block_idx = Int(res[0])
                    var j_base = Int(res[1])
                    var block_start = block_idx * 2 * stride
                    var top_idx_base = block_start + j_base
                    var bot_idx_base = top_idx_base + stride

                    var top_idxs = SIMD[DType.int64, simd_width]()
                    var bot_idxs = SIMD[DType.int64, simd_width]()
                    for i in range(simd_width):
                        top_idxs[i] = Int64(top_idx_base + i)
                        bot_idxs[i] = Int64(bot_idx_base + i)

                    var w_re = ptr_fac_re.load[width=simd_width](j_base)
                    var w_im = ptr_fac_im.load[width=simd_width](j_base)

                    var top_re = ptr_re.gather(top_idxs)
                    var top_im = ptr_im.gather(top_idxs)
                    var bot_re = ptr_re.gather(bot_idxs)
                    var bot_im = ptr_im.gather(bot_idxs)

                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im

                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    ptr_re.scatter(top_idxs, sum_re)
                    ptr_im.scatter(top_idxs, sum_im)
                    ptr_re.scatter(bot_idxs, t_re)
                    ptr_im.scatter(bot_idxs, t_im)

                parallelize[worker_fac]((n // 2) // simd_width, 8)
            else:
                # Buffer Mode (Origin: ptr_tw_buf_re)

                # Pack first
                for idx in range(0, stride, simd_width):
                    var idx_base = idx
                    var src_idxs = SIMD[DType.int64, simd_width]()
                    for i in range(simd_width):
                        src_idxs[i] = Int64((idx_base + i) * factor_stride)
                    var val_re = ptr_fac_re.gather(src_idxs)
                    var val_im = ptr_fac_im.gather(src_idxs)
                    ptr_tw_buf_re.store(idx_base, val_re)
                    ptr_tw_buf_im.store(idx_base, val_im)

                # Run using buffer
                @parameter
                fn worker_buf(vec_idx: Int):
                    var idx_base = vec_idx * simd_width
                    var res = fd.__divmod__(TargetType(idx_base))
                    var block_idx = Int(res[0])
                    var j_base = Int(res[1])
                    var block_start = block_idx * 2 * stride
                    var top_idx_base = block_start + j_base
                    var bot_idx_base = top_idx_base + stride

                    var top_idxs = SIMD[DType.int64, simd_width]()
                    var bot_idxs = SIMD[DType.int64, simd_width]()
                    for i in range(simd_width):
                        top_idxs[i] = Int64(top_idx_base + i)
                        bot_idxs[i] = Int64(bot_idx_base + i)

                    var w_re = ptr_tw_buf_re.load[width=simd_width](j_base)
                    var w_im = ptr_tw_buf_im.load[width=simd_width](j_base)

                    var top_re = ptr_re.gather(top_idxs)
                    var top_im = ptr_im.gather(top_idxs)
                    var bot_re = ptr_re.gather(bot_idxs)
                    var bot_im = ptr_im.gather(bot_idxs)

                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im

                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    ptr_re.scatter(top_idxs, sum_re)
                    ptr_im.scatter(top_idxs, sum_im)
                    ptr_re.scatter(bot_idxs, t_re)
                    ptr_im.scatter(bot_idxs, t_im)

                parallelize[worker_buf]((n // 2) // simd_width, 8)

        else:
            # Fallback (Small Strides or Large SIMD width)
            if stride == 4:
                # Optimized Stride 4 (Vectorized Swap + Kernel)
                swap_state_to_distance_8_simd(state, 2)
                stride4_swapped_simd(
                    state, ptr_fac_re, ptr_fac_im, factor_stride
                )
                swap_state_to_distance_8_simd(state, 2)
            elif stride == 2:
                # Optimized Fused Stride 2 + 1
                swap_state_to_distance_8_simd(state, 1)
                fused_stride2_stride1_swapped(state)
                swap_state_to_distance_8_simd(state, 1)
                # Handled Stride 1 implicitly
                break
            else:
                # Fallback Scalar Loop
                for idx in range(n // 2):
                    var res = fd.__divmod__(TargetType(idx))
                    var block_idx = Int(res[0])
                    var j = Int(res[1])

                    var block_start = block_idx * 2 * stride
                    var top_idx = block_start + j
                    var bot_idx = top_idx + stride

                    var w_re = ptr_fac_re[j * factor_stride]
                    var w_im = ptr_fac_im[j * factor_stride]

                    var top_re = ptr_re[top_idx]
                    var top_im = ptr_im[top_idx]
                    var bot_re = ptr_re[bot_idx]
                    var bot_im = ptr_im[bot_idx]

                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im

                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    ptr_re[top_idx] = sum_re
                    ptr_im[top_idx] = sum_im
                    ptr_re[bot_idx] = t_re
                    ptr_im[bot_idx] = t_im

        stride = stride // 2

    # Lists freed automatically

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    var final_ptr_re = state.re.unsafe_ptr()
    var final_ptr_im = state.im.unsafe_ptr()

    @parameter
    fn norm_worker(i: Int):
        final_ptr_re[i] *= scale
        final_ptr_im[i] *= scale

    parallelize[norm_worker](n)
