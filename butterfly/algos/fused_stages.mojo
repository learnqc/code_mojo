from butterfly.core.state import QuantumState
from butterfly.core.types import Type, FloatType, float_bytes
from butterfly.algos.adjacent_pairs import insert_zero_bit
from algorithm import parallelize
from sys.info import simd_width_of, bitwidthof


fn fused_stage0_swap_simd(
    mut state: QuantumState,
    factors_re: List[FloatType],
    factors_im: List[FloatType],
    target_bit: Int,
):
    """
    Performs Stage 0 Butterfly (Stride N/2) AND Permutes Bit 'target_bit' (N/2) with Bit 2 (Val 4).
    """
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var n = state.size()
    var stride = 1 << target_bit

    alias simd_width = simd_width_of[Type]()

    var limit = n // 4
    alias target_block_bytes = 4096
    var block_size = target_block_bytes // float_bytes
    var num_blocks = (limit + block_size - 1) // block_size

    @parameter
    fn worker(idx: Int):
        var start = idx * block_size
        var end = start + block_size
        if end > limit:
            end = limit

        for k in range(start, end, simd_width):
            # Optimize: Calculate scalar base indices and use contiguous load/store
            # Since simd_width <= 4 (usually) and we skip bit 2 (val 4), the indices in q are contiguous.
            # q[0] is the start index.

            # Compute Scalar q start
            # k is the start of the block.
            # insert_zero_bit(k, 2).
            # But insert_zero_bit is vector function?
            # We can use the vector result q[0].

            # Recalculate q vector for correctness ref but use q[0] for loading
            var k_vec = SIMD[DType.int64, simd_width]()
            for i in range(simd_width):
                k_vec[i] = Int64(k + i)

            var q_vec = insert_zero_bit(k_vec, 2)
            q_vec = insert_zero_bit(q_vec, target_bit)

            # Extract scalar bases
            var idx_00_base = q_vec[0]
            var idx_01_base = idx_00_base | 4
            var idx_10_base = idx_00_base | stride
            var idx_11_base = idx_10_base | 4

            # Contiguous Loads
            var re_00 = ptr_re.load[width=simd_width](idx_00_base)
            var im_00 = ptr_im.load[width=simd_width](idx_00_base)
            var re_01 = ptr_re.load[width=simd_width](idx_01_base)
            var im_01 = ptr_im.load[width=simd_width](idx_01_base)
            var re_10 = ptr_re.load[width=simd_width](idx_10_base)
            var im_10 = ptr_im.load[width=simd_width](idx_10_base)
            var re_11 = ptr_re.load[width=simd_width](idx_11_base)
            var im_11 = ptr_im.load[width=simd_width](idx_11_base)

            var w_re_00 = ptr_fac_re.load[width=simd_width](idx_00_base)
            var w_im_00 = ptr_fac_im.load[width=simd_width](idx_00_base)
            var w_re_01 = ptr_fac_re.load[width=simd_width](idx_01_base)
            var w_im_01 = ptr_fac_im.load[width=simd_width](idx_01_base)

            var sum_re_1 = re_00 + re_10
            var sum_im_1 = im_00 + im_10
            var diff_re_1 = re_00 - re_10
            var diff_im_1 = im_00 - im_10

            var t_re_1 = diff_re_1 * w_re_00 - diff_im_1 * w_im_00
            var t_im_1 = diff_re_1 * w_im_00 + diff_im_1 * w_re_00

            var sum_re_2 = re_01 + re_11
            var sum_im_2 = im_01 + im_11
            var diff_re_2 = re_01 - re_11
            var diff_im_2 = im_01 - im_11

            var t_re_2 = diff_re_2 * w_re_01 - diff_im_2 * w_im_01
            var t_im_2 = diff_re_2 * w_im_01 + diff_im_2 * w_re_01

            # Store Permuted: N/2 <-> 4 (Delayed Swap)
            # sum_re_1 (L 00/0) -> idx_00 (P 00).
            ptr_re.store[width=simd_width](idx_00_base, sum_re_1)
            ptr_im.store[width=simd_width](idx_00_base, sum_im_1)
            # t_re_2 (L 11/12) -> idx_11 (P 11).
            ptr_re.store[width=simd_width](idx_11_base, t_re_2)
            ptr_im.store[width=simd_width](idx_11_base, t_im_2)

            # sum_re_2 (L 10/4) -> idx_10 (P 10). (Start Permutation here: L4 to P8)
            ptr_re.store[width=simd_width](idx_10_base, sum_re_2)
            ptr_im.store[width=simd_width](idx_10_base, sum_im_2)

            # t_re_1 (L 01/8) -> idx_01 (P 01). (L8 to P4)
            ptr_re.store[width=simd_width](idx_01_base, t_re_1)
            ptr_im.store[width=simd_width](idx_01_base, t_im_1)

    parallelize[worker](num_blocks)


fn fused_restore_order_simd(
    mut state: QuantumState,
    factors_re: List[FloatType],
    factors_im: List[FloatType],
    target_bit: Int,
):
    """
    Performs Stage 'Stride 4' (Logical) but on data that resides at Stride N/2 (Physical).
    Then Swaps Bit 'target_bit' (N/2) with Bit 2 (Val 4) to restore canonical order.
    """
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var n = state.size()
    var stride = 1 << target_bit

    alias simd_width = simd_width_of[Type]()

    # Twiddle factor stride for Stage 4 (Stride 4).
    # Logic: Factor Stride = N / 2 / 4 = N/8.
    var factor_stride = n // 8

    var limit = n // 4
    alias target_block_bytes = 4096
    var block_size = target_block_bytes // float_bytes
    var num_blocks = (limit + block_size - 1) // block_size

    @parameter
    fn worker(idx: Int):
        var start = idx * block_size
        var end = start + block_size
        if end > limit:
            end = limit

        for k in range(start, end, simd_width):
            # Recalculate scalar bases for contiguous load
            var k_vec = SIMD[DType.int64, simd_width]()
            for i in range(simd_width):
                k_vec[i] = Int64(k + i)

            var q_vec = insert_zero_bit(k_vec, 2)
            q_vec = insert_zero_bit(q_vec, target_bit)

            var idx_00_base = q_vec[0]
            var idx_01_base = idx_00_base | 4
            var idx_10_base = idx_00_base | stride
            var idx_11_base = idx_10_base | 4

            # Contiguous Loads for Data
            var re_00 = ptr_re.load[width=simd_width](idx_00_base)
            var im_00 = ptr_im.load[width=simd_width](idx_00_base)
            var re_01 = ptr_re.load[width=simd_width](idx_01_base)
            var im_01 = ptr_im.load[width=simd_width](idx_01_base)
            var re_10 = ptr_re.load[width=simd_width](idx_10_base)
            var im_10 = ptr_im.load[width=simd_width](idx_10_base)
            var re_11 = ptr_re.load[width=simd_width](idx_11_base)
            var im_11 = ptr_im.load[width=simd_width](idx_11_base)

            # Twiddles (Must Gather as they are strided)
            var j1 = q_vec & 3
            var w_idx_1 = j1 * factor_stride
            var w_re_1 = ptr_fac_re.gather(w_idx_1)
            var w_im_1 = ptr_fac_im.gather(w_idx_1)

            var w_re_2 = w_re_1
            var w_im_2 = w_im_1

            var sum_re_1 = re_00 + re_10
            var sum_im_1 = im_00 + im_10
            var diff_re_1 = re_00 - re_10
            var diff_im_1 = im_00 - im_10

            var t_re_1 = diff_re_1 * w_re_1 - diff_im_1 * w_im_1
            var t_im_1 = diff_re_1 * w_im_1 + diff_im_1 * w_re_1

            var sum_re_2 = re_01 + re_11
            var sum_im_2 = im_01 + im_11
            var diff_re_2 = re_01 - re_11
            var diff_im_2 = im_01 - im_11

            var t_re_2 = diff_re_2 * w_re_2 - diff_im_2 * w_im_2
            var t_im_2 = diff_re_2 * w_im_2 + diff_im_2 * w_re_2

            # Store Permuted (Unpermute)
            ptr_re.store[width=simd_width](idx_00_base, sum_re_1)
            ptr_im.store[width=simd_width](idx_00_base, sum_im_1)
            # T2 -> 11
            ptr_re.store[width=simd_width](idx_11_base, t_re_2)
            ptr_im.store[width=simd_width](idx_11_base, t_im_2)

            # T1 (L4) -> 01 (P4)
            ptr_re.store[width=simd_width](idx_01_base, t_re_1)
            ptr_im.store[width=simd_width](idx_01_base, t_im_1)

            # Sum2 (L8) -> 10 (P8)
            ptr_re.store[width=simd_width](idx_10_base, sum_re_2)
            ptr_im.store[width=simd_width](idx_10_base, sum_im_2)

    parallelize[worker](num_blocks)


fn fused_stride2_stride1_swapped(mut state: QuantumState):
    """
    Fused Kernel for Stride 2 and Stride 1.
    Assumes Stride 2 (Bit 1) has been swapped to Stride 4 (Bit 2/Distance 4).
    Stride 1 (Bit 0) remains at Bit 0.

    Processing 8 elements per block (2 vectors of width 4).
    """
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var n = state.size()

    alias width = 4
    # Process 8 elements (2 * 4) per iter?
    # Logic operates on chunks of 8 (Physical 0..7).
    # Since we swapped 2<->4, Logic Stride 2 pairs (0, 4) etc.
    # Logic Stride 1 pairs (0, 1).

    var limit = n
    alias target_block_bytes = 4096
    var block_size = target_block_bytes // float_bytes
    var num_blocks = (limit + block_size - 1) // block_size

    @parameter
    fn worker(idx: Int):
        var start = idx * block_size
        var end = start + block_size
        if end > limit:
            end = limit

        for base in range(start, end, 8):
            # Load 8 elements as 2 vectors of width 4
            # v0: indices 0..3. v1: indices 4..7
            var re0 = ptr_re.load[width=width](base)
            var im0 = ptr_im.load[width=width](base)
            var re1 = ptr_re.load[width=width](base + 4)
            var im1 = ptr_im.load[width=width](base + 4)

            # --- Stage Stride 2 (Logical) ---
            # Physical: Stride 4. Pairs v0 and v1.
            # Twiddles: Depend on Logical j (Bit 0).
            # Indices 0, 2 (Bit 0=0) -> Twiddle 1.
            # Indices 1, 3 (Bit 0=1) -> Twiddle -i.

            # Butterfly
            var sum_re = re0 + re1
            var sum_im = im0 + im1
            var diff_re = re0 - re1
            var diff_im = im0 - im1

            # Apply Twiddles to Diff
            # -i means (re, im) -> (im, -re)
            # 1 means (re, im) -> (re, im)
            # Mask for Bit 0=1: Indices 1, 3.
            # Construct SIMD mask or blend?
            # Or shuffle.
            # diff values at 1, 3 need -i mult.
            # t_re = diff_re. t_im = diff_im.
            # For 1, 3: t_re = diff_im, t_im = -diff_re.

            # Manual shuffle/select
            var t_re = diff_re
            var t_im = diff_im

            # Element 1
            t_re[1] = diff_im[1]
            t_im[1] = -diff_re[1]
            # Element 3
            t_re[3] = diff_im[3]
            t_im[3] = -diff_re[3]

            # Update diff vars for next stage
            var s2_re_top = sum_re
            var s2_im_top = sum_im
            var s2_re_bot = t_re
            var s2_im_bot = t_im

            # --- Stage Stride 1 ---
            # Stride 1 (Physical 1). Pairs (i, i+1).
            # Vector [0, 1, 2, 3]. Pairs (0,1) and (2,3).
            # No Twiddles.

            # Perform on Top (Sum) and Bot (Diff) vectors independently.

            # Manual construction instead of shuffle
            var top_lhs_re = SIMD[Type, 4]()
            var top_lhs_im = SIMD[Type, 4]()
            var top_rhs_re = SIMD[Type, 4]()
            var top_rhs_im = SIMD[Type, 4]()

            # Mask Even: 0, 0, 2, 2
            top_lhs_re[0] = s2_re_top[0]
            top_lhs_re[1] = s2_re_top[0]
            top_lhs_re[2] = s2_re_top[2]
            top_lhs_re[3] = s2_re_top[2]

            top_lhs_im[0] = s2_im_top[0]
            top_lhs_im[1] = s2_im_top[0]
            top_lhs_im[2] = s2_im_top[2]
            top_lhs_im[3] = s2_im_top[2]

            # Mask Odd: 1, 1, 3, 3
            top_rhs_re[0] = s2_re_top[1]
            top_rhs_re[1] = s2_re_top[1]
            top_rhs_re[2] = s2_re_top[3]
            top_rhs_re[3] = s2_re_top[3]

            top_rhs_im[0] = s2_im_top[1]
            top_rhs_im[1] = s2_im_top[1]
            top_rhs_im[2] = s2_im_top[3]
            top_rhs_im[3] = s2_im_top[3]

            var t_sum_re = top_lhs_re + top_rhs_re
            var t_sum_im = top_lhs_im + top_rhs_im
            var t_diff_re = top_lhs_re - top_rhs_re
            var t_diff_im = top_lhs_im - top_rhs_im

            # Manual assign final
            var final_re_top = s2_re_top
            var final_im_top = s2_im_top

            final_re_top[0] = t_sum_re[0]
            final_re_top[1] = t_diff_re[0]
            final_re_top[2] = t_sum_re[2]
            final_re_top[3] = t_diff_re[2]

            final_im_top[0] = t_sum_im[0]
            final_im_top[1] = t_diff_im[0]
            final_im_top[2] = t_sum_im[2]
            final_im_top[3] = t_diff_im[2]

            # Apply same to s2_bot -> final_bot
            var bot_lhs_re = SIMD[Type, 4]()
            var bot_lhs_im = SIMD[Type, 4]()
            var bot_rhs_re = SIMD[Type, 4]()
            var bot_rhs_im = SIMD[Type, 4]()

            # Mask Even
            bot_lhs_re[0] = s2_re_bot[0]
            bot_lhs_re[1] = s2_re_bot[0]
            bot_lhs_re[2] = s2_re_bot[2]
            bot_lhs_re[3] = s2_re_bot[2]

            bot_lhs_im[0] = s2_im_bot[0]
            bot_lhs_im[1] = s2_im_bot[0]
            bot_lhs_im[2] = s2_im_bot[2]
            bot_lhs_im[3] = s2_im_bot[2]

            # Mask Odd
            bot_rhs_re[0] = s2_re_bot[1]
            bot_rhs_re[1] = s2_re_bot[1]
            bot_rhs_re[2] = s2_re_bot[3]
            bot_rhs_re[3] = s2_re_bot[3]

            bot_rhs_im[0] = s2_im_bot[1]
            bot_rhs_im[1] = s2_im_bot[1]
            bot_rhs_im[2] = s2_im_bot[3]
            bot_rhs_im[3] = s2_im_bot[3]

            var b_sum_re = bot_lhs_re + bot_rhs_re
            var b_sum_im = bot_lhs_im + bot_rhs_im
            var b_diff_re = bot_lhs_re - bot_rhs_re
            var b_diff_im = bot_lhs_im - bot_rhs_im

            var final_re_bot = s2_re_bot
            var final_im_bot = s2_im_bot

            final_re_bot[0] = b_sum_re[0]
            final_re_bot[1] = b_diff_re[0]
            final_re_bot[2] = b_sum_re[2]
            final_re_bot[3] = b_diff_re[2]

            final_im_bot[0] = b_sum_im[0]
            final_im_bot[1] = b_diff_im[0]
            final_im_bot[2] = b_sum_im[2]
            final_im_bot[3] = b_diff_im[2]

            # Store
            ptr_re.store[width=width](base, final_re_top)
            ptr_im.store[width=width](base, final_im_top)
            ptr_re.store[width=width](base + 4, final_re_bot)
            ptr_im.store[width=width](base + 4, final_im_bot)

    parallelize[worker](num_blocks)


fn fused_restore_and_tail_simd(
    mut state: QuantumState,
    factors_re: List[FloatType],
    factors_im: List[FloatType],
    target_bit: Int,
):
    """
    Super Fused Kernel:
    1. Restore Order Butterfly (Stride 4) + Unswap.
    2. Stride 2 Butterfly (Trivial Twiddles).
    3. Stride 1 Butterfly (Trivial Twiddles).

    Processes 3 stages in one pass!
    """
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var n = state.size()
    var stride = 1 << target_bit

    alias width = 4
    # Factor Stride for Stride 4 step (N/8)
    var factor_stride = n // 8

    var limit = n // 4
    alias target_block_bytes = 4096
    var block_size = target_block_bytes // float_bytes
    var num_blocks = (limit + block_size - 1) // block_size

    @parameter
    fn worker(idx: Int):
        var start = idx * block_size
        var end = start + block_size
        if end > limit:
            end = limit

        for k in range(start, end, width):
            # --- 1. Load & Restore Stride 4 ---
            var k_vec = SIMD[DType.int64, width]()
            for i in range(width):
                k_vec[i] = Int64(k + i)

            var q_vec = insert_zero_bit(k_vec, 2)
            q_vec = insert_zero_bit(q_vec, target_bit)

            var idx_00_base = q_vec[0]
            var idx_01_base = idx_00_base | 4
            var idx_10_base = idx_00_base | stride
            var idx_11_base = idx_10_base | 4

            # Contiguous Loads
            var re_00 = ptr_re.load[width=width](idx_00_base)
            var im_00 = ptr_im.load[width=width](idx_00_base)
            var re_01 = ptr_re.load[width=width](idx_01_base)  # P4
            var im_01 = ptr_im.load[width=width](idx_01_base)
            var re_10 = ptr_re.load[width=width](idx_10_base)  # P8
            var im_10 = ptr_im.load[width=width](idx_10_base)
            var re_11 = ptr_re.load[width=width](idx_11_base)
            var im_11 = ptr_im.load[width=width](idx_11_base)

            # Twiddles (Stride 4)
            var j1 = q_vec & 3
            var w_idx_1 = j1 * factor_stride
            var w_re_1 = ptr_fac_re.gather(w_idx_1)
            var w_im_1 = ptr_fac_im.gather(w_idx_1)

            # Pair 1: P0 (L0) and P8 (L4). -> L0, L4.
            var sum_re_1 = re_00 + re_10
            var sum_im_1 = im_00 + im_10
            var diff_re_1 = re_00 - re_10
            var diff_im_1 = im_00 - im_10

            var v0_re = sum_re_1  # L0 (0..3)
            var v0_im = sum_im_1
            var v1_re = diff_re_1 * w_re_1 - diff_im_1 * w_im_1  # L4 (4..7)
            var v1_im = diff_re_1 * w_im_1 + diff_im_1 * w_re_1

            # Pair 2: P4 (L8) and P12 (L12). -> L8, L12.
            # Twiddle same? Yes (j depends on bit 0,1).
            var w_re_2 = w_re_1
            var w_im_2 = w_im_1

            var sum_re_2 = re_01 + re_11  # P4 + P12
            var sum_im_2 = im_01 + im_11
            var diff_re_2 = re_01 - re_11
            var diff_im_2 = im_01 - im_11

            var v2_re = sum_re_2  # L8 (8..11)
            var v2_im = sum_im_2
            var v3_re = diff_re_2 * w_re_2 - diff_im_2 * w_im_2  # L12 (12..15)
            var v3_im = diff_re_2 * w_im_2 + diff_im_2 * w_re_2

            # --- 2. Stride 2 & 1 (Tail) ---
            # Process v0, v1, v2, v3 independently.

            # UNROLL 4 times (for v0, v1, v2, v3).

            # --- Block v0 ---
            # Stride 2
            # Manual Element Access (Robust) for v0
            var v0_re_out = v0_re
            var v0_im_out = v0_im

            # Idx 0 (Sum(0,2)):
            var sum0_re = v0_re[0] + v0_re[2]
            var sum0_im = v0_im[0] + v0_im[2]
            var diff0_re = v0_re[0] - v0_re[2]
            var diff0_im = v0_im[0] - v0_im[2]

            # Idx 1 (Sum(1,3)):
            var sum1_re = v0_re[1] + v0_re[3]
            var sum1_im = v0_im[1] + v0_im[3]
            var diff1_re = v0_re[1] - v0_re[3]
            var diff1_im = v0_im[1] - v0_im[3]

            # Apply Twiddles to Diffs
            # Diff0 (Bit 0=0) -> * 1.
            # Diff1 (Bit 0=1) -> * -i. (re, im) -> (im, -re).
            var t_diff1_re = diff1_im
            var t_diff1_im = -diff1_re

            # Stride 1 (on local vector [Sum0, Sum1, Diff0, Diff1])
            # Pair 0 (0,1) -> (Sum0, Sum1).
            var f0_re = sum0_re + sum1_re
            var f0_im = sum0_im + sum1_im
            var f1_re = sum0_re - sum1_re
            var f1_im = sum0_im - sum1_im

            # Pair 2 (2,3) -> (Diff0, T_Diff1).
            var f2_re = diff0_re + t_diff1_re
            var f2_im = diff0_im + t_diff1_im
            var f3_re = diff0_re - t_diff1_re
            var f3_im = diff0_im - t_diff1_im

            # Store Final v0
            v0_re_out[0] = f0_re
            v0_re_out[1] = f1_re
            v0_re_out[2] = f2_re
            v0_re_out[3] = f3_re
            v0_im_out[0] = f0_im
            v0_im_out[1] = f1_im
            v0_im_out[2] = f2_im
            v0_im_out[3] = f3_im

            # --- Block v1 (Clone logic) ---
            var v1_re_out = v1_re
            var v1_im_out = v1_im

            sum0_re = v1_re[0] + v1_re[2]
            sum0_im = v1_im[0] + v1_im[2]
            diff0_re = v1_re[0] - v1_re[2]
            diff0_im = v1_im[0] - v1_im[2]
            sum1_re = v1_re[1] + v1_re[3]
            sum1_im = v1_im[1] + v1_im[3]
            diff1_re = v1_re[1] - v1_re[3]
            diff1_im = v1_im[1] - v1_im[3]

            t_diff1_re = diff1_im
            t_diff1_im = -diff1_re

            f0_re = sum0_re + sum1_re
            f0_im = sum0_im + sum1_im
            f1_re = sum0_re - sum1_re
            f1_im = sum0_im - sum1_im
            f2_re = diff0_re + t_diff1_re
            f2_im = diff0_im + t_diff1_im
            f3_re = diff0_re - t_diff1_re
            f3_im = diff0_im - t_diff1_im

            v1_re_out[0] = f0_re
            v1_re_out[1] = f1_re
            v1_re_out[2] = f2_re
            v1_re_out[3] = f3_re
            v1_im_out[0] = f0_im
            v1_im_out[1] = f1_im
            v1_im_out[2] = f2_im
            v1_im_out[3] = f3_im

            # --- Block v2 ---
            var v2_re_out = v2_re
            var v2_im_out = v2_im

            sum0_re = v2_re[0] + v2_re[2]
            sum0_im = v2_im[0] + v2_im[2]
            diff0_re = v2_re[0] - v2_re[2]
            diff0_im = v2_im[0] - v2_im[2]
            sum1_re = v2_re[1] + v2_re[3]
            sum1_im = v2_im[1] + v2_im[3]
            diff1_re = v2_re[1] - v2_re[3]
            diff1_im = v2_im[1] - v2_im[3]

            t_diff1_re = diff1_im
            t_diff1_im = -diff1_re

            f0_re = sum0_re + sum1_re
            f0_im = sum0_im + sum1_im
            f1_re = sum0_re - sum1_re
            f1_im = sum0_im - sum1_im
            f2_re = diff0_re + t_diff1_re
            f2_im = diff0_im + t_diff1_im
            f3_re = diff0_re - t_diff1_re
            f3_im = diff0_im - t_diff1_im

            v2_re_out[0] = f0_re
            v2_re_out[1] = f1_re
            v2_re_out[2] = f2_re
            v2_re_out[3] = f3_re
            v2_im_out[0] = f0_im
            v2_im_out[1] = f1_im
            v2_im_out[2] = f2_im
            v2_im_out[3] = f3_im

            # --- Block v3 ---
            var v3_re_out = v3_re
            var v3_im_out = v3_im

            sum0_re = v3_re[0] + v3_re[2]
            sum0_im = v3_im[0] + v3_im[2]
            diff0_re = v3_re[0] - v3_re[2]
            diff0_im = v3_im[0] - v3_im[2]
            sum1_re = v3_re[1] + v3_re[3]
            sum1_im = v3_im[1] + v3_im[3]
            diff1_re = v3_re[1] - v3_re[3]
            diff1_im = v3_im[1] - v3_im[3]

            t_diff1_re = diff1_im
            t_diff1_im = -diff1_re

            f0_re = sum0_re + sum1_re
            f0_im = sum0_im + sum1_im
            f1_re = sum0_re - sum1_re
            f1_im = sum0_im - sum1_im
            f2_re = diff0_re + t_diff1_re
            f2_im = diff0_im + t_diff1_im
            f3_re = diff0_re - t_diff1_re
            f3_im = diff0_im - t_diff1_im

            v3_re_out[0] = f0_re
            v3_re_out[1] = f1_re
            v3_re_out[2] = f2_re
            v3_re_out[3] = f3_re
            v3_im_out[0] = f0_im
            v3_im_out[1] = f1_im
            v3_im_out[2] = f2_im
            v3_im_out[3] = f3_im

            # --- 4. Store (Canonical) ---
            # P0 (idx_00) gets v0 (L0..3)
            ptr_re.store[width=width](idx_00_base, v0_re_out)
            ptr_im.store[width=width](idx_00_base, v0_im_out)

            # P4 (idx_01) gets v1 (L4..7)
            ptr_re.store[width=width](idx_01_base, v1_re_out)
            ptr_im.store[width=width](idx_01_base, v1_im_out)

            # P8 (idx_10) gets v2 (L8..11)
            ptr_re.store[width=width](idx_10_base, v2_re_out)
            ptr_im.store[width=width](idx_10_base, v2_im_out)

            # P12 (idx_11) gets v3 (L12..15)
            ptr_re.store[width=width](idx_11_base, v3_re_out)
            ptr_im.store[width=width](idx_11_base, v3_im_out)

    parallelize[worker](num_blocks)
