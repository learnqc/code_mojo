from memory import UnsafePointer
from algorithm import parallelize
from butterfly.core.state import QuantumState
from butterfly.core.types import *


fn swap_state_to_distance_8_simd(mut state: QuantumState, target: Int):
    """
    Permutes the QuantumState such that elements originally separated by 'stride' (1 << target)
    are moved to positions separated by 8 (1 << 3).
    Vectorized implementation using SIMD gather/scatter.

    Valid for target < 3 (strides 1, 2, 4).
    """
    alias dist_val = 8
    var stride = 1 << target

    if stride == dist_val:
        return

    # We process in blocks of 16.
    # In each 16-block [0..15]:
    # - 8 elements have bit 3 (8) set: [8..15]
    # - 8 elements have bit 3 (8) clear: [0..7]
    # In the upper 8 [8..15], we check for bit 'target' (stride) clear.
    # Exactly 4 elements match.
    # We define the offsets for these 4 elements based on target.

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var n = state.size()

    # Offsets relative to the 16-block start
    var offsets: SIMD[DType.int64, 4]
    if target == 0:  # Stride 1
        # Indices in 8..15 with bit 0 clear: 8, 10, 12, 14
        offsets = SIMD[DType.int64, 4](8, 10, 12, 14)
    elif target == 1:  # Stride 2
        # Indices in 8..15 with bit 1 clear: 8, 9, 12, 13
        offsets = SIMD[DType.int64, 4](8, 9, 12, 13)
    elif target == 2:  # Stride 4
        # Indices in 8..15 with bit 2 clear: 8, 9, 10, 11
        offsets = SIMD[DType.int64, 4](8, 9, 10, 11)
    else:
        # Fallback or error (should not be called for >= 8)
        return

    @parameter
    fn worker(block_idx: Int):
        var base = block_idx * 16

        # Calculate indices i (source/upper)
        var i_idxs = offsets + base

        # Calculate indices j (peer/lower)
        # j = i ^ 8 ^ stride
        # XOR with constant vector
        var xor_mask = Int64(dist_val ^ stride)
        var j_idxs = i_idxs ^ xor_mask

        # Perform swap Real
        var val_i_re = ptr_re.gather(i_idxs)
        var val_j_re = ptr_re.gather(j_idxs)
        ptr_re.scatter(j_idxs, val_i_re)
        ptr_re.scatter(i_idxs, val_j_re)

        # Perform swap Imag
        var val_i_im = ptr_im.gather(i_idxs)
        var val_j_im = ptr_im.gather(j_idxs)
        ptr_im.scatter(j_idxs, val_i_im)
        ptr_im.scatter(i_idxs, val_j_im)

    parallelize[worker](n // 16)


fn swap_state_to_distance_4_simd(mut state: QuantumState, target: Int):
    """
    Permutes the QuantumState such that elements originally separated by 'stride' (1 << target)
    are moved to positions separated by 4 (1 << 2).
    Vectorized implementation using SIMD gather/scatter.

    Valid for target < 2 (strides 1, 2).
    """
    alias dist_val = 4
    var stride = 1 << target

    if stride == dist_val:
        return

    # We process in blocks of 8.
    # In each 8-block [0..7]:
    # - 4 elements have bit 2 (4) set: [4..7]
    # - 4 elements have bit 2 (4) clear: [0..3]
    # In the upper 4 [4..7], we check for bit 'target' (stride) clear.
    # Exactly 2 elements match.

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var n = state.size()

    # Offsets relative to the 8-block start
    # We want indices in 4..7 where bit 'target' is 0.
    var offsets: SIMD[DType.int64, 2]
    if target == 0:  # Stride 1
        # Indices in 4..7 with bit 0 clear: 4, 6
        offsets = SIMD[DType.int64, 2](4, 6)
    elif target == 1:  # Stride 2
        # Indices in 4..7 with bit 1 clear: 4, 5
        offsets = SIMD[DType.int64, 2](4, 5)
    else:
        return

    @parameter
    fn worker(block_idx: Int):
        var base = block_idx * 8

        # Calculate indices i (source/upper)
        var i_idxs = offsets + base

        # Calculate indices j (peer/lower)
        # j = i ^ 4 ^ stride
        var xor_mask = Int64(dist_val ^ stride)
        var j_idxs = i_idxs ^ xor_mask

        # Perform swap Real
        var val_i_re = ptr_re.gather(i_idxs)
        var val_j_re = ptr_re.gather(j_idxs)
        ptr_re.scatter(j_idxs, val_i_re)
        ptr_re.scatter(i_idxs, val_j_re)

        # Perform swap Imag
        var val_i_im = ptr_im.gather(i_idxs)
        var val_j_im = ptr_im.gather(j_idxs)
        ptr_im.scatter(j_idxs, val_i_im)
        ptr_im.scatter(i_idxs, val_j_im)

    parallelize[worker](n // 8)


fn swap_bits_simd(mut state: QuantumState, bit_a: Int, bit_b: Int):
    """
    Swaps bit 'bit_a' and 'bit_b' for all indices in the state using SIMD.
    Generalized bit swap permutation.
    """
    if bit_a == bit_b:
        return

    # Ensure a < b
    var a = bit_a
    var b = bit_b
    if a > b:
        a = bit_b
        b = bit_a

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var n = state.size()
    var n_pairs = n >> 2

    @parameter
    fn worker(idx: Int):
        alias width = 4
        var base_k = idx * width

        # Construct k_vec [base, base+1...]
        var k_vec = SIMD[DType.int64, width]()
        for i in range(width):
            k_vec[i] = Int64(base_k + i)

        # Insert 0 at bit position 'a'
        # Expands k from N-2 bits to N-1 bits
        var mask_a = (1 << a) - 1
        var lo_a = k_vec & mask_a
        var hi_a = k_vec >> a
        var k_exp_a = lo_a | (hi_a << (a + 1))

        # Insert 0 at bit position 'b'
        # Expands to N bits
        var mask_b = (1 << b) - 1
        var lo_b = k_exp_a & mask_b
        var hi_b = k_exp_a >> b
        var idx00 = lo_b | (hi_b << (b + 1))

        var idx_a = idx00 | (1 << a)  # bit a=1, b=0
        var idx_b = idx00 | (1 << b)  # bit a=0, b=1

        # Perform Swap
        var val_a_re = ptr_re.gather(idx_a)
        var val_b_re = ptr_re.gather(idx_b)
        ptr_re.scatter(idx_b, val_a_re)
        ptr_re.scatter(idx_a, val_b_re)

        var val_a_im = ptr_im.gather(idx_a)
        var val_b_im = ptr_im.gather(idx_b)
        ptr_im.scatter(idx_b, val_a_im)
        ptr_im.scatter(idx_a, val_b_im)

    # Number of SIMD blocks
    parallelize[worker](n_pairs // 4)
