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
