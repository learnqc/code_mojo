from testing import assert_equal
from stdlib.bit.bit import pop_count
from algorithm import parallelize
from stdlib.bit.bit import count_trailing_zeros
from sys.info import simd_width_of


fn swap_to_adjacent_pairs[
    T: ImplicitlyCopyable & Movable
](mut lst: List[T], target: UInt) raises:
    assert_equal(pop_count(len(lst)), 1)
    stride = 1 << Int(target)

    r = 0
    for j in range(0, len(lst), 4):
        idx = j - r

        lst.swap_elements(idx + 1, idx + stride)

        r += 2
        if r == stride:
            r = 0


fn swap_to_new_stride[
    T: ImplicitlyCopyable & Movable
](mut lst: List[T], stride: Int, new_stride: Int) raises:
    assert_equal(pop_count(len(lst)), 1)
    assert_true(stride != new_stride)

    r = 0
    for j in range(0, len(lst), 4):
        idx = j - r

        lst.swap_elements(idx + new_stride, idx + stride)

        r += 2
        if r == stride:
            r = 0


fn swap_to_distance_8[
    T: ImplicitlyCopyable & Movable
](mut lst: List[T], target: UInt) raises:
    """
    Permutes the list such that elements originally separated by 'stride' (1 << target)
    are moved to positions separated by 8 (1 << 3).
    Effectively swaps bit 'target' with bit 3 in the index.
    """
    assert_equal(pop_count(len(lst)), 1)

    alias dist_val = 8
    stride = 1 << Int(target)

    if stride == dist_val:
        return

    # Swap pairs (i, j) where i has bit 3 (Val 8) set and bit 'target' clear,
    # and j has bit 3 clear and bit 'target' set.
    # j = i ^ 8 ^ stride
    for i in range(len(lst)):
        if (i & dist_val) != 0 and (i & stride) == 0:
            var j = i ^ dist_val ^ stride
            lst.swap_elements(i, j)


from butterfly.core.state import QuantumState


fn swap_state_to_distance_8(mut state: QuantumState, target: Int):
    """
    Permutes the QuantumState such that elements originally separated by 'stride' (1 << target)
    are moved to positions separated by 8 (1 << 3).
    Operates on both real and imaginary parts.
    """
    swap_state_to_distance(state, target, 8)


@always_inline
fn insert_zero_bit(val: Int, pos: Int) -> Int:
    """Inserts a 0 bit at position 'pos' in 'val'.
    Bits at stride >= pos are shifted left.
    """
    var high_mask = -1 << pos
    var high = val & high_mask
    var low = val & ~high_mask
    return (high << 1) | low


@always_inline
fn insert_zero_bit[
    width: Int
](val: SIMD[DType.int64, width], pos: Int) -> SIMD[DType.int64, width]:
    var high_mask = Int64(-1) << pos
    var high = val & high_mask
    var low = val & ~high_mask
    return (high << 1) | low


fn swap_state_to_distance(mut state: QuantumState, target: Int, dist_val: Int):
    stride = 1 << target

    if stride == dist_val:
        return

    # Calculate bit positions
    # We want to insert 0 at bit 'target' (stride) and bit 'dist_pos' (dist_val)
    # dist_val is typically 8 (bit 3). But we should be general.
    # count_trailing_zeros requires a non-zero input.
    # Stride and dist_val are powers of 2.
    var pos_a = Int(count_trailing_zeros(stride))
    var pos_b = Int(count_trailing_zeros(dist_val))

    var low_pos = pos_a
    var high_pos = pos_b
    if pos_a > pos_b:
        low_pos = pos_b
        high_pos = pos_a

    # We construct indices where bit 'low_pos' and bit 'high_pos' are BOTH 0.
    # We iterate k from 0 to N/4 (size >> 2).
    # Then we construct `idx` by inserting 0 at low_pos, then inserting 0 at high_pos (adjusted).
    # Note: after inserting at low_pos, the original high_pos bit moves to high_pos + 1 if high_pos >= low_pos.
    # But insert_zero_bit logic handles "pos" relative to the current value.
    # If we insert at low_pos first (smaller), the remaining bits shift up.
    # The effective position of high_pos in the *compressed* value is (high_pos - 1).
    # Let's check:
    # Value k has N-2 bits.
    # We insert 0 at low_pos -> creates N-1 bit value with 0 at low_pos.
    # Then we insert 0 at high_pos -> creates N bit value.
    # Since high_pos > low_pos, the second insertion point in the INTERMEDIATE value is just high_pos - 1.
    # Wait, insert_zero_bit splits at 'pos'.
    # If we want 0 at bit L and bit H (where H > L):
    # k has bits 0..N-3.
    # We want to place k's bits into holes.
    # Hole at L: k[0:L] goes to res[0:L].
    # Hole at H: construction is easiest if we do largest hole first?
    # No, let's step through.
    # Result should have 0 at L and 0 at H.
    # k = ... b_H ... b_L ... (relative to original holes)
    #
    # Let's use the standard method:
    # insert 0 at low_pos. (Now bit low_pos is 0).
    # Now we want to insert 0 at high_pos.
    # Since high_pos > low_pos, the bit at high_pos in the NEW value comes from bit high_pos-1 in the INTERMEDIATE value?
    # Actually, simply:
    # val = insert_zero_bit(k, low_pos)
    # val = insert_zero_bit(val, high_pos)
    # -> This puts 0 at low_pos (shifted if high_pos <= low_pos, but we sorted them).
    # If high_pos > low_pos:
    #    First ins at low_pos: bits >= low_pos shift up.
    #    Second ins at high_pos: bits >= high_pos shift up.
    #    The 0 at low_pos (which is < high_pos) is NOT shifted by the second op logic if we do it right?
    #    Actually if we insert at high_pos SECOND, it shifts everything >= high_pos.
    #    Since low_pos < high_pos, the 0 at low_pos stays at low_pos.
    #    The 0 at high_pos is newly created.
    #    So yes, `insert_zero_bit(insert_zero_bit(k, low_pos), high_pos)` works if high_pos > low_pos.
    #    Wait. If I insert at low_pos, all bits >= low_pos move to +1.
    #    So a bit that WAS at "high_pos - 1" is now at "high_pos".
    #    If we then insert at high_pos, we split right there.
    #    Let's trace.
    #    Want 0 at 0 and 3. k=0 (00).
    #    ins(0, 0) -> 0.
    #    ins(0, 3) -> 0. Correct.
    #    k=1 (01).
    #    ins(1, 0) -> 10 (2).
    #    ins(2, 3) -> 0010 (2). Correct (bits 0 and 3 are 0).
    #    k=3 (11).
    #    ins(3, 0) -> 110 (6).
    #    ins(6, 3) -> 0110 (6). Correct.
    #    It seems correct to insert lower then higher?
    #    Let's try insert higher then lower.
    #    k=3 (11). high=3, low=0.
    #    ins(3, 3) -> 011 (3).  (Bits at >=3 shift left. 3(11) has no bits >=3? 11 is bits 0,1. valid)
    #    ins(3, 0) -> 110 (6). Same result.
    #    Actually `insert_zero_bit` expects `pos` within the range of `val`'s significant bits + 1.
    #    If we generate N size state, we iterate k up to N/4-1. Max bits N-2.
    #    So we insert to make N-1 bits, then N bits.
    #    The order: construct from LSB to MSB or inverse?
    #    If we insert at `low_pos` first:
    #      We take (N-2) bits, split at low_pos. Make (N-1) bits. 0 at low_pos.
    #      Then split at `high_pos`.
    #      Since `high_pos` > `low_pos`, the split point `high_pos` is "above" the first 0.
    #      So the first 0 (at low_pos) is in the "low" part of the second split?
    #      Yes. `low = val & ~(-1<<high)` preserves everything below high.
    #      So `low_pos` is preserved.
    #      So this works.

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var size = state.size()
    var num_pairs = size >> 2

    # Parallelize
    alias chunk_size = 512
    var num_chunks = num_pairs // chunk_size

    @parameter
    fn worker(chunk_idx: Int):
        alias simd_width = simd_width_of[DType.float64]()
        # Ensure SIMD width is compatible (power of 2, fits in chunk_size)
        var start = chunk_idx * chunk_size
        var end = start + chunk_size

        # Vectorized loop
        for k in range(start, end, simd_width):
            var k_vec = SIMD[DType.int64, simd_width]()
            for i in range(simd_width):
                k_vec[i] = Int64(k + i)

            var idx = insert_zero_bit(k_vec, low_pos)
            idx = insert_zero_bit(idx, high_pos)

            var i_vec = idx | stride
            var j_vec = idx | dist_val

            # Swap real
            var val_i_re = ptr_re.gather(i_vec)
            var val_j_re = ptr_re.gather(j_vec)
            ptr_re.scatter(j_vec, val_i_re)
            ptr_re.scatter(i_vec, val_j_re)

            # Swap imag
            var val_i_im = ptr_im.gather(i_vec)
            var val_j_im = ptr_im.gather(j_vec)
            ptr_im.scatter(j_vec, val_i_im)
            ptr_im.scatter(i_vec, val_j_im)

    parallelize[worker](num_chunks)

    # Remainder
    for k in range(num_chunks * chunk_size, num_pairs):
        var idx = insert_zero_bit(k, low_pos)
        idx = insert_zero_bit(idx, high_pos)

        var i = idx | stride
        var j = idx | dist_val

        # Swap real
        var temp_re = ptr_re[i]
        ptr_re[i] = ptr_re[j]
        ptr_re[j] = temp_re

        # Swap imag
        var temp_im = ptr_im[i]
        ptr_im[i] = ptr_im[j]
        ptr_im[j] = temp_im
