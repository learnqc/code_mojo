from testing import assert_equal
from stdlib.bit.bit import pop_count


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
    alias dist_val = 8
    stride = 1 << target

    if stride == dist_val:
        return

    # Swap pairs (i, j) where i has bit 3 (Val 8) set and bit 'target' clear,
    # and j has bit 3 clear and bit 'target' set.
    # j = i ^ 8 ^ stride

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    for i in range(state.size()):
        if (i & dist_val) != 0 and (i & stride) == 0:
            var j = i ^ dist_val ^ stride

            # Swap real
            var temp_re = ptr_re[i]
            ptr_re[i] = ptr_re[j]
            ptr_re[j] = temp_re

            # Swap imag
            var temp_im = ptr_im[i]
            ptr_im[i] = ptr_im[j]
            ptr_im[j] = temp_im
