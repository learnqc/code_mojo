from testing import assert_equal
from stdlib.bit.bit import pop_count

fn swap_to_adjacent_pairs[T: ImplicitlyCopyable & Movable](mut lst: List[T], target: UInt) raises:
    assert_equal(pop_count(len(lst)), 1)
    stride = 1 << target

    r  = 0
    for j in range(0, len(lst), 4):
        idx = j - r

        lst.swap_elements(idx + 1, idx + stride)

        r += 2
        if r == stride:
            r = 0
