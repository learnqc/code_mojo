from testing import assert_equal

from butterfly.algos.adjacent_pairs import *

def main():
    alias n = 3
    alias N = 1 << n
    l = [j for j in range(N)]
    #     l = InlineArray[Int, N](fill=0)
    #     for i in range(len(l)):
    #         l[i] = i

    for target in range(n):
        swap_to_adjacent_pairs(l, target)
        stride = 1 << target
        for j in range(len(l)//2):
            assert_equal( l[2*j] + stride, l[2*j + 1])

    from butterfly.utils.state import *
    target = 2
    s = init_state(n)
    transform(s, 0, X)
    swap_to_adjacent_pairs(s, target)
    assert_equal(s[1 << target], `1`)
