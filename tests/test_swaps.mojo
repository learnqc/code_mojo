from butterfly.algos.adjacent_pairs import (
    swap_to_adjacent_pairs,
    swap_to_distance_8,
    swap_to_new_stride,
)
from collections import List
from testing import assert_true, assert_equal


fn test_swap_to_adjacent() raises:
    print("\n=== Testing swap_to_adjacent_pairs ===")
    alias n = 3
    alias N = 1 << n

    for t in range(n):
        print("Target:", t, "Stride:", 1 << t)
        var l = List[Int]()
        for i in range(N):
            l.append(i)

        swap_to_adjacent_pairs(l, UInt(t))

        # Verify pairs
        var stride = 1 << t
        for i in range(0, N, 2):
            var diff = abs(l[i + 1] - l[i])
            assert_equal(diff, stride)
    print("All adjacent swap tests passed.")


fn test_swap_to_dist8() raises:
    print("\n=== Testing swap_to_distance_8 ===")
    alias n = 5  # 32 elements
    alias N = 1 << n

    for t in range(n):
        if (1 << t) == 8:
            continue

        print("Target:", t, "Stride:", 1 << t)
        var l = List[Int]()
        for i in range(N):
            l.append(i)

        swap_to_distance_8(l, UInt(t))

        # Verify pairs at distance 8
        var stride = 1 << t
        for i in range(N):
            if (i & 8) == 0:
                var val_a = l[i]
                var val_b = l[i + 8]
                var diff = abs(val_b - val_a)
                assert_equal(diff, stride)
    print("All distance-8 swap tests passed.")


fn test_swap_to_dist4() raises:
    alias n = 4  # 32 elements
    alias N = 1 << n

    var l = List[Int]()
    for i in range(N):
        l.append(i)

    for i in range(N):
        print(l[i], end=", " if i < N - 1 else "\n")

    swap_to_new_stride(l, 1, 1)

    for i in range(N):
        print(l[i], end=", ")


fn main() raises:
    test_swap_to_adjacent()
    test_swap_to_dist8()
    test_swap_to_dist4()
