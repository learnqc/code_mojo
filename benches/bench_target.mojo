import benchmark

from butterfly.core.state import *
from butterfly.utils.state import print_state

fn test_target[n: Int, t: Int, swap: Bool = False]():
    f: fn(mut state: State, target: UInt, gate: Gate) = transform_swap if swap else transform
    try:
        state = init_state(n)
        for _ in range(9):
            f(state, t, H)
    except e:
        print("Caught an error:", e)

fn test_target_grid[r: Int, c: Int, t: Int]():
    try:
        state = init_state_grid(r, c)
        for _ in range(9):
            transform_grid(state, t, H)
    except e:
        print("Caught an error:", e)


fn test_target_a[n: Int, t: Int]():
    try:
        state = init_state_a[n]()
        for _ in range(9):
            transform_a(state, t, H)
    except e:
        print("Caught an error:", e)


def main():
    alias n: UInt = 31

    alias iter = Int(10)

#     var report_target_low = benchmark.run[test_target[n, 0]](iter)
#     report_target_low.print_full("low target ms")
#
#     var report_target_low_grid = benchmark.run[test_target_grid[5, n-5, 0]](iter)
#     report_target_low_grid.print_full("low target grid ms")

    var report_target_high = benchmark.run[test_target[n, n-1]](iter)
    report_target_high.print_full("high target ms")

    var report_target_high_grid_0 = benchmark.run[test_target_grid[0, n, n-1]](iter)
    report_target_high_grid_0.print_full("high target grid_0 ms")

#     var report_target_high_grid_1 = benchmark.run[test_target_grid[1, n-1, n-1]](iter)
#     report_target_high_grid_1.print_full("high target grid_1 ms")
#
#     var report_target_high_grid_2 = benchmark.run[test_target_grid[2, n-2, n-1]](iter)
#     report_target_high_grid_2.print_full("high target grid_2 ms")
#
#     var report_target_high_grid_3 = benchmark.run[test_target_grid[3, n-3, n-1]](iter)
#     report_target_high_grid_3.print_full("high target grid_3 ms")

#     var report_target_low_a = benchmark.run[test_target_a[n, 0]](iter)
#     report_target_low_a.print_full("low target array ms")
#
#     var report_target_low_swap = benchmark.run[test_target[n, 0, True]](iter)
#     report_target_low_swap.print_full("low target swap ms")
#
#
#     var report_target_high_a = benchmark.run[test_target_a[n, n-1]](iter)
#     report_target_high_a.print_full("high target array ms")
#
#     var report_target_high_swap = benchmark.run[test_target[n, n-1, True]](iter)
#     report_target_high_swap.print_full("high target swap ms")

