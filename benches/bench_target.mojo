import benchmark

from butterfly.core.state import *
from butterfly.utils.visualization import print_state


fn test_target[n: Int, t: Int, swap: Bool = False, par: Int = 1]():
    #     f: fn(mut state: State, target: Int, gate: Gate) = transform_swap if swap else transform
    try:
        state = init_state(n)
        for _ in range(9):
            transform[par](state, t, H)
    except e:
        print("Caught an error:", e)


fn test_target_grid[r: Int, c: Int, t: Int, par: Int = 1]():
    try:
        state = init_state_grid(r, c)
        for _ in range(9):
            transform_grid[par](state, t, H)
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
    alias n: Int = 20

    alias iter = 2

    #     var report_target_low = benchmark.run[test_target[n, 0]](iter)
    #     report_target_low.print_full("low target ms")
    #
    #     var report_target_low_grid = benchmark.run[test_target_grid[5, n-5, 0]](iter)
    #     report_target_low_grid.print_full("low target grid ms")

    var report_target_high = benchmark.run[test_target[n, n - 1]](iter)
    report_target_high.print_full("high target ms")

    #     var report_target_high_par = benchmark.run[test_target[n, n-1, False, 2]](iter)
    #     report_target_high_par.print_full("high target par ms")
    #
    #     var report_target_high_grid_0 = benchmark.run[test_target_grid[0, n, n-1]](iter)
    #     report_target_high_grid_0.print_full("high target grid_0 ms")

    #     var report_target_high_grid_1 = benchmark.run[test_target_grid[1, n-1, n-1]](iter)
    #     report_target_high_grid_1.print_full("high target grid_1 ms")
    #
    #     var report_target_high_grid_2 = benchmark.run[test_target_grid[2, n-2, n-1]](iter)
    #     report_target_high_grid_2.print_full("high target grid_2 ms")
    #
    var report_target_high_grid_3 = benchmark.run[
        test_target_grid[5, n - 5, n - 1]
    ](iter)
    report_target_high_grid_3.print_full("high target grid_5 ms")

    var report_target_high_grid_0 = benchmark.run[
        test_target_grid[5, n - 5, n - 1, 8]
    ](iter)
    report_target_high_grid_0.print_full("high target par grid_5 ms")


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
