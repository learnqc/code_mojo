import benchmark

from butterfly.core.state import *
from butterfly.utils.state import print_grid_state

alias unit = benchmark.Unit.ms

fn test_target[n: Int, t: Int, par: UInt = 0]():
    try:
        state = init_state(n)
        for _ in range(5):
            transform[par](state, t, H)
    except e:
        print("Caught an error:", e)

fn test_target_grid[n: Int, t: Int, r: Int, par: UInt = 0]():
    c = n - r
    try:
        state = init_state_grid(r, c)
        for _ in range(9):
            transform_grid[par](state, t, H)
    except e:
        print("Caught an error:", e)


def main():
    alias n: UInt = 30

    alias iter = Int(5)

    alias threads = 8
    alias target = n-1

    var report_target = benchmark.run[test_target[n, target]](iter)
    report_target.print("List. bits={}, target={}".format(n, target))
    t0 = report_target.mean(unit)

    report_target = benchmark.run[test_target[n, target, threads]](iter)
    report_target.print("Parallel List. bits={}, target={}, threads={}".format(n, target, threads))
    t1 = report_target.mean(unit)

    alias row_bits = 3

    report_target = benchmark.run[test_target_grid[n, target, row_bits]](iter)
    report_target.print("Grid. bits={}, target={}, row_bits={}".format(n, target, row_bits))
    t2 = report_target.mean(unit)

    report_target = benchmark.run[test_target_grid[n, target, row_bits, threads]](iter)
    report_target.print("Parallel Grid. bits={}, target={}, row_bits={}, threads={}".format(n, target, row_bits, threads))
    t3 = report_target.mean(unit)

    print("Parallel List over List speedup:", t0/t1)
    print("Parallel Grid over Grid speedup:", t2/t3)
    print("Grid over List speedup:", t0/t2)
    print("Parallel Grid over List speedup:", t0/t3)

