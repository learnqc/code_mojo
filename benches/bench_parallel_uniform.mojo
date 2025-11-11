import benchmark

from butterfly.core.state import *
from butterfly.utils.state import print_grid_state

alias unit = benchmark.Unit.ms

fn test_uniform[n: Int, par: UInt = 0]():
    try:
        state = init_state(n)
        for t in range(n):
            transform[par](state, t, H)
    except e:
        print("Caught an error:", e)

fn test_uniform_grid[n: Int, r: Int, par: UInt = 0]():
    try:
        state = init_state_grid(r, n - r)
        for t in range(n):
            transform_grid[par](state, t, H)
    except e:
        print("Caught an error:", e)

def main():
    alias n: UInt = 30

    alias iter = Int(5)

    alias threads = 8

    alias row_bits = 3

    var report_target = benchmark.run[test_uniform[n]](iter)
    report_target.print("Parallel List. bits={}".format(n))
    u0 = report_target.mean(unit)

    report_target = benchmark.run[test_uniform[n, threads]](iter)
    report_target.print("Parallel List. bits={}, threads={}".format(n, threads))
    u1 = report_target.mean(unit)

    report_target = benchmark.run[test_uniform_grid[n, row_bits, threads]](iter)
    report_target.print("Uniform Parallel Grid. bits={}, row_bits={}, threads={}".format(n, row_bits, threads))
    u2 = report_target.mean(unit)

    print("Uniform Parallel List over List speedup:", u0/u1)
    print("Uniform Parallel Grid over List speedup:", u0/u2)

#     state = init_state_grid(row_bits, n - row_bits)
#     for t in range(n):
#         transform_grid[8](state, t, H)
#
#     print_grid_state(state)