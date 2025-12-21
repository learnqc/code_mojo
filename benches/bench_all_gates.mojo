import benchmark
from butterfly.core.state import *
from butterfly.core.gates import *
from butterfly.core.types import *

alias unit = benchmark.Unit.ms


fn test_X[n: Int, t: Int]():
    state = init_state(n)
    for _ in range(10):
        transform(state, t, X)


fn test_Y[n: Int, t: Int]():
    state = init_state(n)
    for _ in range(10):
        transform(state, t, Y)


fn test_Z[n: Int, t: Int]():
    state = init_state(n)
    for _ in range(10):
        transform(state, t, Z)


fn test_H[n: Int, t: Int]():
    state = init_state(n)
    for _ in range(10):
        transform(state, t, H)


fn test_P[n: Int, t: Int]():
    state = init_state(n)
    for _ in range(10):
        transform(state, t, P(pi / 3))


fn test_RX[n: Int, t: Int]():
    state = init_state(n)
    for _ in range(10):
        transform(state, t, RX(pi / 3))


fn test_RY[n: Int, t: Int]():
    state = init_state(n)
    for _ in range(10):
        transform(state, t, RY(pi / 3))


fn test_RZ[n: Int, t: Int]():
    state = init_state(n)
    for _ in range(10):
        transform(state, t, RZ(pi / 3))


def main():
    alias n: Int = 20
    alias iter = 100

    print("Benchmarking Gates (n =", n, ")")
    print("--------------------------------------------------")

    # Targets: Low, Mid, High
    alias t_low = 3
    alias t_mid = n // 2
    alias t_high = n - 1

    # Gates
    # Fixed
    var report_X_low = benchmark.run[test_X[n, t_low]](2, iter)
    var report_X_mid = benchmark.run[test_X[n, t_mid]](2, iter)
    var report_X_high = benchmark.run[test_X[n, t_high]](2, iter)

    var report_Y_low = benchmark.run[test_Y[n, t_low]](2, iter)
    var report_Y_mid = benchmark.run[test_Y[n, t_mid]](2, iter)
    var report_Y_high = benchmark.run[test_Y[n, t_high]](2, iter)

    var report_Z_low = benchmark.run[test_Z[n, t_low]](2, iter)
    var report_Z_mid = benchmark.run[test_Z[n, t_mid]](2, iter)
    var report_Z_high = benchmark.run[test_Z[n, t_high]](2, iter)

    var report_H_low = benchmark.run[test_H[n, t_low]](2, iter)
    var report_H_mid = benchmark.run[test_H[n, t_mid]](2, iter)
    var report_H_high = benchmark.run[test_H[n, t_high]](2, iter)

    # Parameterized (using pi/4)

    var report_P_low = benchmark.run[test_P[n, t_low]](2, iter)
    var report_P_mid = benchmark.run[test_P[n, t_mid]](2, iter)
    var report_P_high = benchmark.run[test_P[n, t_high]](2, iter)

    var report_RX_low = benchmark.run[test_RX[n, t_low]](2, iter)
    var report_RX_mid = benchmark.run[test_RX[n, t_mid]](2, iter)
    var report_RX_high = benchmark.run[test_RX[n, t_high]](2, iter)

    var report_RY_low = benchmark.run[test_RY[n, t_low]](2, iter)
    var report_RY_mid = benchmark.run[test_RY[n, t_mid]](2, iter)
    var report_RY_high = benchmark.run[test_RY[n, t_high]](2, iter)

    var report_RZ_low = benchmark.run[test_RZ[n, t_low]](2, iter)
    var report_RZ_mid = benchmark.run[test_RZ[n, t_mid]](2, iter)
    var report_RZ_high = benchmark.run[test_RZ[n, t_high]](2, iter)

    # Printing Results
    print("Gate | Target | Mean Time (ms)")
    print("-----|--------|---------------")

    print("X    | Low    |", report_X_low.mean(unit))
    print("X    | Mid    |", report_X_mid.mean(unit))
    print("X    | High   |", report_X_high.mean(unit))
    print("-----|--------|---------------")

    print("Y    | Low    |", report_Y_low.mean(unit))
    print("Y    | Mid    |", report_Y_mid.mean(unit))
    print("Y    | High   |", report_Y_high.mean(unit))
    print("-----|--------|---------------")

    print("Z    | Low    |", report_Z_low.mean(unit))
    print("Z    | Mid    |", report_Z_mid.mean(unit))
    print("Z    | High   |", report_Z_high.mean(unit))
    print("-----|--------|---------------")

    print("H    | Low    |", report_H_low.mean(unit))
    print("H    | Mid    |", report_H_mid.mean(unit))
    print("H    | High   |", report_H_high.mean(unit))
    print("-----|--------|---------------")

    print("P    | Low    |", report_P_low.mean(unit))
    print("P    | Mid    |", report_P_mid.mean(unit))
    print("P    | High   |", report_P_high.mean(unit))
    print("-----|--------|---------------")

    print("RX   | Low    |", report_RX_low.mean(unit))
    print("RX   | Mid    |", report_RX_mid.mean(unit))
    print("RX   | High   |", report_RX_high.mean(unit))
    print("-----|--------|---------------")

    print("RY   | Low    |", report_RY_low.mean(unit))
    print("RY   | Mid    |", report_RY_mid.mean(unit))
    print("RY   | High   |", report_RY_high.mean(unit))
    print("-----|--------|---------------")

    print("RZ   | Low    |", report_RZ_low.mean(unit))
    print("RZ   | Mid    |", report_RZ_mid.mean(unit))
    print("RZ   | High   |", report_RZ_high.mean(unit))


# /Users/lg/DevMojo/code_mojo/.pixi/envs/default/bin/mojo run /Users/lg/DevMojo/code_mojo/benches/bench_all_gates.mojo
# (base) lg@Ls-MacBook-Pro-2 code_mojo % /Users/lg/DevMojo/code_mojo/.pixi/envs/default/bin/mojo run /Us
# ers/lg/DevMojo/code_mojo/benches/bench_all_gates.mojo

# Benchmarking Gates (n= 20 )
# --------------------------------------------------
# Gate | Target | Mean Time (ms)
# -----|--------|---------------
# X    | Low    | 7.452307482993197
# X    | Mid    | 10.892720197044335
# X    | High   | 10.974951243781094
# -----|--------|---------------
# Y    | Low    | 7.441479423868312
# Y    | Mid    | 10.973763184079601
# Y    | High   | 10.841612935323383
# -----|--------|---------------
# Z    | Low    | 7.478663237311385
# Z    | Mid    | 10.993018905472637
# Z    | High   | 10.916092537313434
# -----|--------|---------------
# H    | Low    | 7.49691769547325
# H    | Mid    | 11.296699497487438
# H    | High   | 11.498757360406092
# -----|--------|---------------
# P    | Low    | 9.291239669421488
# P    | Mid    | 11.629572020725389
# P    | High   | 11.69821139896373
# -----|--------|---------------
# RX   | Low    | 9.019727272727271
# RX   | Mid    | 11.707869743589745
# RX   | High   | 11.642029743589744
# -----|--------|---------------
# RY   | Low    | 9.223700826446281
# RY   | Mid    | 11.676822564102563
# RY   | High   | 11.6718
# -----|--------|---------------
# RZ   | Low    | 9.028344628099173
# RZ   | Mid    | 11.923058031088082
# RZ   | High   | 11.684448704663213
