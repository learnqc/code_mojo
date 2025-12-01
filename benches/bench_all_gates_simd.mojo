import benchmark
from butterfly.core.state import *
from butterfly.core.gates import *
from butterfly.core.types import *
from butterfly.utils.visualization import print_state

alias unit = benchmark.Unit.ms


fn test_X_simd[n: Int, t: Int]():
    try:
        state = init_state(n)
        for _ in range(10):
            transform_simd[1 << n](state, t, X)
    except e:
        print("Caught an error:", e)


fn test_Y_simd[n: Int, t: Int]():
    try:
        state = init_state(n)
        for _ in range(10):
            transform_simd[1 << n](state, t, Y)
    except e:
        print("Caught an error:", e)


fn test_Z_simd[n: Int, t: Int]():
    try:
        state = init_state(n)
        for _ in range(10):
            transform_simd[1 << n](state, t, Z)
    except e:
        print("Caught an error:", e)


fn test_H_simd[n: Int, t: Int]():
    try:
        state = init_state(n)
        for _ in range(10):
            transform_simd[1 << n](state, t, H)
    except e:
        print("Caught an error:", e)


fn test_P_simd[n: Int, t: Int]():
    try:
        state = init_state(n)
        for _ in range(10):
            transform_simd[1 << n](state, t, P(pi/3))
    except e:
        print("Caught an error:", e)


fn test_RX_simd[n: Int, t: Int]():
    try:
        state = init_state(n)
        for _ in range(10):
            transform_simd[1 << n](state, t, RX(pi/3))
    except e:
        print("Caught an error:", e)


fn test_RY_simd[n: Int, t: Int]():
    try:
        state = init_state(n)
        for _ in range(10):
            transform_simd[1 << n](state, t, RY(pi/3))
    except e:
        print("Caught an error:", e)


fn test_RZ_simd[n: Int, t: Int]():
    try:
        state = init_state(n)
        for _ in range(10):
            transform_simd[1 << n](state, t, RZ(pi/3))
    except e:
        print("Caught an error:", e)


def main():
    alias n: Int = 20
    alias iter = 100

    print("Benchmarking Gates SIMD (n =", n, ")")
    print("--------------------------------------------------")

    # Targets: Low, Mid, High
    alias t_low = 3
    alias t_mid = n // 2
    alias t_high = n - 1

    # Gates
    # Fixed
    # Fixed
    var report_X_low = benchmark.run[test_X_simd[n, t_low]](2, iter)
    var report_X_mid = benchmark.run[test_X_simd[n, t_mid]](2, iter)
    var report_X_high = benchmark.run[test_X_simd[n, t_high]](2, iter)

    var report_Y_low = benchmark.run[test_Y_simd[n, t_low]](2, iter)
    var report_Y_mid = benchmark.run[test_Y_simd[n, t_mid]](2, iter)
    var report_Y_high = benchmark.run[test_Y_simd[n, t_high]](2, iter)

    var report_Z_low = benchmark.run[test_Z_simd[n, t_low]](2, iter)
    var report_Z_mid = benchmark.run[test_Z_simd[n, t_mid]](2, iter)
    var report_Z_high = benchmark.run[test_Z_simd[n, t_high]](2, iter)

    var report_H_low = benchmark.run[test_H_simd[n, t_low]](2, iter)
    var report_H_mid = benchmark.run[test_H_simd[n, t_mid]](2, iter)
    var report_H_high = benchmark.run[test_H_simd[n, t_high]](2, iter)

    # Parameterized (using pi/4)

    var report_P_low = benchmark.run[test_P_simd[n, t_low]](2, iter)
    var report_P_mid = benchmark.run[test_P_simd[n, t_mid]](2, iter)
    var report_P_high = benchmark.run[test_P_simd[n, t_high]](2, iter)

    var report_RX_low = benchmark.run[test_RX_simd[n, t_low]](2, iter)
    var report_RX_mid = benchmark.run[test_RX_simd[n, t_mid]](2, iter)
    var report_RX_high = benchmark.run[test_RX_simd[n, t_high]](2, iter)

    var report_RY_low = benchmark.run[test_RY_simd[n, t_low]](2, iter)
    var report_RY_mid = benchmark.run[test_RY_simd[n, t_mid]](2, iter)
    var report_RY_high = benchmark.run[test_RY_simd[n, t_high]](2, iter)

    var report_RZ_low = benchmark.run[test_RZ_simd[n, t_low]](2, iter)
    var report_RZ_mid = benchmark.run[test_RZ_simd[n, t_mid]](2, iter)
    var report_RZ_high = benchmark.run[test_RZ_simd[n, t_high]](2, iter)

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


# low = 0
# Benchmarking Gates SIMD (n= 20 )
# --------------------------------------------------
# Gate | Target | Mean Time (ms)
# -----|--------|---------------
# X    | Low    | 8.142483146067416
# X    | Mid    | 4.426159999999999
# X    | High   | 3.494529411764706
# -----|--------|---------------
# Y    | Low    | 7.617977528089888
# Y    | Mid    | 3.8901320754716977
# Y    | High   | 1.6061842105263158
# -----|--------|---------------
# Z    | Low    | 7.336797752808989
# Z    | Mid    | 1.6441538461538463
# Z    | High   | 1.6304545454545454
# -----|--------|---------------
# H    | Low    | 7.506533707865168
# H    | Mid    | 1.6167664670658684
# H    | High   | 1.5479537572254334
# -----|--------|---------------
# P    | Low    | 9.0885
# P    | Mid    | 1.8228092485549132
# P    | High   | 1.5675449101796406
# -----|--------|---------------
# RX   | Low    | 9.04066091954023
# RX   | Mid    | 1.6232716763005781
# RX   | High   | 1.5923583815028903
# -----|--------|---------------
# RY   | Low    | 9.019701149425288
# RY   | Mid    | 1.544878612716763
# RY   | High   | 1.9675325443786982
# -----|--------|---------------
# RZ   | Low    | 9.185660919540231
# RZ   | Mid    | 1.6832732919254658
# RZ   | High   | 1.5976766467065868

# low = 3
# Benchmarking Gates SIMD (n= 20 )
# --------------------------------------------------
# Gate | Target | Mean Time (ms)
# -----|--------|---------------
# X    | Low    | 2.0396257668711657
# X    | Mid    | 1.7418208092485548
# X    | High   | 1.6360115606936414
# -----|--------|---------------
# Y    | Low    | 1.5755151515151513
# Y    | Mid    | 1.6386228571428572
# Y    | High   | 1.5793815028901734
# -----|--------|---------------
# Z    | Low    | 1.6122666666666667
# Z    | Mid    | 1.6089132947976879
# Z    | High   | 1.6340584795321635
# -----|--------|---------------
# H    | Low    | 1.646421965317919
# H    | Mid    | 1.65039263803681
# H    | High   | 1.7678402366863903
# -----|--------|---------------
# P    | Low    | 1.6689693251533742
# P    | Mid    | 2.6222890173410405
# P    | High   | 2.3747028571428572
# -----|--------|---------------
# RX   | Low    | 1.6126363636363634
# RX   | Mid    | 1.5370342857142856
# RX   | High   | 1.6277485714285713
# -----|--------|---------------
# RY   | Low    | 1.7380231213872832
# RY   | Mid    | 1.7873542857142857
# RY   | High   | 1.7937604790419162
# -----|--------|---------------
# RZ   | Low    | 1.575757396449704
# RZ   | Mid    | 1.595
# RZ   | High   | 1.5620742857142857
