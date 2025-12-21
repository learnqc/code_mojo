import benchmark

from butterfly.core.state import *
from butterfly.algos.value_encoding import (
    encode_value,
    encode_value_simd,
    encode_value_simd_interval,
    encode_value_swap,
    encode_value_mix,
)
from butterfly.algos.value_encoding_fast import encode_value_super_fast
from butterfly.core.circuit import QuantumCircuit
from butterfly.algos.value_encoding import encode_value_circuit
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.utils.visualization import print_state


# fn test_encode_value[n: Int, v: FloatType]():
#     _state = encode_value(n, v)
#     #         print_state(_state)


fn test_encode_value_simd[n: Int, v: FloatType]():
    _state = encode_value_simd[n](v)
    #         print_state(_state)


# fn test_encode_value_simd_interval[n: Int, v: FloatType]():
#     _state = encode_value_simd_interval[n](v)
#     #         print_state(_state)


fn test_encode_value_super_fast[n: Int, v: FloatType]():
    _state = encode_value_super_fast[n](v)


fn test_encode_value_v2[n: Int, v: FloatType]():
    var circuit = QuantumCircuit(n)
    encode_value_circuit(circuit, n, v)
    circuit.execute_simd_v2()


fn test_encode_value_v3[n: Int, v: FloatType]():
    var circuit = QuantumCircuit(n)
    encode_value_circuit(circuit, n, v)
    execute_fused_v3[n](circuit.state, circuit.transformations, block_log=20)


# fn test_encode_value_swap[n: Int, v: FloatType]():
#     _ = encode_value_swap(n, v)
#     #         print_state(state)


# fn test_encode_value_mix[n: Int, v: FloatType]():
#     _state = encode_value_mix(n, v)
#     #         print_state(_state)


def main():
    alias n: Int = 25
    alias v = 4.7

    alias iter = 5

    # var report_encode_value = benchmark.run[test_encode_value[n, v]](2, iter)
    # report_encode_value.print_full("encode_value")

    var report_encode_value_simd = benchmark.run[test_encode_value_simd[n, v]](
        2, iter
    )
    report_encode_value_simd.print_full("encode_value simd")

    # var report_encode_value_simd_interval = benchmark.run[
    #     test_encode_value_simd_interval[n, v]
    # ](2, iter)
    # report_encode_value_simd_interval.print_full("encode_value simd interval")

    var report_encode_value_super_fast = benchmark.run[
        test_encode_value_super_fast[n, v]
    ](2, iter)
    report_encode_value_super_fast.print_full("encode_value super fast")

    var report_encode_value_v2 = benchmark.run[test_encode_value_v2[n, v]](
        2, iter
    )
    report_encode_value_v2.print_full("encode_value v2")

    var report_encode_value_v3 = benchmark.run[test_encode_value_v3[n, v]](
        2, iter
    )
    report_encode_value_v3.print_full("encode_value v3")

    # var report_encode_value_swap = benchmark.run[test_encode_value_swap[n, v]](
    #     2, iter
    # )
    # report_encode_value_swap.print_full("encode_value swap")

    # var report_encode_value_mix = benchmark.run[test_encode_value_mix[n, v]](
    #     2, iter
    # )
    # report_encode_value_mix.print_full("encode_value mix")
