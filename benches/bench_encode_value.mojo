import benchmark

from butterfly.core.state import *
from butterfly.algos.value_encoding import (
    encode_value,
    encode_value_simd,
    encode_value_swap,
    encode_value_mix,
)
from butterfly.utils.visualization import print_state


fn test_encode_value[n: Int, v: FloatType]():
    _state = encode_value(n, v)
    #         print_state(_state)


fn test_encode_value_simd[n: Int, v: FloatType]():
    _state = encode_value_simd[n](v)
    #         print_state(_state)


fn test_encode_value_swap[n: Int, v: FloatType]():
    _ = encode_value_swap(n, v)
    #         print_state(state)


fn test_encode_value_mix[n: Int, v: FloatType]():
    _state = encode_value_mix(n, v)
    #         print_state(_state)


def main():
    alias n: Int = 10
    alias v = 4.7

    alias iter = 5

    var report_encode_value = benchmark.run[test_encode_value[n, v]](2, iter)
    report_encode_value.print_full("encode_value")

    var report_encode_value_simd = benchmark.run[test_encode_value_simd[n, v]](
        2, iter
    )
    report_encode_value_simd.print_full("encode_value simd")

    var report_encode_value_swap = benchmark.run[test_encode_value_swap[n, v]](
        2, iter
    )
    report_encode_value_swap.print_full("encode_value swap")

    var report_encode_value_mix = benchmark.run[test_encode_value_mix[n, v]](
        2, iter
    )
    report_encode_value_mix.print_full("encode_value mix")
