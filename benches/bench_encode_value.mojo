import benchmark

from butterfly.core.state import *
from butterfly.algos.value_encoding import (
    encode_value,
    encode_value_swap,
    encode_value_mix,
)
from butterfly.utils.visualization import print_state


fn test_encode_value[n: Int, v: FloatType]():
    try:
        _state = encode_value(n, v)
    #         print_state(_state)
    except e:
        print("Caught an error:", e)


fn test_encode_value_swap[n: Int, v: FloatType]():
    try:
        _ = encode_value_swap(n, v)
    #         print_state(state)
    except e:
        print("Caught an error:", e)


fn test_encode_value_mix[n: Int, v: FloatType]():
    try:
        _state = encode_value_mix(n, v)
    #         print_state(_state)
    except e:
        print("Caught an error:", e)


def main():
    alias n: Int = 25
    alias v = 4.7

    alias iter = Int(1)

    var report_encode_value = benchmark.run[test_encode_value[n, v]](iter)
    var report_encode_value_swap = benchmark.run[test_encode_value_swap[n, v]](
        iter
    )
    var report_encode_value_mix = benchmark.run[test_encode_value_mix[n, v]](
        iter
    )

    report_encode_value.print_full("encode_value ms")
    report_encode_value_swap.print_full("encode_value swap ms")
    report_encode_value_mix.print_full("encode_value mix ms")
