import benchmark

from butterfly.core.state import *
from butterfly.algos.value_encoding import encode_value

fn test_encode_value[n: UInt, v: FloatType]():
    try:
        _ = encode_value(n, v)
    except e:
        print("Caught an error:", e)

def main():
    alias n: UInt = 10
    alias v = 4.7

    alias iter = Int(1_000)

    var report_encode_value = benchmark.run[test_encode_value[n, v]](iter)

    report_encode_value.print_full("ms")
