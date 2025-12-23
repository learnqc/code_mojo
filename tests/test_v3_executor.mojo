from butterfly.core.circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.core.gates import H, X, P
from math import pi


fn main() raises:
    print("Testing v3 Fused Executor...")
    alias n = 12
    var v = 0.5

    # 1. Standard approach
    var c_ref = QuantumCircuit(n)
    c_ref.h(0)
    c_ref.h(1)
    c_ref.x(11)
    c_ref.p(2, pi / 3)
    c_ref.execute()
    var s_ref = c_ref.get_state()

    # 2. v3 approach
    var c_v3 = QuantumCircuit(n)
    c_v3.h(0)
    c_v3.h(1)
    c_v3.x(11)
    c_v3.p(2, pi / 3)

    execute_fused_v3[1 << n](c_v3.state, c_v3.transformations)
    var s_v3 = c_v3.get_state()

    # Verify
    var diff = 0.0
    for i in range(1 << n):
        var d_re = s_ref.re[i] - s_v3.re[i]
        var d_im = s_ref.im[i] - s_v3.im[i]
        diff += d_re * d_re + d_im * d_im

    if diff < 1e-10:
        print("Success: v3 matches reference!")
    else:
        print("Failure: v3 mismatch! diff =", diff)
