from testing import assert_true, assert_almost_equal
from butterfly.core.state import (
    QuantumState,
    generate_state,
    init_state,
    mc_transform,
    mc_transform_interval,
    mc_transform_simd,
)
from butterfly.core.gates import X, H
from butterfly.core.types import Amplitude, pi
from algorithm import vectorize


fn test_ccx() raises:
    # Test Toffoli (CCX) gate on 3 qubits
    # Controls: 0, 1; Target: 2
    # Expectation: |110> -> |111> ( indices 3 (011) -> 7 (111) in little endian? Check conventions)
    # Our convention: qubit 0 is LSB.
    # |110> -> bit 0=0, bit 1=1, bit 2=1 -> index 6 (110 binary is 6)
    # Target 2 flipped: |111> -> index 7
    # Wait, simple check: apply to all basis states.

    var n = 3

    # run it on a superposition
    state_naive = generate_state(n, seed=123)
    state_opt = state_naive

    var controls = List[Int]()
    controls.append(0)
    controls.append(1)

    mc_transform(state_naive, controls, 2, X)
    mc_transform_interval(state_opt, controls, 2, X)

    for i in range(state_naive.size()):
        assert_almost_equal(state_naive.re[i], state_opt.re[i])
        assert_almost_equal(state_naive.im[i], state_opt.im[i])

    print("mc_transform vs mc_transform_interval consistency check passed")


fn test_toffoli_logic() raises:
    # Verify specific logic: CCX |110> -> |111>
    # Bits: q0=1, q1=1, q2=0. Index = 1 + 2 + 0 = 3.
    # Target q2. New state q0=1, q1=1, q2=1. Index = 7.

    var n = 3
    # Flip to |110> (q0, q1 set)
    # q0 is 1, q1 is 1.
    # q0 is 1<<0 = 1. q1 is 1<<1 = 2. Index 3.
    state = init_state(n)
    state[0] = Amplitude(0, 0)
    state[3] = Amplitude(1, 0)  # |110> (q0=1, q1=1, q2=0) -> state[3]

    var controls = List[Int]()
    controls.append(0)
    controls.append(1)

    mc_transform_interval(state, controls, 2, X)

    # Should be |111> -> index 3 + 4 = 7.
    assert_almost_equal(state[3].re, 0.0)
    assert_almost_equal(state[7].re, 1.0)
    print("CCX logic check passed")


fn test_mc_transform_simd() raises:
    # Verify SIMD implementation against interval/naive
    var n = 6

    # Random initial state
    state_simd = generate_state(n, seed=42)
    state_ref = state_simd  # Implicit copy

    var controls = List[Int]()
    controls.append(0)
    controls.append(2)
    controls.append(4)
    # Target 3
    # Gate X

    mc_transform_simd[1 << 6](state_simd, controls, 3, X)
    mc_transform_interval(state_ref, controls, 3, X)

    for i in range(state_simd.size()):
        assert_almost_equal(state_simd.re[i], state_ref.re[i])
        assert_almost_equal(state_simd.im[i], state_ref.im[i])

    print("mc_transform_simd consistency check passed")


fn main() raises:
    test_ccx()
    test_toffoli_logic()
    test_mc_transform_simd()
