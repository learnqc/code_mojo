from butterfly.core.state import (
    QuantumState,
    transform_u,
    c_transform_u,
    transform,
    c_transform,
)
from butterfly.core.gates import H, X
from butterfly.core.types import Amplitude
from testing import assert_almost_equal
from math import sqrt


fn main() raises:
    print("--- Testing Unitary Transformations ---")

    # 1. Test transform_u (Hadamard 2x2)
    var state1 = QuantumState(1)
    var u_h = List[Amplitude]()
    var s = 1.0 / sqrt(2.0)
    u_h.append(Amplitude(s, 0))
    u_h.append(Amplitude(s, 0))
    u_h.append(Amplitude(s, 0))
    u_h.append(Amplitude(-s, 0))

    print("Testing transform_u (Hadamard)...")
    transform_u(state1, u_h, 0, 1)

    var expected_h = 1.0 / sqrt(2.0)
    assert_almost_equal(state1[0].re, expected_h)
    assert_almost_equal(state1[1].re, expected_h)

    # 2. Test transform_u with offset (H on qubit 1 of 2 qubits)
    var state2 = QuantumState(2)
    print("Testing transform_u on qubit 1...")
    transform_u(state2, u_h, 1, 1)
    # Binary states: |00>, |10> (indices 0, 2) should have 1/sqrt(2)
    assert_almost_equal(state2[0].re, expected_h)
    assert_almost_equal(state2[2].re, expected_h)
    assert_almost_equal(state2[1].re, 0.0)

    # 3. Test c_transform_u
    var state3 = QuantumState(2)
    # CNOT using transform_u (4x4 matrix)
    # Actually let's test c_transform_u (H on qubit 0 controlled by qubit 1)
    print("Testing c_transform_u (C-H)...")
    transform(state3, 1, X)  # Set control bit
    c_transform_u(state3, u_h, 1, 0, 1)
    # State was |10> (idx 2), now should be (|10> + |11>)/sqrt(2) (idx 2, 3)
    assert_almost_equal(state3[2].re, expected_h)
    assert_almost_equal(state3[3].re, expected_h)

    print("Success! Unitary transformations match expected results.")
