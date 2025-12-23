from butterfly.core.circuit import QuantumCircuit
from butterfly.core.state import dagger
from butterfly.core.types import Amplitude, pi
from testing import assert_almost_equal
from math import sqrt, cos, sin


fn main() raises:
    print("--- Testing Unitary Dagger ---")

    # 1. Complex unitary (Rotation gate Rz(theta))
    # Rz(theta) = [[exp(-i*theta/2), 0], [0, exp(i*theta/2)]]
    # Rz(theta)^\dagger = Rz(-theta)
    var theta = pi / 4
    var u = List[Amplitude]()
    u.append(Amplitude(cos(-theta / 2), sin(-theta / 2)))
    u.append(Amplitude(0.0, 0.0))
    u.append(Amplitude(0.0, 0.0))
    u.append(Amplitude(cos(theta / 2), sin(theta / 2)))

    var u_dag = dagger(u, 1)

    # Should be [[exp(i*theta/2), 0], [0, exp(-i*theta/2)]]
    assert_almost_equal(u_dag[0].re, cos(theta / 2))
    assert_almost_equal(u_dag[0].im, sin(theta / 2))
    assert_almost_equal(u_dag[3].re, cos(-theta / 2))
    assert_almost_equal(u_dag[3].im, sin(-theta / 2))
    print("Test 1 (Rz complex dagger) passed.")

    # 2. Non-diagonal real unitary (Hadamard)
    # H = 1/sqrt(2) [[1, 1], [1, -1]]
    # H^\dagger = H
    var inv_sqrt2 = 1.0 / sqrt(2.0)
    var h = List[Amplitude]()
    h.append(Amplitude(inv_sqrt2))
    h.append(Amplitude(inv_sqrt2))
    h.append(Amplitude(inv_sqrt2))
    h.append(Amplitude(-inv_sqrt2))

    var h_dag = dagger(h, 1)
    for i in range(4):
        assert_almost_equal(h_dag[i].re, h[i].re)
    print("Test 2 (Hadamard real dagger) passed.")

    # 3. 2-qubit unitary (CNOT)
    # CNOT is real and symmetric, so CNOT^\dagger = CNOT
    var cnot = List[Amplitude]()
    for i in range(16):
        cnot.append(Amplitude(0.0))
    cnot[0] = Amplitude(1.0)  # |00> -> |00>
    cnot[5] = Amplitude(1.0)  # |01> -> |01>
    cnot[11] = Amplitude(1.0)  # |10> -> |11>
    cnot[14] = Amplitude(1.0)  # |11> -> |10>

    var cnot_dag = dagger(cnot, 2)
    for i in range(16):
        assert_almost_equal(cnot_dag[i].re, cnot[i].re)
    print("Test 3 (CNOT dagger) passed.")

    # 4. dagger(dagger(U)) == U
    var u_dag_dag = dagger(u_dag, 1)
    for i in range(4):
        assert_almost_equal(u_dag_dag[i].re, u[i].re)
        assert_almost_equal(u_dag_dag[i].im, u[i].im)
    print("Test 4 (Double dagger) passed.")

    print("Success! Dagger utility works correctly.")
