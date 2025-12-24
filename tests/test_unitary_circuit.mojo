from butterfly.core.circuit import QuantumCircuit
from butterfly.core.circuit import run_circuit
from butterfly.core.types import Amplitude
from testing import assert_almost_equal
from math import sqrt


fn main() raises:
    print("--- Testing Unitary Transformations in Circuit ---")

    # 1. Hadamard as a unitary
    var u_h = List[Amplitude]()
    var inv_sqrt2 = 1.0 / sqrt(2.0)
    u_h.append(Amplitude(inv_sqrt2))
    u_h.append(Amplitude(inv_sqrt2))
    u_h.append(Amplitude(inv_sqrt2))
    u_h.append(Amplitude(-inv_sqrt2))

    var qc1 = QuantumCircuit(1)
    qc1.u(u_h^, 0)  # Apply H as unitary
    var state1 = run_circuit[1](qc1)

    assert_almost_equal(state1[0].re, inv_sqrt2)
    assert_almost_equal(state1[1].re, inv_sqrt2)
    print("Test 1 (add_unitary H) passed.")

    # 2. Controlled Unitary (should behave like CNOT if U=X)
    var u_x = List[Amplitude]()
    u_x.append(Amplitude(0.0))
    u_x.append(Amplitude(1.0))
    u_x.append(Amplitude(1.0))
    u_x.append(Amplitude(0.0))

    var qc2 = QuantumCircuit(2)
    qc2.x(1)  # Set control
    qc2.cu(u_x^, 1, 0)  # Apply X controlled by 1 (control=1, target=0)
    var state2 = run_circuit[2](qc2)

    # State should be |11> (idx 3)
    assert_almost_equal(state2[3].re, 1.0)
    print("Test 2 (add_controlled_unitary X=CNOT) passed.")

    # 3. Append circuit with unitaries
    var u_h_v2 = List[Amplitude]()
    u_h_v2.append(Amplitude(inv_sqrt2))
    u_h_v2.append(Amplitude(inv_sqrt2))
    u_h_v2.append(Amplitude(inv_sqrt2))
    u_h_v2.append(Amplitude(-inv_sqrt2))

    var qc_sub = QuantumCircuit(1)
    qc_sub.u(u_h_v2^, 0)

    var qc_main = QuantumCircuit(1)
    var reg = qc_main.add_register("reg", 1)
    qc_main.append_circuit(qc_sub, reg)
    var state3 = run_circuit[1](qc_main)

    assert_almost_equal(state3[0].re, inv_sqrt2)
    assert_almost_equal(state3[1].re, inv_sqrt2)
    print("Test 3 (append_circuit with unitaries) passed.")

    print("Success! Integrated unitary transformations work correctly.")
