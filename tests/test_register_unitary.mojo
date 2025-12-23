from butterfly.core.circuit import QuantumCircuit, QuantumRegister
from butterfly.core.types import Amplitude
from testing import assert_almost_equal
from math import sqrt


fn main() raises:
    print("--- Testing Register-Based Unitary Transformations ---")

    # 1. Test append_u on a 2-qubit register
    var qc = QuantumCircuit(2)
    var reg = qc.add_register("data", 2)

    # Identity as a 4x4 matrix
    var u_id = List[Amplitude]()
    for i in range(16):
        if i % 5 == 0:
            u_id.append(Amplitude(1.0))
        else:
            u_id.append(Amplitude(0.0))

    # Apply to register
    qc.append_u(u_id^, reg)
    qc.execute()

    assert_almost_equal(qc.state[0].re, 1.0)
    print("Test 1 (append_u identity) passed.")

    # 2. Test c_append_u (Controlled Register Unitary)
    var qc2 = QuantumCircuit(3)
    var reg2 = qc2.add_register("targets", 2)  # Qubits 0, 1
    var control = 2

    # X gate on qubit 0 as a 4x4 unitary (X \otimes I)
    # Basis: |q1 q0> -> 00, 01, 10, 11
    # Flip q0: 00 <-> 01, 10 <-> 11
    var u_x0 = List[Amplitude]()
    # Row 0: 0 1 0 0
    u_x0.append(Amplitude(0.0))
    u_x0.append(Amplitude(1.0))
    u_x0.append(Amplitude(0.0))
    u_x0.append(Amplitude(0.0))
    # Row 1: 1 0 0 0
    u_x0.append(Amplitude(1.0))
    u_x0.append(Amplitude(0.0))
    u_x0.append(Amplitude(0.0))
    u_x0.append(Amplitude(0.0))
    # Row 2: 0 0 0 1
    u_x0.append(Amplitude(0.0))
    u_x0.append(Amplitude(0.0))
    u_x0.append(Amplitude(0.0))
    u_x0.append(Amplitude(1.0))
    # Row 3: 0 0 1 0
    u_x0.append(Amplitude(0.0))
    u_x0.append(Amplitude(0.0))
    u_x0.append(Amplitude(1.0))
    u_x0.append(Amplitude(0.0))

    # Set control (qubit 2 = index 4) FIRST so it's set when c_append_u executes
    qc2.x(control)

    qc2.c_append_u(u_x0^, control, reg2)

    qc2.execute()

    # Expect qubit 0 to be flipped (state |100> -> index 4. After flip q0 -> |101> -> index 5)
    assert_almost_equal(qc2.state[5].re, 1.0)
    print("Test 2 (c_append_u controlled X0) passed.")

    # 3. Test unitary() on single target (m=1)
    print("Testing unitary() on single target:")
    var qc3 = QuantumCircuit(2)
    var u_x = List[Amplitude]()
    u_x.append(Amplitude(0.0))
    u_x.append(Amplitude(1.0))
    u_x.append(Amplitude(1.0))
    u_x.append(Amplitude(0.0))

    qc3.unitary(u_x^, 0)  # Implicitly m=1
    qc3.execute()
    assert_almost_equal(qc3.state[1].re, 1.0)
    print("Test 3 (unitary m=1) passed.")

    print("Success!")
