from butterfly.core.circuit import QuantumCircuit, Register
from butterfly.core.gates import H, X
from testing import assert_almost_equal
from math import sqrt


fn main() raises:
    print("--- Testing Circuit Composition ---")

    # 1. Sub-circuit (Bell pair)
    var bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(1, 0)

    # 2. Main circuit
    var main = QuantumCircuit(4)
    var reg1 = main.add_register("reg1", 2)
    var reg2 = main.add_register("reg2", 2)

    print("Appending Bell circuit to reg1...")
    main.append_circuit(bell, reg1)

    print("Appending Bell circuit to reg2...")
    main.append_circuit(bell, reg2)

    print("Executing...")
    main.execute()

    # 3. Verify
    # State should be (|00> + |11>)/sqrt(2) ⊗ (|00> + |11>)/sqrt(2)
    # Map to 4 qubits: (|0000> + |0011> + |1100> + |1111>) / 2
    # Indices: 0, 3, 12, 15

    expected = 0.5
    assert_almost_equal(main.state[0].re, expected)
    assert_almost_equal(main.state[3].re, expected)
    assert_almost_equal(main.state[12].re, expected)
    assert_almost_equal(main.state[15].re, expected)

    print("Success! State matches expected superposition.")
