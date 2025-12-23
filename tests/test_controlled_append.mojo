from butterfly.core.circuit import QuantumCircuit, Register
from butterfly.core.gates import X
from testing import assert_almost_equal
from math import sqrt


fn main() raises:
    print("--- Testing Controlled Circuit Appending ---")

    # 1. Simple sub-circuit (X on qubit 0)
    var sub = QuantumCircuit(1)
    sub.x(0)

    # 2. Test c_append_circuit (should result in CNOT)
    print("Testing c_append_circuit (CNOT)...")
    var main1 = QuantumCircuit(2)
    var reg1 = main1.add_register("reg1", 1)  # target at qubit 0
    # Initialize control qubit
    main1.x(1)

    # Append sub (X at 0) to reg1, controlled by qubit 1
    main1.c_append_circuit(sub, reg1, 1)

    main1.execute()
    # State should be |11> (idx 3)
    assert_almost_equal(main1.state[3].re, 1.0)
    assert_almost_equal(main1.state[2].re, 0.0)

    # 3. Test mc_append_circuit (should result in Toffoli)
    print("Testing mc_append_circuit (Toffoli)...")
    var main2 = QuantumCircuit(3)
    var reg2 = main2.add_register("reg2", 1)  # target at qubit 0
    # Initialize control qubits
    main2.x(1)
    main2.x(2)

    var controls = List[Int]()
    controls.append(1)
    controls.append(2)
    main2.mc_append_circuit(sub, reg2, controls^)

    main2.execute()
    # State should be |111> (idx 7)
    assert_almost_equal(main2.state[7].re, 1.0)
    assert_almost_equal(main2.state[6].re, 0.0)

    print("Success! Controlled appending matches expected results.")
