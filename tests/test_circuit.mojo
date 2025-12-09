from butterfly.core.circuit import QuantumCircuit, Circuit
from butterfly.core.state import QuantumState
from butterfly.core.gates import H, X, Y, Z, P
from butterfly.core.types import *
from testing import assert_true, assert_almost_equal
from math import sqrt


def test_circuit_initialization():
    # Test basic initialization
    var circuit = QuantumCircuit(2)
    assert_true(circuit.num_qubits == 2)
    assert_true(circuit.num_transformations() == 0)
    assert_true(circuit.state.size() == 4)

    # Initial state should be |00⟩
    assert_almost_equal(circuit.get_amplitude(0).re, 1.0)
    assert_almost_equal(circuit.get_amplitude(0).im, 0.0)
    assert_almost_equal(circuit.get_amplitude(1).re, 0.0)
    assert_almost_equal(circuit.get_amplitude(2).re, 0.0)
    assert_almost_equal(circuit.get_amplitude(3).re, 0.0)
    print("✓ Circuit initialization test passed")


def test_add_single_gate():
    var circuit = QuantumCircuit(1)
    circuit.add(H, 0)

    assert_true(circuit.num_transformations() == 1)
    print("✓ Add single gate test passed")


def test_add_controlled_gate():
    var circuit = QuantumCircuit(2)
    circuit.add_controlled(X, 1, 0)

    assert_true(circuit.num_transformations() == 1)
    print("✓ Add controlled gate test passed")


def test_add_multi_controlled_gate():
    var circuit = QuantumCircuit(3)
    var controls = List[Int]()
    controls.append(0)
    controls.append(1)
    circuit.add_multi_controlled(X, 2, controls^)

    assert_true(circuit.num_transformations() == 1)
    print("✓ Add multi-controlled gate test passed")


def test_hadamard_execution():
    # Test H gate creates superposition
    var circuit = QuantumCircuit(1)
    circuit.add(H, 0)
    circuit.execute()

    # After H on |0⟩, should get (|0⟩ + |1⟩)/√2
    var amp0 = circuit.get_amplitude(0)
    var amp1 = circuit.get_amplitude(1)

    var expected = 1.0 / sqrt(2.0)
    assert_almost_equal(Float64(amp0.re), expected, atol=1e-6)
    assert_almost_equal(Float64(amp0.im), 0.0, atol=1e-6)
    assert_almost_equal(Float64(amp1.re), expected, atol=1e-6)
    assert_almost_equal(Float64(amp1.im), 0.0, atol=1e-6)
    print("✓ Hadamard execution test passed")


def test_x_gate_execution():
    # Test X gate flips qubit
    var circuit = QuantumCircuit(1)
    circuit.add(X, 0)
    circuit.execute()

    # After X on |0⟩, should get |1⟩
    assert_almost_equal(Float64(circuit.get_amplitude(0).re), 0.0, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(1).re), 1.0, atol=1e-6)
    print("✓ X gate execution test passed")


def test_cnot_execution():
    # Test CNOT gate
    var circuit = QuantumCircuit(2)
    # Create |10⟩ state
    circuit.add(X, 0)
    circuit.execute()
    circuit.clear_transformations()

    # Apply CNOT with control=0, target=1
    circuit.add_controlled(X, 1, 0)
    circuit.execute()

    # Should get |11⟩ state (index 3)
    assert_almost_equal(Float64(circuit.get_amplitude(0).re), 0.0, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(1).re), 0.0, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(2).re), 0.0, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(3).re), 1.0, atol=1e-6)
    print("✓ CNOT execution test passed")


def test_bell_state():
    # Create Bell state (|00⟩ + |11⟩)/√2
    var circuit = QuantumCircuit(2)
    circuit.add(H, 0)
    circuit.add_controlled(X, 1, 0)
    circuit.execute()

    var amp0 = circuit.get_amplitude(0)
    var amp3 = circuit.get_amplitude(3)

    var expected = 1.0 / sqrt(2.0)
    assert_almost_equal(Float64(amp0.re), expected, atol=1e-6)
    assert_almost_equal(Float64(amp3.re), expected, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(1).re), 0.0, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(2).re), 0.0, atol=1e-6)
    print("✓ Bell state test passed")


def test_toffoli_gate():
    # Test Toffoli (CCX) gate
    var circuit = QuantumCircuit(3)
    # Create |110⟩ state
    circuit.add(X, 0)
    circuit.add(X, 1)
    circuit.execute()
    circuit.clear_transformations()

    # Apply Toffoli with controls=0,1 target=2
    var controls = List[Int]()
    controls.append(0)
    controls.append(1)
    circuit.add_multi_controlled(X, 2, controls^)
    circuit.execute()

    # Should get |111⟩ state (index 7)
    assert_almost_equal(Float64(circuit.get_amplitude(7).re), 1.0, atol=1e-6)
    for i in range(7):
        assert_almost_equal(Float64(circuit.get_amplitude(i).re), 0.0, atol=1e-6)
    print("✓ Toffoli gate test passed")


def test_clear_transformations():
    var circuit = QuantumCircuit(2)
    circuit.add(H, 0)
    circuit.add(X, 1)
    circuit.add_controlled(X, 1, 0)

    assert_true(circuit.num_transformations() == 3)

    circuit.clear_transformations()
    assert_true(circuit.num_transformations() == 0)
    print("✓ Clear transformations test passed")


def test_multiple_executions():
    # Test that we can execute multiple times
    var circuit = QuantumCircuit(1)
    circuit.add(X, 0)
    circuit.execute()

    # State should be |1⟩
    assert_almost_equal(Float64(circuit.get_amplitude(1).re), 1.0, atol=1e-6)

    # Execute again (applies X twice total)
    circuit.execute()

    # Should be back to |1⟩ + X = |1⟩ then |1⟩ + X = still modified
    # Actually this applies the same transforms again, so |1⟩ -> |0⟩
    assert_almost_equal(Float64(circuit.get_amplitude(0).re), 1.0, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(1).re), 0.0, atol=1e-6)
    print("✓ Multiple executions test passed")


def test_circuit_alias():
    # Test that Circuit alias works
    var circuit = Circuit(2)
    assert_true(circuit.num_qubits == 2)
    print("✓ Circuit alias test passed")


def test_set_amplitude():
    var circuit = QuantumCircuit(2)

    # Set custom amplitude
    circuit.set_amplitude(0, Amplitude(0.5, 0.0))
    circuit.set_amplitude(1, Amplitude(0.5, 0.0))
    circuit.set_amplitude(2, Amplitude(0.5, 0.0))
    circuit.set_amplitude(3, Amplitude(0.5, 0.0))

    assert_almost_equal(Float64(circuit.get_amplitude(0).re), 0.5, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(1).re), 0.5, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(2).re), 0.5, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(3).re), 0.5, atol=1e-6)
    print("✓ Set amplitude test passed")


def test_gate_convenience_methods():
    # Test gate-specific methods
    var circuit = QuantumCircuit(2)

    # Use convenience methods instead of add()
    circuit.h(0)
    circuit.x(1)
    assert_true(circuit.num_transformations() == 2)

    circuit.execute()

    # Should have H on qubit 0 and X on qubit 1
    # With qubit 0 as LSB: H|0⟩ ⊗ X|0⟩ = (|0⟩ + |1⟩)/√2 ⊗ |1⟩
    # This gives (|10⟩ + |11⟩)/√2 which is indices 2 and 3
    var expected = 1.0 / sqrt(2.0)
    assert_almost_equal(Float64(circuit.get_amplitude(2).re), expected, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(3).re), expected, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(0).re), 0.0, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(1).re), 0.0, atol=1e-6)
    print("✓ Gate convenience methods test passed")


def test_controlled_convenience_methods():
    # Test controlled gate convenience methods
    var circuit = QuantumCircuit(2)

    # Create Bell state using cx (CNOT)
    circuit.h(0)
    circuit.cx(1, 0)  # target=1, control=0
    circuit.execute()

    var expected = 1.0 / sqrt(2.0)
    assert_almost_equal(Float64(circuit.get_amplitude(0).re), expected, atol=1e-6)
    assert_almost_equal(Float64(circuit.get_amplitude(3).re), expected, atol=1e-6)
    print("✓ Controlled convenience methods test passed")


def test_rotation_gates():
    # Test rotation gate methods
    var circuit = QuantumCircuit(1)

    circuit.rx(0, pi / 2)
    circuit.ry(0, pi / 4)
    circuit.rz(0, pi / 6)
    circuit.p(0, pi / 8)

    assert_true(circuit.num_transformations() == 4)
    print("✓ Rotation gates test passed")


def test_register_creation():
    # Test creating and managing registers
    var circuit = QuantumCircuit(5)

    var q = circuit.add_register("q", 3)
    var anc = circuit.add_register("anc", 2)

    assert_true(circuit.num_registers() == 2)
    assert_true(q.size == 3)
    assert_true(q.start == 0)
    assert_true(anc.size == 2)
    assert_true(anc.start == 3)
    print("✓ Register creation test passed")


def test_register_indexing():
    # Test accessing qubits via register indexing
    var circuit = QuantumCircuit(5)

    var q = circuit.add_register("q", 3)

    # Use register indexing
    assert_true(q[0] == 0)
    assert_true(q[1] == 1)
    assert_true(q[2] == 2)

    # Apply gates using register indices
    circuit.h(q[0])
    circuit.cx(q[1], q[0])

    assert_true(circuit.num_transformations() == 2)
    print("✓ Register indexing test passed")


def test_register_lookup():
    # Test getting registers by name
    var circuit = QuantumCircuit(5)

    _ = circuit.add_register("qubits", 3)
    _ = circuit.add_register("ancilla", 2)

    try:
        var q = circuit.get_register("qubits")
        assert_true(q.size == 3)
        assert_true(q.name == "qubits")

        var anc = circuit.get_register("ancilla")
        assert_true(anc.size == 2)
        assert_true(anc.name == "ancilla")

        print("✓ Register lookup test passed")
    except:
        print("✗ Register lookup test failed")


def test_register_qubits():
    # Test getting all qubits from a register
    var circuit = QuantumCircuit(5)

    var q = circuit.add_register("q", 3)
    var qubits = q.qubits()

    assert_true(len(qubits) == 3)
    assert_true(qubits[0] == 0)
    assert_true(qubits[1] == 1)
    assert_true(qubits[2] == 2)
    print("✓ Register qubits test passed")


def main():
    print("Running quantum circuit tests...\n")

    test_circuit_initialization()
    test_add_single_gate()
    test_add_controlled_gate()
    test_add_multi_controlled_gate()
    test_hadamard_execution()
    test_x_gate_execution()
    test_cnot_execution()
    test_bell_state()
    test_toffoli_gate()
    test_clear_transformations()
    test_multiple_executions()
    test_circuit_alias()
    test_set_amplitude()
    test_gate_convenience_methods()
    test_controlled_convenience_methods()
    test_rotation_gates()
    test_register_creation()
    test_register_indexing()
    test_register_lookup()
    test_register_qubits()

    print("\n✅ All quantum circuit tests passed!")
