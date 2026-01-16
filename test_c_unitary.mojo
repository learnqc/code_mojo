from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.types import Amplitude

fn main() raises:
    print("Testing multi-control c_unitary method")
    print("=" * 40)

    # Create a simple single-qubit unitary (Pauli-X gate)
    var u = List[Amplitude](capacity=4)
    u.append(Amplitude(0.0, 0.0))  # |0⟩ -> |0⟩
    u.append(Amplitude(1.0, 0.0))  # |0⟩ -> |1⟩
    u.append(Amplitude(1.0, 0.0))  # |1⟩ -> |0⟩
    u.append(Amplitude(0.0, 0.0))  # |1⟩ -> |1⟩

    # Create circuit with multi-control unitary
    var circuit = QuantumCircuit(3)

    # Add multi-controlled unitary: controls on qubits 0,1, target on qubit 2
    var controls = List[Int]()
    controls.append(0)
    controls.append(1)
    circuit.c_unitary(u^, controls, 2, "multi_control_x")

    print("✓ Multi-control unitary added to circuit")
    print("Circuit has", len(circuit.transformations), "transformations")
