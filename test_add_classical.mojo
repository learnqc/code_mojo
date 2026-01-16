from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.state import QuantumState

# Example classical function
fn example_classical_op(mut state: QuantumState, targets: List[Int]) raises:
    # Simple example: this function could perform classical operations
    # on the quantum state (like measurements, bit flips, etc.)
    pass

fn test_add_classical():
    var circuit = QuantumCircuit(3)

    # Add a classical operation
    var targets = List[Int]()
    targets.append(0)
    targets.append(2)

    circuit.add_classical("example_op", targets, example_classical_op)

    print("✓ Classical transformation added to circuit")
    print("Circuit has", len(circuit.transformations), "transformations")

fn main():
    test_add_classical()

