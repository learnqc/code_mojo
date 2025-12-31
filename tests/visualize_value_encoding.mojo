from butterfly.core.circuit import QuantumCircuit
from butterfly.algos.value_encoding_circuit import encode_value_circuit
from butterfly.utils.visualization import print_circuit, print_state
from butterfly.core.state import QuantumState


fn main() raises:
    alias n = 3
    var v: Float64 = 4.7

    print("=== Value Encoding Circuit (n=3, v=4.7) ===")

    # Create the circuit
    var qc = QuantumCircuit(n)
    encode_value_circuit(n, qc, v, swap=True)

    # 1. Visualize the circuit
    print("\n[Circuit Architecture]")
    print_circuit(qc)

    # 2. Run and visualize the final state
    print("\n[Final Quantum State]")
    var state = qc.run()
    print_state(state)

    print("\nSample completed successfully.")
