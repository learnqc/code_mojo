from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.algos.amplitude_estimation import amplitude_estimation_circuit
from butterfly.core.types import pi
from butterfly.utils.visualization import animate_execution_table


fn main() raises:
    test_animate_amplitude_estimation()

fn test_animate_amplitude_estimation() raises:
    print("Animating amplitude estimation...")
    # Estimate a = sin^2(theta / 2) for a single-qubit RY preparation.
    var theta = pi / 3.0

    var num_state_qubits = 1
    var A = QuantumCircuit(num_state_qubits)
    A.ry(0, theta)

    var O = QuantumCircuit(num_state_qubits)
    O.z(0)  # Mark |1>

    var precision = 3
    var qc = amplitude_estimation_circuit(precision, A, O)

    var state = QuantumState(qc.num_qubits)

    # Animate as a table; small delay to auto-advance so CI/local runs don't block.
    animate_execution_table(
        qc,
        state,
        short=True,
        use_color=True,
        show_step_label=True,
        delay_s=0.25,
        step_on_input=False,
        redraw_in_place=True,
    )

    print("Amplitude estimation animation complete.")
