from butterfly.core.quantum_circuit import QuantumCircuit, QuantumRegister
from butterfly.core.state import QuantumState
from butterfly.core.executors import execute
from butterfly.algos.amplitude_estimation import amplitude_estimation_circuit
from butterfly.core.types import pi, FloatType
import math


fn main() raises:
    test_qae_rotation()
    print("Amplitude Estimation verification successful.")


fn test_qae_rotation() raises:
    # 1. Problem Setup
    # We want to estimate 'a', the probability of measuring '1' after an RY rotation.
    # State |psi> = RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    # Probability a = sin^2(theta/2)
    # Let theta = pi / 3
    # a = sin^2(pi/6) = (0.5)^2 = 0.25

    var theta = pi / 3.0
    var expected_prob = math.sin(theta / 2.0) ** 2
    print("Expected Probability (a):", expected_prob)

    # 2. Preparation Circuit A
    var num_state_qubits = 1
    var A = QuantumCircuit(num_state_qubits)
    A.ry(0, theta)  # Target qubit 0, angle theta

    # 3. Oracle O
    # Marks the state |1>
    var O = QuantumCircuit(num_state_qubits)
    O.z(0)  # Phase flip for |1> (Z gate is diag(1, -1))

    # 4. Amplitude Estimation
    var precision = 5  # Number of evaluation qubits
    var qc = amplitude_estimation_circuit(precision, A, O)

    var state = QuantumState(qc.num_qubits)
    execute(state, qc)

    # 5. Analyze Results
    # We measure the 'c' register (evaluation qubits).
    # The outcome 'y' (integer) relates to 'a' roughly by: a = sin^2(y * pi / 2^precision)

    # Find the most probable state in the evaluation register
    # The evaluation register is the first 'precision' qubits (0 to precision-1)
    # The state register is after that.
    # We need to sum probabilities over the state register for each evaluation outcome.

    var max_prob = 0.0
    var best_y = 0

    var N_eval = 1 << precision

    # Simplification: Assume measuring 0 on state register is enough?
    # Actually QAE entangles them. Better to find the peak probability in the full state vector
    # and map it back to y.
    # Or just iterate all 2^(n+m) states and bucket by evaluation bits.

    for y in range(N_eval):
        var prob_y: FloatType
        # Iterate over state register outcomes (0 and 1)
        # Outcome index = y * 2 + state_val
        # state_reg starts at qubit 'precision'
        # index = (state_val << precision) + y ? No.
        # Our register order: c (0..m-1), q (m)
        # Usually qubit 0 is LSB.
        # c register: 0 to m-1.
        # q register: m
        # index = (q_val << m) + c_val?
        # Let's check bit ordering in QuantumCircuit.
        # Usually indices are |qn ... q0>.
        # If c is added first, it is usually lower indices?
        # QuantumRegister just holds indices.
        # qc.add_register("c", n) -> indices 0..n-1
        # qc.add_register("q", m) -> indices n..n+m-1
        # So c is LSBs if we interpret index as integer.
        # state index k = (q_part << n) + c_part
        # So y is simply k & ((1<<n) - 1).

        # Sum prob for this y
        var idx0 = y  # q=0
        var idx1 = y + (1 << precision)  # q=1

        var amp0 = state[idx0]
        var amp1 = state[idx1]
        prob_y = (amp0.re**2 + amp0.im**2) + (amp1.re**2 + amp1.im**2)

        if prob_y > max_prob:
            max_prob = prob_y
            best_y = y

    # 6. Convert best_y to estimate
    # Note: Our Grover operator has a global phase shift of pi relative to the standard definition
    # G_mine = - G_standard. This shifts the eigenvalues by e^(i*pi).
    # So the estimated phase is 0.5 +/- theta/pi instead of theta/pi.
    # sin(0.5*pi + x) = cos(x). So we use cos^2 derived from the measured phase.

    var y_frac = FloatType(best_y) / FloatType(N_eval)
    var estimate = math.cos(y_frac * pi) ** 2

    print("Most probable outcome y:", best_y)
    print("Estimated Probability:", estimate)

    # Error Check
    var error = estimate - expected_prob
    if error < 0:
        error = -error
    print("Error:", error)

    if error > 0.1:
        print("Amplitude Estimation failed to converge.")
        raise Error("QAE Error too high")
