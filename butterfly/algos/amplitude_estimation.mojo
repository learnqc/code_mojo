from butterfly.core.quantum_circuit import QuantumCircuit, QuantumRegister
from butterfly.algos.grover import grover_iterate_circuit, inversion_0_circuit
from butterfly.algos.value_encoding_circuit import iqft_circuit

fn append_amplitude_estimation(
    mut qc: QuantumCircuit,
    c: QuantumRegister,
    q: QuantumRegister,
    prepare: QuantumCircuit,
    oracle: QuantumCircuit,
    swap: Bool = True,
) raises:
    """
    Append Amplitude Estimation algorithm to a circuit using evaluation and state registers.
    Matches user's Python implementation logic.
    """
    var n = c.length

    # 1. Prepare state register
    _ = qc.append_circuit(prepare, q)

    # 2. Evaluations: Hadamards
    for i in range(n):
        qc.h(c.start + i) #TODO

    # 3. Iterative controlled Grover iterations
    # Manually construct Q = A S0 A^dag O
    var n_state = prepare.num_qubits
    var grover_op = QuantumCircuit(n_state)
    var q_dummy = QuantumRegister("q", n_state)

    _ = grover_op.append_circuit(oracle, q_dummy)
    _ = grover_op.append_circuit(prepare.inverse(), q_dummy)

    # We need inversion_0_circuit from grover, assuming it is public.
    _ = grover_op.append_circuit(inversion_0_circuit(n_state), q_dummy)

    _ = grover_op.append_circuit(prepare, q_dummy)

    var step = grover_op^

    for i in range(n):
        var ctrl_idx = c.start + (c.start + i if swap else c.start + n - 1 - i) #TODO
        for _ in range(1 << i):
            # c_append_circuit should be available in QuantumCircuit as verified
            _ = qc.c_append_circuit(ctrl_idx, step, q)

    # 4. Inverse QFT on evaluation register
    # if swap:
    #     qc.iqft(c, reversed=False, swap=True)
    # else:
    #     # Python: qc.iqft(c[::-1], swap=False)
    #     # In our API, reversed=True iterates the register backwards.
    #     _ = qc.iqft(c, reversed=True, swap=False)

    var iqft = QuantumCircuit(n)
    _ = iqft_circuit(iqft, [n - 1 - j for j in range(n)], swap=False) #TODO swap
    _ = qc.append_circuit(iqft, c)

fn amplitude_estimation_circuit(
    n: Int,
    prepare: QuantumCircuit,
    oracle: QuantumCircuit,
    swap: Bool = True,
) raises -> QuantumCircuit:
    """
    Factory to create a new Amplitude Estimation circuit.
    """
    var total_q = n + prepare.num_qubits
    var qc = QuantumCircuit(total_q)
    var c = QuantumRegister("c", n)
    var q = QuantumRegister("q", prepare.num_qubits)

    append_amplitude_estimation(qc, c, q, prepare, oracle, swap)

    return qc^
