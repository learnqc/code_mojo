from butterfly.algos.qft_classical import apply_qft_classical, apply_iqft_classical
from butterfly.algos.value_encoding_circuit import iqft_circuit
from butterfly.core.quantum_circuit import QuantumCircuit, QuantumRegister
from butterfly.core.executors import execute
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.types import FloatType


fn assert_state_close(
    a: QuantumState,
    b: QuantumState,
    tol: FloatType,
) raises:
    if a.size() != b.size():
        raise Error("State size mismatch")
    for i in range(a.size()):
        var dr = a.re[i] - b.re[i]
        var di = a.im[i] - b.im[i]
        if dr < 0:
            dr = -dr
        if di < 0:
            di = -di
        if dr > tol or di > tol:
            print("State mismatch at index " + String(i))
            print("A:", a.re[i], a.im[i])
            print("B:", b.re[i], b.im[i])
            print_state(a, "A")
            print_state(b, "B")
            raise Error("State mismatch at " + String(i))


fn print_state(state: QuantumState, label: String):
    print("State", label, "size", state.size())
    for i in range(state.size()):
        var re = state.re[i]
        var im = state.im[i]
        print(i, ":", re, im)


fn apply_qft_quantum(
    state: QuantumState,
    targets: List[Int],
    swap: Bool,
) raises -> QuantumState:
    var n = 0
    var tmp = state.size()
    while tmp > 1:
        tmp >>= 1
        n += 1
    var q = QuantumRegister("q", n)
    var qc = QuantumCircuit(q)
    iqft_circuit(qc, targets, swap=swap)
    qc = qc.inverse()
    var out = state.copy()
    execute(out, qc)
    return out^


fn apply_iqft_quantum(
    state: QuantumState,
    targets: List[Int],
    swap: Bool,
) raises -> QuantumState:
    var n = 0
    var tmp = state.size()
    while tmp > 1:
        tmp >>= 1
        n += 1
    var q = QuantumRegister("q", n)
    var qc = QuantumCircuit(q)
    iqft_circuit(qc, targets, swap=swap)
    var out = state.copy()
    execute(out, qc)
    return out^


fn run_case(
    state: QuantumState,
    targets: List[Int],
    swap: Bool,
) raises:
    var quantum_qft = apply_qft_quantum(state, targets, swap)
    var classical_qft = state.copy()
    apply_qft_classical(classical_qft, targets, swap=swap)
    assert_state_close(quantum_qft, classical_qft, FloatType(1e-5))

    var quantum_iqft = apply_iqft_quantum(state, targets, swap)
    var classical_iqft = state.copy()
    apply_iqft_classical(classical_iqft, targets, swap=swap)
    assert_state_close(quantum_iqft, classical_iqft, FloatType(1e-5))


fn main() raises:
    print("Testing classical QFT/IQFT against gate-based circuits...")
    alias n = 4
    var seeds = List[Int](3, 7, 11)
    for seed in seeds:
        var state = generate_state(n, seed)
        run_case(state, List[Int](3, 2, 1, 0), swap=False)
        run_case(state, List[Int](3, 2, 1, 0), swap=True)
        run_case(state, List[Int](3, 1), swap=False)
        run_case(state, List[Int](3, 1), swap=True)
    print("Classical QFT/IQFT tests passed!")
