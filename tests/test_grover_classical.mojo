from butterfly.algos.grover_classical import grover, oracle
from butterfly.algos.grover import (
    grover_iterate_circuit,
    classical_oracle,
)
from butterfly.core.quantum_circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalTransform,
    ClassicalReplacementKind,
    replace_quantum_with_classical,
)
from butterfly.core.executors import execute
from butterfly.core.state import QuantumState


fn is_target_7(i: Int) -> Bool:
    return i == 7


fn is_target_5(i: Int) -> Bool:
    return i == 5


fn oracle_target_7(mut state: QuantumState) raises:
    oracle(state, is_target_7)


fn oracle_target_5(mut state: QuantumState) raises:
    oracle(state, is_target_5)


fn build_uniform_state(n: Int) raises -> QuantumState:
    var q = QuantumRegister("q", n)
    var qc = QuantumCircuit(q)
    for i in range(n):
        qc.h(i)
    var state = QuantumState(n)
    execute(state, qc)
    return state^


fn main() raises:
    test_grover_classical_target_7()
    test_grover_classical_target_5()
    test_grover_diffuser_replacement()
    print("Classical Grover tests passed!")


fn test_grover_classical_target_7() raises:
    print("Testing classical Grover (target |111>)...")
    alias n = 3
    var state = build_uniform_state(n)
    grover(state, oracle_target_7, 2, predicate=is_target_7, check=True)
    var prob = state[7].re ** 2 + state[7].im ** 2
    print("Probability |111⟩:", prob)
    if prob < 0.9:
        raise Error("Low probability")


fn test_grover_classical_target_5() raises:
    print("Testing classical Grover (target |101>)...")
    alias n = 3
    var state = build_uniform_state(n)
    grover(state, oracle_target_5, 2, predicate=is_target_5, check=True)
    var prob = state[5].re ** 2 + state[5].im ** 2
    print("Probability |101⟩:", prob)
    if prob < 0.9:
        raise Error("Low probability")


fn test_grover_diffuser_replacement() raises:
    print("Testing Grover diffuser replacement (shortcut oracle)...")
    alias n = 3
    var items = List[Int]()
    items.append(6)  # Target |110>

    var q = QuantumRegister("q", n)
    var qc = QuantumCircuit(q)
    for i in range(n):
        qc.h(i)

    var iterations = 2
    var oracle = ClassicalTransform(
        "sign_flip_oracle",
        items,
        classical_oracle,
    )
    var step = grover_iterate_circuit(oracle^, n)
    _ = replace_quantum_with_classical(
        step,
        List[ClassicalReplacementKind](
            ClassicalReplacementKind.GROVER_DIFFUSER,
        ),
    )
    for _ in range(iterations):
        _ = qc.append_circuit(step, q)

    var state = QuantumState(n)
    execute(state, qc)

    var prob = state[6].re ** 2 + state[6].im ** 2
    print("Probability |110⟩:", prob)
    if prob < 0.9:
        raise Error("Low probability")
