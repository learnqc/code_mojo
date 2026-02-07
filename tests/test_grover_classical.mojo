from butterfly.algos.grover_classical import (
    grover,
    oracle,
    inversion,
    apply_grover_diffuser,
)
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
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.types import Amplitude, FloatType
from math import sqrt


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
    test_diffuser_matches_inversion_random()
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


fn test_diffuser_matches_inversion_random() raises:
    print("Testing diffuser matches inversion on random states...")
    alias n = 4
    var seeds = List[Int](11, 23, 37)

    for seed in seeds:
        var base = generate_state(n, seed)

        run_diffuser_case(
            base,
            List[Int](),
            List[Int](3, 6),
        )
        run_diffuser_case(
            base,
            List[Int](1, 3),
            List[Int](5),
        )
        run_diffuser_case(
            base,
            List[Int](2),
            List[Int](0, 15),
        )
        run_diffuser_case(
            base,
            List[Int](0, 1, 2, 3),
            List[Int](7),
        )


fn run_diffuser_case(
    base: QuantumState,
    targets: List[Int],
    items: List[Int],
) raises:
    var a = base.copy()
    var b = base.copy()

    apply_items_oracle(a, items)
    apply_items_oracle(b, items)

    apply_grover_diffuser(a, targets)
    apply_diffuser_via_inversion(b, targets)

    assert_state_close(a, b, FloatType(1e-6))


fn apply_items_oracle(
    mut state: QuantumState,
    items: List[Int],
) raises:
    for item in items:
        if item < 0 or item >= state.size():
            raise Error("Item out of bounds: " + String(item))
        state.re[item] = -state.re[item]
        state.im[item] = -state.im[item]


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
            raise Error("State mismatch at " + String(i))


fn apply_diffuser_via_inversion(
    mut state: QuantumState,
    targets: List[Int],
) raises:
    var targets_norm = normalize_targets(state, targets)
    var n = num_qubits_from_size(state.size())
    if len(targets_norm) == 0:
        return

    var is_target = List[Bool](length=n, fill=False)
    for t in targets_norm:
        is_target[t] = True
    var other_bits = List[Int]()
    for i in range(n):
        if not is_target[i]:
            other_bits.append(i)

    var num_targets = len(targets_norm)
    var block_size = 1 << num_targets
    var inv_sqrt = FloatType(1) / sqrt(FloatType(block_size))
    var blocks = 1 << len(other_bits)

    for block in range(blocks):
        var base = base_index_from_bits(block, other_bits)
        var original = List[Amplitude](
            length=block_size,
            fill=Amplitude(inv_sqrt, 0),
        )
        var current = List[Amplitude](
            length=block_size,
            fill=Amplitude(0, 0),
        )
        for tmask in range(block_size):
            var idx = base
            for j in range(num_targets):
                if (tmask >> j) & 1 == 1:
                    idx |= 1 << targets_norm[j]
            current[tmask] = state[idx]

        inversion(original, current)

        for tmask in range(block_size):
            var idx = base
            for j in range(num_targets):
                if (tmask >> j) & 1 == 1:
                    idx |= 1 << targets_norm[j]
            state[idx] = current[tmask]


fn num_qubits_from_size(size: Int) raises -> Int:
    if size <= 0 or (size & (size - 1)) != 0:
        raise Error("State size must be a power of two")
    var n = 0
    var tmp = size
    while tmp > 1:
        tmp >>= 1
        n += 1
    return n


fn sort_targets(mut targets: List[Int]):
    var n = len(targets)
    for i in range(n):
        for j in range(i + 1, n):
            if targets[i] > targets[j]:
                var tmp = targets[i]
                targets[i] = targets[j]
                targets[j] = tmp


fn normalize_targets(
    state: QuantumState,
    targets: List[Int],
) raises -> List[Int]:
    var n = num_qubits_from_size(state.size())
    if len(targets) == 0:
        var full = List[Int](capacity=n)
        for i in range(n):
            full.append(i)
        return full^
    var normalized = targets.copy()
    sort_targets(normalized)
    var seen = List[Bool](length=n, fill=False)
    for t in normalized:
        if t < 0 or t >= n:
            raise Error("Target out of bounds: " + String(t))
        if seen[t]:
            raise Error("Duplicate target: " + String(t))
        seen[t] = True
    return normalized^


fn base_index_from_bits(
    bits: Int,
    positions: List[Int],
) -> Int:
    var base = 0
    for i in range(len(positions)):
        if (bits >> i) & 1 == 1:
            base |= 1 << positions[i]
    return base
