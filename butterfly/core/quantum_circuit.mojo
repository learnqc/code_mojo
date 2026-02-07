from butterfly.core.circuit import (
    Register,
    Circuit,
    ClassicalTransformation,
    MeasurementTransformation,
    SwapTransformation,
    QubitReversalTransformation,
    Transformation,
)
from butterfly.core.state import (
    apply_bit_reverse,
    apply_permute_qubits,
    apply_swap,
    apply_measure,
)
from butterfly.core.state import QuantumState
from collections import List

alias QuantumRegister = Register
alias QuantumCircuit = Circuit[QuantumState]
alias ClassicalTransform = ClassicalTransformation[QuantumState]
alias MeasurementTransform = MeasurementTransformation[QuantumState]
alias QuantumTransformation = Transformation[QuantumState]


struct ClassicalReplacementKind(Copyable, Movable, ImplicitlyCopyable):
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    fn __str__(self) -> String:
        return classical_replacement_kind_name(self)

    @staticmethod
    fn from_int(value: Int) raises -> ClassicalReplacementKind:
        if not is_valid_classical_replacement_kind(value):
            raise Error(
                "Unknown classical replacement kind: " + String(value)
            )
        return ClassicalReplacementKind(value)

    alias SWAP = ClassicalReplacementKind(0)
    alias QUBIT_REVERSAL = ClassicalReplacementKind(1)


fn is_valid_classical_replacement_kind(value: Int) -> Bool:
    return value >= ClassicalReplacementKind.SWAP.value and value <= ClassicalReplacementKind.QUBIT_REVERSAL.value


fn classical_replacement_kind_name(kind: ClassicalReplacementKind) -> String:
    if kind == ClassicalReplacementKind.SWAP:
        return "SWAP"
    if kind == ClassicalReplacementKind.QUBIT_REVERSAL:
        return "QUBIT_REVERSAL"
    return "UNKNOWN"


struct ClassicalReplacementRule(Copyable, Movable):
    var name: String
    var apply: fn(
        List[QuantumTransformation],
        Int,
        mut List[QuantumTransformation],
    ) -> Int

    fn __init__(
        out self,
        name: String,
        apply: fn(
            List[QuantumTransformation],
            Int,
            mut List[QuantumTransformation],
        ) -> Int,
    ):
        self.name = name
        self.apply = apply


fn bit_reverse_tr() -> ClassicalTransform:
    return ClassicalTransform(
        "BITREV",
        List[Int](),
        apply_bit_reverse,
    )


fn bit_reverse_tr(targets: List[Int]) -> ClassicalTransform:
    return ClassicalTransform(
        "BITREV",
        targets,
        apply_bit_reverse,
    )


fn permute_qubits_tr(order: List[Int]) -> ClassicalTransform:
    return ClassicalTransform(
        "PERMUTE",
        order,
        apply_permute_qubits,
    )


fn swap_tr(a: Int, b: Int) -> ClassicalTransform:
    return ClassicalTransform(
        "SWAP",
        List[Int](a, b),
        apply_swap,
    )


fn bit_reverse(mut circuit: Circuit):
    circuit.transformations.append(bit_reverse_tr())


fn bit_reverse(mut circuit: Circuit, targets: List[Int]):
    circuit.transformations.append(bit_reverse_tr(targets))


fn permute_qubits(mut circuit: Circuit, order: List[Int]):
    circuit.transformations.append(permute_qubits_tr(order))


fn measure(mut circuit: Circuit, targets: List[Int]):
    circuit.transformations.append(
        MeasurementTransform(
            targets,
            None,
            apply_measure,
            [],
        )
    )


fn measure(
    mut circuit: Circuit,
    targets: List[Int],
    values: List[Optional[Bool]],
):
    circuit.transformations.append(
        MeasurementTransform(
            targets,
            None,
            apply_measure,
            values,
        )
    )


fn measure(
    mut circuit: Circuit,
    targets: List[Int],
    values: List[Optional[Bool]],
    seed: Int,
):
    circuit.transformations.append(
        MeasurementTransform(
            targets,
            seed,
            apply_measure,
            values,
        )
    )


fn replace_quantum_with_classical(mut circuit: QuantumCircuit) -> Int:
    return apply_classical_rewrites(circuit, default_classical_rules())


fn replace_quantum_with_classical(
    mut circuit: QuantumCircuit,
    rules: List[ClassicalReplacementRule],
) -> Int:
    return apply_classical_rewrites(circuit, rules)


fn replace_quantum_with_classical(
    mut circuit: QuantumCircuit,
    kinds: List[ClassicalReplacementKind],
) raises -> Int:
    return apply_classical_rewrites(circuit, classical_rules_for(kinds))


fn apply_classical_rewrites(
    mut circuit: QuantumCircuit,
    rules: List[ClassicalReplacementRule],
) -> Int:
    var replaced = 0
    var updated = List[QuantumTransformation](
        capacity=len(circuit.transformations)
    )
    var i = 0
    var total = len(circuit.transformations)
    while i < total:
        var consumed = 0
        for rule in rules:
            consumed = rule.apply(circuit.transformations, i, updated)
            if consumed > 0:
                replaced += 1
                break
        if consumed <= 0:
            updated.append(circuit.transformations[i].copy())
            i += 1
        else:
            i += consumed
    circuit.transformations = updated^
    return replaced


fn default_classical_rules() -> List[ClassicalReplacementRule]:
    var rules = List[ClassicalReplacementRule](capacity=2)
    rules.append(swap_replacement_rule())
    rules.append(qrev_replacement_rule())
    return rules^


fn classical_rules_for(
    kinds: List[ClassicalReplacementKind],
) raises -> List[ClassicalReplacementRule]:
    var selected = List[ClassicalReplacementRule](capacity=len(kinds))
    for kind in kinds:
        selected.append(classical_rule_for_kind(kind))
    return selected^


fn classical_rule_for_kind(
    kind: ClassicalReplacementKind,
) raises -> ClassicalReplacementRule:
    if kind == ClassicalReplacementKind.SWAP:
        return swap_replacement_rule()
    if kind == ClassicalReplacementKind.QUBIT_REVERSAL:
        return qrev_replacement_rule()
    raise Error(
        "Unknown classical replacement kind: " + String(kind.value)
    )


fn swap_replacement_rule() -> ClassicalReplacementRule:
    @always_inline
    fn apply(
        trs: List[QuantumTransformation],
        idx: Int,
        mut out: List[QuantumTransformation],
    ) -> Int:
        var tr = trs[idx]
        if not tr.isa[SwapTransformation]():
            return 0
        var swap_data = tr[SwapTransformation].copy()
        out.append(swap_tr(swap_data.a, swap_data.b))
        return 1

    return ClassicalReplacementRule("SWAP", apply)


fn qrev_replacement_rule() -> ClassicalReplacementRule:
    @always_inline
    fn apply(
        trs: List[QuantumTransformation],
        idx: Int,
        mut out: List[QuantumTransformation],
    ) -> Int:
        var tr = trs[idx]
        if not tr.isa[QubitReversalTransformation]():
            return 0
        var qrev_tr = tr[QubitReversalTransformation].copy()
        if len(qrev_tr.targets) == 0:
            out.append(bit_reverse_tr())
        else:
            out.append(bit_reverse_tr(qrev_tr.targets))
        return 1

    return ClassicalReplacementRule("QUBIT_REVERSAL", apply)
