from butterfly.core.circuit import (
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


alias QuantumCircuit = Circuit[QuantumState]
alias ClassicalTransform = ClassicalTransformation[QuantumState]
alias MeasurementTransform = MeasurementTransformation[QuantumState]
alias QuantumTransformation = Transformation[QuantumState]


struct TransformationKind:
    alias SWAP = -1
    alias QUBIT_REVERSAL = -2


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
    return classical_for_types(
        circuit,
        List[Int](TransformationKind.SWAP, TransformationKind.QUBIT_REVERSAL),
    )


fn classical_for_types(
    mut circuit: QuantumCircuit,
    kinds: List[Int],
) -> Int:
    @always_inline
    fn has_kind(kinds: List[Int], kind: Int) -> Bool:
        for k in kinds:
            if k == kind:
                return True
        return False

    var replaced = 0
    var updated = List[QuantumTransformation](
        capacity=len(circuit.transformations)
    )
    for tr in circuit.transformations:
        if tr.isa[SwapTransformation]():
            if has_kind(kinds, TransformationKind.SWAP):
                var swap_data = tr[SwapTransformation].copy()
                updated.append(
                    swap_tr(swap_data.a, swap_data.b)
                )
                replaced += 1
            else:
                updated.append(tr.copy())
        elif tr.isa[QubitReversalTransformation]():
            if has_kind(kinds, TransformationKind.QUBIT_REVERSAL):
                var qrev_tr = tr[QubitReversalTransformation].copy()
                if len(qrev_tr.targets) == 0:
                    updated.append(bit_reverse_tr())
                else:
                    updated.append(bit_reverse_tr(qrev_tr.targets))
                replaced += 1
            else:
                updated.append(tr.copy())
        else:
            updated.append(tr.copy())
    circuit.transformations = updated^
    return replaced
