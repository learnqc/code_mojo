"""
Circuit-based value encoding
"""

from butterfly.core.circuit import (
    QuantumCircuit,
    QuantumRegister,
    execute_simd_v2,
)
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.core.state import QuantumState
from butterfly.core.gates import H, P
from butterfly.core.types import FloatType
from math import pi
from butterfly.utils.visualization import print_state
from butterfly.core.execution_strategy import SIMD_V2, FUSED_V3


fn encode_value_circuit_parametric[
    n: Int
](mut circuit: QuantumCircuit, v: FloatType, swap: Bool = True):
    encode_value_circuit(n, circuit, v, swap)


fn encode_value_circuit(
    n: Int, mut circuit: QuantumCircuit, v: FloatType, swap: Bool = True
):
    """Build value encoding circuit using correct IQFT implementation."""
    from butterfly.algos.qft import iqft

    for j in range(n):
        circuit.h(j)
    for j in range(n):
        if swap:
            circuit.p(j, 2 * pi / 2 ** (n - j) * v)
        else:
            circuit.p(j, 2 * pi / 2 ** (j + 1) * v)

    var targets = [n - 1 - j for j in range(n)]
    if swap:
        targets = [j for j in range(n)]
    iqft(circuit=circuit, targets=targets, do_swap=swap)


fn encode_value_circuits[
    n: Int
](v: FloatType, swap: Bool = True) -> List[QuantumCircuit]:
    """Split value encoding into 3 circuits for hybrid execution.

    Returns:
        List of 3 circuits:
        1. Preparation: H gates + P gates (non-controlled, good for v3).
        2. IQFT: Inverse QFT transformation (controlled gates, good for v2).
        3. Bit reversal: Permutation (if swap=False, otherwise empty).
    """
    # Build the full circuit first
    var full_circuit = QuantumCircuit(n)
    encode_value_circuit(n, full_circuit, v, swap)

    # Count transformations: n H gates + n P gates = 2n gates before IQFT
    var prep_end = 2 * n

    # Split transformations into 3 circuits
    var circuits = List[QuantumCircuit]()

    # Circuit 1: Preparation (H + P gates)
    var prep = QuantumCircuit(n)
    for i in range(prep_end):
        prep.transformations.append(full_circuit.transformations[i])
    circuits.append(prep^)

    # Circuit 2: IQFT (everything after preparation)
    var iqft_circuit = QuantumCircuit(n)
    for i in range(prep_end, len(full_circuit.transformations)):
        iqft_circuit.transformations.append(full_circuit.transformations[i])
    circuits.append(iqft_circuit^)

    # Circuit 3: Bit reversal (empty for now, can be extracted later if needed)
    var bit_rev = QuantumCircuit(n)
    circuits.append(bit_rev^)

    return circuits^


fn encode_value_circuits_runtime(
    n: Int, v: FloatType, swap: Bool = False
) -> List[QuantumCircuit]:
    """Non-parametric version for runtime dispatch in benchmarks.

    Split value encoding into 3 circuits for hybrid execution.

    Returns:
        List of 3 circuits:
        1. Preparation: H gates + P gates (non-controlled, good for v3).
        2. IQFT: Inverse QFT transformation (controlled gates, good for v2).
        3. Bit reversal: Permutation (if swap=False, otherwise empty).
    """
    var circuits = List[QuantumCircuit]()

    # Circuit 1: Preparation (H + P gates)
    var prep = QuantumCircuit(n)
    for j in range(n):
        prep.h(j)
    for j in range(n):
        if swap:
            prep.p(j, 2 * pi / 2 ** (n - j) * v)
        else:
            prep.p(j, 2 * pi / 2 ** (j + 1) * v)
    circuits.append(prep^)

    # Circuit 2: IQFT
    from butterfly.algos.qft import iqft

    var iqft_circuit = QuantumCircuit(n)
    targets = [n - 1 - j for j in range(n)]
    if swap:
        targets = [j for j in range(n)]
    # iqft_circuit.iqft(QuantumRegister("", 0, n), reversed=not swap, swap=False)
    iqft(circuit=iqft_circuit, targets=targets, do_swap=False)
    circuits.append(iqft_circuit^)

    # Circuit 3: Bit reversal (empty - bit reversal is handled by IQFT swap parameter)
    var bit_rev = QuantumCircuit(n)
    if swap:
        bit_rev.bit_reverse()
    circuits.append(bit_rev^)

    return circuits^


fn main() raises:
    alias n = 3
    alias N = 1 << n
    v = 4.7

    # qc = QuantumCircuit(n)
    # encode_value_circuit[n](qc, v, True)

    # var state = QuantumState(n)
    # execute_fused_v3[N](state, qc)
    # print_state(state)

    # var state2 = QuantumState(n)
    # execute_simd_v2[n](state2, qc)
    # print_state(state2)

    # var state3 = QuantumState(n)
    # qc.execute_with_strategy[n](state3, SIMD_V2)
    # print_state(state3)

    # var state4 = QuantumState(n)
    # qc.execute_with_strategy[n](state4, FUSED_V3)
    # print_state(state4)

    # var state5 = QuantumState(n)
    # qc.execute(state5)
    # print_state(state5)

    # var state6 = qc.run_with_strategy(FUSED_V3)
    # print_state(state6)

    # var state7 = qc.run()
    # print_state(state7)

    circuits = encode_value_circuits_runtime(n, v, True)
    var state8 = QuantumState(n)
    circuits[0].execute_with_strategy[n](state8, FUSED_V3)
    circuits[1].execute_with_strategy[n](state8, SIMD_V2)
    # circuits[1].execute(state8)
    # circuits[1].run()
    circuits[2].execute_with_strategy[n](state8, SIMD_V2)
    print_state(state8)
