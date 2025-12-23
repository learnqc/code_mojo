from butterfly.core.state import QuantumState, iqft as iqft_state
from butterfly.core.circuit import QuantumCircuit
from butterfly.algos.qft import iqft as iqft_circuit
from testing import assert_almost_equal
from butterfly.core.types import Amplitude
from collections import List


fn test_iqft_circuit() raises:
    var n = 4
    var targets = List[Int]()
    for i in range(n):
        targets.append(i)

    # 1. State-based IQFT
    var state1 = QuantumState(n)
    # Set some initial state
    state1[3] = Amplitude(0.707, 0)
    state1[7] = Amplitude(0.707, 0)
    iqft_state(state1, targets, swap=True)

    # 2. Circuit-based IQFT
    var state2 = QuantumState(n)
    state2[3] = Amplitude(0.707, 0)
    state2[7] = Amplitude(0.707, 0)
    var circuit = QuantumCircuit(state2^, n)
    iqft_circuit(circuit, targets, do_swap=True)
    circuit.execute()

    # Compare
    for i in range(1 << n):
        assert_almost_equal(state1[i].re, circuit.state[i].re)
        assert_almost_equal(state1[i].im, circuit.state[i].im)

    print("IQFT Circuit Verification Passed!")


fn main() raises:
    test_iqft_circuit()
