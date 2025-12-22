from butterfly.core.circuit import QuantumCircuit
from butterfly.core.state import generate_state, QuantumState
from butterfly.algos.qft import qft, iqft
from butterfly.core.types import Amplitude, Type
from butterfly.core.fft_v5 import apply_cft
import math


def test_v5_contiguous(
    n_qubits: Int, targets: List[Int], inverse: Bool, do_swap: Bool
):
    var label = "Forward" if not inverse else "Inverse"
    var swap_label = "With Swap" if do_swap else "No Swap"
    print("Testing CFT delegation (" + label + ", " + swap_label + ")...")

    var state_ref = generate_state(n_qubits)
    for i in range(1 << n_qubits):
        state_ref.re[i] = math.cos(Float64(i))
        state_ref.im[i] = math.sin(Float64(i))

    var state_v5 = state_ref.copy()

    # 1. Reference QFT
    var circ = QuantumCircuit(n_qubits)
    circ.state = state_ref^
    if not inverse:
        qft(circ, targets, do_swap=do_swap)
    else:
        iqft(circ, targets, do_swap=do_swap)
    circ.execute()

    # 2. v5 Implementation (via apply_cft)
    apply_cft(state_v5, targets, inverse=inverse, do_swap=do_swap)

    # Compare
    var diff_sum: Float64 = 0.0
    for i in range(1 << n_qubits):
        diff_sum += abs(circ.state.re[i] - state_v5.re[i]) + abs(
            circ.state.im[i] - state_v5.im[i]
        )

    print("  Diff Sum:", diff_sum)
    if diff_sum > 1e-10:
        print("  FAILED")
    else:
        print("  PASSED")


fn main() raises:
    # Test cases: (n_qubits, targets)
    # 1. Full range
    test_v5_contiguous(4, List[Int](0, 1, 2, 3), inverse=False, do_swap=True)
    test_v5_contiguous(4, List[Int](0, 1, 2, 3), inverse=True, do_swap=True)

    # 2. Lower register
    test_v5_contiguous(5, List[Int](0, 1, 2), inverse=False, do_swap=True)
    test_v5_contiguous(5, List[Int](0, 1, 2), inverse=False, do_swap=True)
    test_v5_contiguous(5, List[Int](0, 1, 2), inverse=True, do_swap=True)

    # 2b. Full register (Delegation Check for N=32)
    test_v5_contiguous(5, List[Int](0, 1, 2, 3, 4), inverse=False, do_swap=True)

    # 3. Middle register
    test_v5_contiguous(6, List[Int](1, 2, 3, 4), inverse=False, do_swap=True)
    test_v5_contiguous(6, List[Int](1, 2, 3, 4), inverse=True, do_swap=True)

    # 4. Upper register
    test_v5_contiguous(6, List[Int](3, 4, 5), inverse=False, do_swap=True)
    test_v5_contiguous(6, List[Int](3, 4, 5), inverse=True, do_swap=True)

    # 5. Single qubit (extreme case)
    test_v5_contiguous(4, List[Int](2), inverse=False, do_swap=True)

    # 6. Large-ish system to check parallelization
    test_v5_contiguous(10, List[Int](4, 5, 6), inverse=False, do_swap=True)

    print("\nAll tests finished.")
