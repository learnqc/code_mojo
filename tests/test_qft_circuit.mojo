from butterfly.core.circuit import QuantumCircuit, Register, QFT, IQFT
from testing import assert_almost_equal
from math import pi


fn main() raises:
    print("--- Testing QFT and IQFT Circuit Implementations ---")

    # 1. Test QFT + IQFT = Identity
    var n = 3
    var qc = QuantumCircuit(n)
    var reg = qc.add_register("data", n)

    # Apply some initial gate
    qc.x(0)  # State |001>

    # Apply QFT
    qc.qft(reg)

    # Apply IQFT
    qc.iqft(reg)

    qc.execute()

    # Should be back to |001> (index 1)
    assert_almost_equal(qc.state[1].re, 1.0)
    print("Test 1 (QFT + IQFT = Identity) passed.")

    # 2. Test QFT factory
    var qft_qc = QFT(2)
    qft_qc.execute()
    # QFT on |00> should give 1/2(|00> + |01> + |10> + |11>)
    # amplitude should be 0.5
    for i in range(4):
        assert_almost_equal(qft_qc.state[i].re, 0.5)
    print("Test 2 (QFT factory) passed.")

    # 3. Test IQFT with reversed targets
    var qc_rev = QuantumCircuit(2)
    var reg_rev = qc_rev.add_register("reg", 2)
    qc_rev.qft(reg_rev, reversed=True)
    qc_rev.iqft(reg_rev, reversed=True)
    qc_rev.execute()
    # Should still be back to |00>
    assert_almost_equal(qc_rev.state[0].re, 1.0)
    print("Test 3 (Reversed QFT/IQFT) passed.")

    print("Success! QFT and IQFT work correctly.")
