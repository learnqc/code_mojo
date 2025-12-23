from butterfly.core.circuit import QuantumCircuit, Register, QFT, IQFT
from butterfly.core.types import pi
from butterfly.utils.visualization import print_state
from testing import assert_almost_equal
from math import sqrt


fn main() raises:
    print("--- Testing Value Encoding (n=3, v=4.7) ---")

    var n = 3
    var v = 4.7
    var qc = QuantumCircuit(n)
    var reg = qc.add_register("q", n)

    # 1. Uniform superposition
    for i in range(n):
        qc.h(i)

    # 2. Phase encoding: exp(2*pi*i * v * index / 2^n)
    for j in range(n):
        # We target qubit j with phase corresponding to its weight 2^j
        # theta = 2 * pi * v * 2^j / 2^n = 2 * pi * v / 2^(n-j)
        var theta = 2 * pi * v / (2.0 ** (n - j))
        qc.p(j, theta)

    # 3. Inverse QFT
    qc.iqft(reg)

    # execute the circuit
    qc.execute()

    # 4. Print the resulting state table
    print("Resulting State Table (v=4.7):")
    print_state(qc.state)

    print(
        "\nNote: v=4.7 (normalized to 4.7/8 = 0.5875) should peak near index 5"
        " (5/8 = 0.625) or 4 (4/8 = 0.5)."
    )
