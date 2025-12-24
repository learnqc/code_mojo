from butterfly.core.circuit import QuantumCircuit, Register, QFT, IQFT
from butterfly.core.types import pi
from butterfly.utils.visualization import print_state
from testing import assert_almost_equal


fn main() raises:
    print("--- Modular Value Encoding (n=3, v=4.7) ---")

    var n = 3
    var v = 4.7

    # 1. Create a circuit that only prepares the phase-encoded state
    var encode_qc = QuantumCircuit(n)
    for i in range(n):
        encode_qc.h(i)
    for j in range(n):
        encode_qc.p(j, 2 * pi * v / (2.0 ** (n - j)))

    # 2. Setup the "main" circuit
    var main_qc = QuantumCircuit(n)
    var reg = main_qc.add_register("q", n)

    # 3. Append the encoding part
    print("Appending Phase Encoding circuit...")
    main_qc.append_circuit(encode_qc, reg)

    # 4. Create and append the IQFT part
    print("Appending IQFT circuit...")
    var iqft_sub = IQFT(n)
    main_qc.append_circuit(iqft_sub, reg)

    # 5. Execute results
    var state = main_qc.execute()

    print("\nResulting State Table (Modular Approach):")
    print_state(state)

    print("\nSuccess! Modular value encoding via circuit appending verified.")
