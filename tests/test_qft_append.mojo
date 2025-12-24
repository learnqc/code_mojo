from butterfly.core.circuit import QuantumCircuit, Register, QFT, IQFT
from butterfly.core.types import pi
from butterfly.core.circuit import run_circuit
from butterfly.core.state import iqft
from butterfly.utils.visualization import print_state
from testing import assert_almost_equal


fn main() raises:
    print("--- Testing QFT Circuit Appending ---")

    alias n = 3
    var v = 2.0

    # 1. Create a "parent" circuit
    var main_qc = QuantumCircuit(n)
    var reg = main_qc.add_register("data", n)

    # Setup initial state
    print("Step 1: Preparing uniform superposition and encoding phases...")
    for i in range(n):
        main_qc.h(i)
    for j in range(n):
        main_qc.p(j, 2 * pi * v / 2 ** (n - j))

    # 2. Create a separate IQFT circuit
    print("Step 2: Creating standalone IQFT circuit...")
    var iqft_sub = IQFT(n, reversed=False, swap=True)

    # 3. Append IQFT to the main circuit
    print("Step 3: Appending IQFT circuit to main circuit...")
    main_qc.append_circuit(iqft_sub, reg)
    # main_qc.iqft(reg)

    # 4. Execute
    print("Step 4: Executing combined circuit...")
    var state = main_qc.execute()

    # 5. Verify results
    print("Resulting State Table (v=2.0 expected at index 2):")
    print_state(state)

    assert_almost_equal(state[2].re, 1.0)
    print("\nSuccess! QFT circuit appending verified.")
