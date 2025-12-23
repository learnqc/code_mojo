from butterfly.core.circuit import QuantumCircuit, QuantumRegister
from butterfly.algos.grover import grover_circuit
from testing import assert_almost_equal
from butterfly.core.types import pi
import math


fn main() raises:
    test_grover_gates()
    test_grover_shortcut()
    print("Grover's algorithm tests passed!")


fn test_grover_gates() raises:
    print("Testing Grover with GATES (use_shortcut=False)...")
    var n = 3
    var items = List[Int]()
    items.append(7)  # Target |111>

    var qc = QuantumCircuit(n)
    _ = qc.add_register("q", n)
    for i in range(n):
        qc.h(i)

    var iterations = 2
    qc.append_circuit(
        grover_circuit(items, n, iterations, use_shortcut=False),
        qc.registers[0].copy(),
    )

    qc.execute()

    var prob = qc.get_amplitude(7).re ** 2 + qc.get_amplitude(7).im ** 2
    print("Probability |111>:", prob)
    if prob < 0.9:
        raise Error("Low probability")


fn test_grover_shortcut() raises:
    print("\nTesting Grover with SHORTCUT (use_shortcut=True)...")
    var n = 3
    var items = List[Int]()
    items.append(5)  # Target |101>

    var qc = QuantumCircuit(n)
    _ = qc.add_register("q", n)
    for i in range(n):
        qc.h(i)

    var iterations = 2
    qc.append_circuit(
        grover_circuit(items, n, iterations, use_shortcut=True),
        qc.registers[0].copy(),
    )

    qc.execute()

    var prob = qc.get_amplitude(5).re ** 2 + qc.get_amplitude(5).im ** 2
    print("Probability |101>:", prob)
    if prob < 0.9:
        raise Error("Low probability")
