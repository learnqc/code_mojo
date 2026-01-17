from butterfly.core.quantum_circuit import QuantumCircuit, QuantumRegister
from butterfly.algos.grover import grover_circuit
from butterfly.core.executors import execute
from butterfly.core.state import QuantumState
from butterfly.utils.visualization import animate_execution_table


fn main() raises:
    test_animate_grover()

fn test_animate_grover() raises:
    var use_shortcut = False
    print("Animating Grover (use_shortcut=" + String(use_shortcut) + ")...")
    alias n = 3
    var items = List[Int](3)

    var q = QuantumRegister("q", n)
    var qc = QuantumCircuit(q)
    for i in range(n):
        qc.h(i)

    var iterations = 2
    _ = qc.append_circuit(
        grover_circuit(items, n, iterations, use_shortcut=use_shortcut),
        q,
    )

    var state = QuantumState(n)

    # Animate as a table; small delay to auto-advance so CI/local runs don't block.
    animate_execution_table(
        qc,
        state,
        short=True,
        use_color=True,
        show_step_label=True,
        delay_s=0.25,
        step_on_input=False,
        redraw_in_place=True,
    )

    print("Grover animation complete.")
