from butterfly.core.quantum_circuit import QuantumCircuit, QuantumRegister
from butterfly.algos.grover import grover_circuit
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType, pi
from butterfly.utils.visualization import (
    FrameSource,
    animate_frame_source_pair,
)

from math import sqrt


fn main() raises:
    test_animate_dual_grover()

fn test_animate_dual_grover() raises:
    var use_shortcut = False
    print("Animating dual Grover (use_shortcut=" + String(use_shortcut) + ")...")
    alias n = 8
    var items = List[Int](3)

    var q = QuantumRegister("q", n)
    var qc = QuantumCircuit(q)
    for i in range(n):
        qc.h(i)

    var m_items = len(items)
    if m_items <= 0:
        m_items = 1
    var n_states = FloatType(1 << n)
    var iters_f = FloatType(pi) / 4.0 * sqrt(n_states / FloatType(m_items))
    var iterations = Int(round(iters_f))
    _ = qc.append_circuit(
        grover_circuit(items, n, iterations, use_shortcut=use_shortcut),
        q,
    )

    var left_state = QuantumState(n)
    var right_state = QuantumState(n)

    var left_source = FrameSource.grid(
        qc.copy(),
        left_state,
        col_bits=4,
        use_log=True,
        show_bin_labels=True,
        use_bg=True,
        show_chars=False,
        show_step_label=True,
    )
    var right_source = FrameSource.table(
        qc.copy(),
        right_state,
        show_step_label=True,
        row_separators=True,
        max_rows=16,
    )

    animate_frame_source_pair(
        left_source,
        right_source,
        gap=6,
        delay_s=0.033,
        step_on_input=False,
    )

    print("Dual Grover animation complete.")
