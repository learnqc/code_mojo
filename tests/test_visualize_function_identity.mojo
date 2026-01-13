from collections import List

from butterfly.algos.function_encoding import build_polynomial_circuit
from butterfly.core.executors import execute_scalar
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.utils.visualization import print_state_grid_colored_cells, animate_execution


fn main() raises:
    alias key_size = 3
    alias value_size = 3
    var terms = List[Tuple[FloatType, List[Int]]]()
    for i in range(key_size):
        var coeff = FloatType(1 << i)
        terms.append((coeff, List[Int](i)))

    var circuit = build_polynomial_circuit(key_size, value_size, terms)
    var col_bits = value_size

    # var state = QuantumState(key_size + value_size)
    # execute_scalar(state, circuit)

    # print_state_grid_colored_cells(
    #     state,
    #     col_bits,
    #     use_log=True,
    #     origin_bottom=False,
    #     show_bin_labels=True,
    # )

    state = QuantumState(key_size + value_size)
    animate_execution(
        circuit,
        state,
        col_bits=col_bits,
        use_log=True,
        show_bin_labels=True,
        origin_bottom=True,
        use_bg=True,
        show_chars=False,
        redraw_in_place=True,
        delay_s=0.25,
        step_on_input=False
    )
