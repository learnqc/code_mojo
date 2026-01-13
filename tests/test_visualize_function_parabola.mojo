from collections import List

from butterfly.algos.function_encoding import build_polynomial_circuit
from butterfly.core.executors import execute_scalar
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.utils.visualization import print_state_grid_colored_cells, animate_execution


fn main() raises:
    alias key_size = 2
    alias value_size = 4
    # var terms = build_parabola_terms(key_size, FloatType(2.0))
    var terms = List[Tuple[FloatType, List[Int]]]()
    terms.append((FloatType(4), List[Int](1)))
    terms.append((FloatType(4), List[Int](1, 0)))
    terms.append((FloatType(1), List[Int](0)))
    terms.append((FloatType(3), List[Int]()))      

    var circuit = build_polynomial_circuit(key_size, value_size, terms)

    # var state = QuantumState(key_size + value_size)
    # execute_scalar(state, circuit)
    # print_state_grid_colored_cells(
    #     state,
    #     key_size,
    #     use_log=True,
    #     origin_bottom=True,
    #     show_bin_labels=True,
    # )

    state = QuantumState(key_size + value_size)
    animate_execution(
        circuit,
        state,
        col_bits=key_size,
        use_log=True,
        show_bin_labels=True,
        origin_bottom=True,
        use_bg=True,
        show_chars=False,
        redraw_in_place=True,
        delay_s=0.25,
        step_on_input=False
    )
