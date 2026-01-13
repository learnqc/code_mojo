from collections import List

from butterfly.algos.function_encoding import build_polynomial_circuit
from butterfly.core.executors import execute_scalar
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.utils.visualization import (
    print_state,
    print_state_grid_colored_cells,
)


fn main() raises:
    alias key_size = 3
    alias value_size = 3
    var terms = List[Tuple[FloatType, List[Int]]]()
    terms.append((FloatType(1.0), List[Int](0)))
    terms.append((FloatType(1.0), List[Int](1)))

    var circuit = build_polynomial_circuit(key_size, value_size, terms)
    var state = QuantumState(key_size + value_size)
    execute_scalar(state, circuit)

    print("\n[State Table]")
    print_state(state, short=False)
    print("\n[Grid View]")
    var col_bits = value_size
    print_state_grid_colored_cells(state, col_bits, use_log=True, origin_bottom=True, show_bin_labels=True)
