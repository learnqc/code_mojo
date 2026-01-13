from butterfly.algos.value_encoding_circuit import encode_value_circuit
from butterfly.core.executors import execute_scalar
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.utils.visualization import print_state, print_state_grid_colored_cells


fn main() raises:
    alias n = 5
    var v = FloatType(4.7)

    var circuit = encode_value_circuit(n, v)
    var state = QuantumState(n)
    execute_scalar(state, circuit)

    # Match grid layout: row_size = 1 << col_bits.
    var col_bits = max(n - 3, 3)
    print("\n[State Table]")
    print_state(state, short=False)
    print("\n[Grid View]")
    print_state_grid_colored_cells(state, col_bits, use_log=True)
