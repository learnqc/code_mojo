from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.core.executors import execute
from butterfly.utils.context import ExecContext, ExecutionStrategy

fn test_classical_op(mut state: QuantumState, targets: List[Int]) raises:
    print("✓ Classical transformation executed in SIMD mode!")

var circuit = QuantumCircuit(2)

# Add a Hadamard gate
circuit.h(0)

# Add a classical operation
var targets = List[Int]()
targets.append(0)
circuit.add_classical("test_op", targets, test_classical_op)

# Add another gate
circuit.h(1)

print("Circuit has", len(circuit.transformations), "transformations")

var state = QuantumState(2)
var ctx = ExecContext()
ctx.execution_strategy = ExecutionStrategy.SIMD

execute(state, circuit, ctx)

print("Test completed successfully")
