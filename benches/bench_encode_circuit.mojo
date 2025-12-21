from benchmark import keep, run, Unit
from butterfly.core.circuit import QuantumCircuit
from butterfly.algos.value_encoding import encode_value_circuit
from butterfly.core.types import Amplitude
from butterfly.core.execute_simd_dispatch import execute_transformations_simd


fn bench_encode_circuit(n: Int, v: Float64) raises:
    var circuit = QuantumCircuit(n)
    encode_value_circuit(circuit, n, v)
    circuit.fuse()

    print(
        "--------------------------------------------------------------------------------"
    )
    print("Value Encoding Circuit (N=" + String(n) + ")")
    print("Total Transformations: " + String(circuit.num_transformations()))

    @parameter
    fn wrapper_simd():
        execute_transformations_simd[1 << 25](
            circuit.state, circuit.transformations
        )
        keep(circuit.state.re.unsafe_ptr())

    print("SIMD Execution (N=25):")
    var report3 = run[wrapper_simd](2, 5)
    report3.print(Unit.s)

    # @parameter
    # fn wrapper_std():
    #     circuit.execute()
    #     keep(circuit.state.re.unsafe_ptr())

    @parameter
    fn wrapper_opt():
        circuit.execute_optimized()
        keep(circuit.state.re.unsafe_ptr())

    # print("Standard Execution:")
    # var report = run[wrapper_std]()
    # report.print(Unit.ms)

    print("Optimized Execution (Greedy Fusion):")
    var report2 = run[wrapper_opt]()
    report2.print(Unit.ms)


fn main() raises:
    bench_encode_circuit(25, 1.23)
