from benchmark import keep, run, Unit
from math import pi
from butterfly.core.types import FloatType
from butterfly.core.gates import H, P
from butterfly.core.state import bit_reverse_state, QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.circuit_simd import QuantumCircuitSIMD

from butterfly.algos.value_encoding import encode_value_circuit
from butterfly.core.types import Amplitude


fn bench_encode_circuit_super_fast[n: Int](value: Float64) raises:
    """
    Benchmark value encoding circuit using execute_simd().
    Compares standard execute() vs execute_simd().
    """
    print(
        "--------------------------------------------------------------------------------"
    )
    print("Value Encoding Circuit Benchmark (N=" + String(n) + ")")

    # 1. Performance Measurement
    # We use separate instances to avoid state accumulation if possible,
    # or just acknowledge that benchmark runs on a non-zero state after iteration 1.
    # Mathematically the work is the same.

    var circuit = QuantumCircuit(n)
    encode_value_circuit(circuit, n, value)

    var circuit_simd = QuantumCircuitSIMD[n]()
    for i in range(len(circuit.transformations)):
        circuit_simd.transformations.append(circuit.transformations[i].copy())

    print("Total Transformations: " + String(circuit.num_transformations()))

    @parameter
    fn wrapper_circuit():
        var c = circuit.copy()
        c.execute_simd()
        keep(c.state.re.unsafe_ptr())

    @parameter
    fn wrapper_circuit_v2():
        var c = circuit.copy()
        c.execute_simd_v2()
        keep(c.state.re.unsafe_ptr())

    @parameter
    fn wrapper_simd_v1():
        var c = circuit_simd.copy()
        c.execute_simd()
        keep(c.state.re.unsafe_ptr())

    @parameter
    fn wrapper_simd_v2():
        var c = circuit_simd.copy()
        c.execute_simd_v2()
        keep(c.state.re.unsafe_ptr())

    @parameter
    fn wrapper_simd_run():
        var c = circuit_simd.copy()
        c.run()
        keep(c.state.re.unsafe_ptr())

    # NOTE: wrapper including .copy() measures execution + copy.
    # If copy is fast, it's fine. If we want pure execution, we accept in-place modification.
    # Standard butterfly benchmarks usually measure in-place to get peak throughput.

    @parameter
    fn wrapper_circuit_inplace():
        circuit.execute_simd()
        keep(circuit.state.re.unsafe_ptr())

    @parameter
    fn wrapper_circuit_v2_inplace():
        circuit.execute_simd_v2()
        keep(circuit.state.re.unsafe_ptr())

    @parameter
    fn wrapper_simd_v1_inplace():
        circuit_simd.execute_simd()
        keep(circuit_simd.state.re.unsafe_ptr())

    @parameter
    fn wrapper_simd_v2_inplace():
        circuit_simd.execute_simd_v2()
        keep(circuit_simd.state.re.unsafe_ptr())

    @parameter
    fn wrapper_simd_run_inplace():
        circuit_simd.run()
        keep(circuit_simd.state.re.unsafe_ptr())

    print("\n[In-place Execution - modifies state every iteration]")
    print("circuit.execute_simd():")
    run[wrapper_circuit_inplace](2, 5).print(Unit.ms)

    print("circuit.execute_simd_v2():")
    run[wrapper_circuit_v2_inplace](2, 5).print(Unit.ms)

    print("circuit_simd.execute_simd():")
    run[wrapper_simd_v1_inplace](2, 5).print(Unit.ms)

    print("circuit_simd.execute_simd_v2():")
    run[wrapper_simd_v2_inplace](2, 5).print(Unit.ms)

    print("circuit_simd.run() [direct v2]:")
    run[wrapper_simd_run_inplace](2, 5).print(Unit.ms)

    # 2. Correctness Verification (Starting from fresh |0> state)
    print("\nVerifying Correctness (Fresh State)...")

    var c_baseline = QuantumCircuit(n)
    encode_value_circuit(c_baseline, n, value)
    c_baseline.execute()

    # Check QuantumCircuit cases
    var c_v1 = QuantumCircuit(n)
    encode_value_circuit(c_v1, n, value)
    c_v1.execute_simd()
    verify("circuit.execute_simd()", c_baseline.state, c_v1.state)

    var c_v2 = QuantumCircuit(n)
    encode_value_circuit(c_v2, n, value)
    c_v2.execute_simd_v2()
    verify("circuit.execute_simd_v2()", c_baseline.state, c_v2.state)

    # Check QuantumCircuitSIMD cases
    var cs_v1 = QuantumCircuitSIMD[n]()
    # Copy transformations from the same builder logic
    var circuit_builder = QuantumCircuit(n)
    encode_value_circuit(circuit_builder, n, value)
    for i in range(len(circuit_builder.transformations)):
        cs_v1.transformations.append(circuit_builder.transformations[i].copy())
    cs_v1.execute_simd()
    verify("circuit_simd.execute_simd()", c_baseline.state, cs_v1.state)

    var cs_v2 = QuantumCircuitSIMD[n]()
    for i in range(len(circuit_builder.transformations)):
        cs_v2.transformations.append(circuit_builder.transformations[i].copy())
    cs_v2.execute_simd_v2()
    verify("circuit_simd.execute_simd_v2()", c_baseline.state, cs_v2.state)

    var cs_run = QuantumCircuitSIMD[n]()
    for i in range(len(circuit_builder.transformations)):
        cs_run.transformations.append(circuit_builder.transformations[i].copy())
    cs_run.run()
    verify("circuit_simd.run()", c_baseline.state, cs_run.state)


fn verify(name: String, baseline: QuantumState, test: QuantumState):
    var errs = 0
    var n = len(baseline)
    for i in range(min(1000, n)):
        if (
            abs(baseline.re[i] - test.re[i]) > 1e-9
            or abs(baseline.im[i] - test.im[i]) > 1e-9
        ):
            errs += 1

    if errs == 0:
        print("  Success: " + name + " matches baseline.")
    else:
        print("  FAILURE: " + name + " mismatch! Errors: " + String(errs))
        # Print first few mismatching samples
        print("    Samples:")
        for i in range(min(5, n)):
            print(
                "    ["
                + String(i)
                + "] Expected: "
                + String(baseline.re[i])
                + " + "
                + String(baseline.im[i])
                + "i"
            )
            print(
                "    ["
                + String(i)
                + "] Actual:   "
                + String(test.re[i])
                + " + "
                + String(test.im[i])
                + "i"
            )


fn main() raises:
    bench_encode_circuit_super_fast[25](1.23)
