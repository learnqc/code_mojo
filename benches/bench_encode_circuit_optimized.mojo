from time import perf_counter_ns
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.core.gates import H, P
from butterfly.core.types import FloatType
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.algos.qft import iqft
from math import pi


fn build_encode_circuit(mut circuit: QuantumCircuit, n: Int, value: Float64):
    """Builds the value encoding circuit using Library IQFT (Reference)."""
    # 1. Hadamard Layer
    for j in range(n):
        circuit.h(j)

    # 2. Phase Encoding Layer
    for j in range(n):
        circuit.p(j, 2 * pi / 2 ** (n - j) * value)

    # 3. Inverse QFT (Library Call)
    # Matches value_encoding.mojo: targets = [n-1, ..., 0]
    var targets = List[Int]()
    for j in range(n):
        targets.append(n - 1 - j)
    iqft(circuit, targets, do_swap=True)


fn benchmark_v3_simd(n: Int, value: Float64, iterations: Int) -> Float64:
    """Benchmark V3 executor for given problem size."""
    # Build circuit once
    var circuit = QuantumCircuit(n)
    build_encode_circuit(circuit, n, value)
    var transformations = circuit.transformations.copy()

    # Warmup
    var warmup_state = QuantumState(n)
    if n == 3:
        execute_fused_v3[3](warmup_state, transformations, block_log=20)
    elif n == 4:
        execute_fused_v3[4](warmup_state, transformations, block_log=20)
    elif n == 5:
        execute_fused_v3[5](warmup_state, transformations, block_log=20)
    elif n == 6:
        execute_fused_v3[6](warmup_state, transformations, block_log=20)
    elif n == 7:
        execute_fused_v3[7](warmup_state, transformations, block_log=20)
    elif n == 8:
        execute_fused_v3[8](warmup_state, transformations, block_log=20)
    elif n == 9:
        execute_fused_v3[9](warmup_state, transformations, block_log=20)
    elif n == 10:
        execute_fused_v3[10](warmup_state, transformations, block_log=20)
    elif n == 11:
        execute_fused_v3[11](warmup_state, transformations, block_log=20)
    elif n == 12:
        execute_fused_v3[12](warmup_state, transformations, block_log=20)
    elif n == 13:
        execute_fused_v3[13](warmup_state, transformations, block_log=20)
    elif n == 14:
        execute_fused_v3[14](warmup_state, transformations, block_log=20)
    elif n == 15:
        execute_fused_v3[15](warmup_state, transformations, block_log=20)
    elif n == 16:
        execute_fused_v3[16](warmup_state, transformations, block_log=20)
    elif n == 17:
        execute_fused_v3[17](warmup_state, transformations, block_log=20)
    elif n == 18:
        execute_fused_v3[18](warmup_state, transformations, block_log=20)
    elif n == 19:
        execute_fused_v3[19](warmup_state, transformations, block_log=20)
    elif n == 20:
        execute_fused_v3[20](warmup_state, transformations, block_log=20)
    elif n == 21:
        execute_fused_v3[21](warmup_state, transformations, block_log=20)
    elif n == 22:
        execute_fused_v3[22](warmup_state, transformations, block_log=20)
    elif n == 23:
        execute_fused_v3[23](warmup_state, transformations, block_log=20)
    elif n == 24:
        execute_fused_v3[24](warmup_state, transformations, block_log=20)
    elif n == 25:
        execute_fused_v3[25](warmup_state, transformations, block_log=20)
    elif n == 26:
        execute_fused_v3[26](warmup_state, transformations, block_log=20)
    elif n == 27:
        execute_fused_v3[27](warmup_state, transformations, block_log=20)
    elif n == 28:
        execute_fused_v3[28](warmup_state, transformations, block_log=20)
    elif n == 29:
        execute_fused_v3[29](warmup_state, transformations, block_log=20)

    var total_time: Float64 = 0.0
    for _ in range(iterations):
        var state = QuantumState(n)

        var t0 = perf_counter_ns()
        if n == 3:
            execute_fused_v3[3](state, transformations, block_log=20)
        elif n == 4:
            execute_fused_v3[4](state, transformations, block_log=20)
        elif n == 5:
            execute_fused_v3[5](state, transformations, block_log=20)
        elif n == 6:
            execute_fused_v3[6](state, transformations, block_log=20)
        elif n == 7:
            execute_fused_v3[7](state, transformations, block_log=20)
        elif n == 8:
            execute_fused_v3[8](state, transformations, block_log=20)
        elif n == 9:
            execute_fused_v3[9](state, transformations, block_log=20)
        elif n == 10:
            execute_fused_v3[10](state, transformations, block_log=20)
        elif n == 11:
            execute_fused_v3[11](state, transformations, block_log=20)
        elif n == 12:
            execute_fused_v3[12](state, transformations, block_log=20)
        elif n == 13:
            execute_fused_v3[13](state, transformations, block_log=20)
        elif n == 14:
            execute_fused_v3[14](state, transformations, block_log=20)
        elif n == 15:
            execute_fused_v3[15](state, transformations, block_log=20)
        elif n == 16:
            execute_fused_v3[16](state, transformations, block_log=20)
        elif n == 17:
            execute_fused_v3[17](state, transformations, block_log=20)
        elif n == 18:
            execute_fused_v3[18](state, transformations, block_log=20)
        elif n == 19:
            execute_fused_v3[19](state, transformations, block_log=20)
        elif n == 20:
            execute_fused_v3[20](state, transformations, block_log=20)
        elif n == 21:
            execute_fused_v3[21](state, transformations, block_log=20)
        elif n == 22:
            execute_fused_v3[22](state, transformations, block_log=20)
        elif n == 23:
            execute_fused_v3[23](state, transformations, block_log=20)
        elif n == 24:
            execute_fused_v3[24](state, transformations, block_log=20)
        elif n == 25:
            execute_fused_v3[25](state, transformations, block_log=20)
        elif n == 26:
            execute_fused_v3[26](state, transformations, block_log=20)
        elif n == 27:
            execute_fused_v3[27](state, transformations, block_log=20)
        elif n == 28:
            execute_fused_v3[28](state, transformations, block_log=20)
        elif n == 29:
            execute_fused_v3[29](state, transformations, block_log=20)

        var t1 = perf_counter_ns()
        total_time += Float64(t1 - t0)

    return total_time / Float64(iterations) / 1e9


fn get_iterations(n: Int) -> Int:
    if n <= 15:
        return 10
    if n <= 20:
        return 5
    if n <= 25:
        return 3
    return 2


fn verify_state(n: Int, value: Float64):
    var circuit = QuantumCircuit(n)
    build_encode_circuit(circuit, n, value)
    var transformations = circuit.transformations.copy()

    var state = QuantumState(n)
    if n == 3:
        execute_fused_v3[3](state, transformations, block_log=20)
    elif n == 4:
        execute_fused_v3[4](state, transformations, block_log=20)
    elif n == 5:
        execute_fused_v3[5](state, transformations, block_log=20)

    print("VERIFICATION_START N=", n)
    for i in range(state.size()):
        print(state.re[i], state.im[i])
    print("VERIFICATION_END")


fn main():
    alias value = 4.7

    # Verify small states (N=3 to 5) - Disabled for pure benchmarking
    # verify_state(3, value)
    # verify_state(4, value)
    # verify_state(5, value)

    print("N,Mojo_Time_s")

    # N=26 to 29
    for n in range(26, 30):
        var iters = get_iterations(n)
        try:
            var t = benchmark_v3_simd(n, value, iters)
            print(n, ",", t)
        except:
            print(n, ",ERROR")
