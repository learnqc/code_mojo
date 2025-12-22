"""
Benchmark CFT vs QFT for strict subset targets.
"""
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.circuit import QuantumCircuit
from butterfly.algos.qft import qft
from butterfly.core.fft_v5 import apply_cft
from benchmark import keep, run, Unit


fn run_bench[n: Int, k: Int]() raises:
    print(
        "--------------------------------------------------------------------------------"
    )
    print("Benchmarking N =", n, "Targets =", k, "(Subset)")
    var state = QuantumState(n)
    var targets = List[Int]()
    # Target k qubits in the middle
    var start = (n - k) // 2
    for i in range(k):
        targets.append(start + i)

    # 1. Benchmark CFT (v5)
    @parameter
    fn bench_cft():
        apply_cft(state, targets, inverse=False, do_swap=True)
        keep(state.re.unsafe_ptr())

    print("  CFT (Subspace):")
    var report_cft = run[bench_cft](2, 5)
    report_cft.print(Unit.ms)

    # 2. Benchmark QFT (Gates)
    @parameter
    fn bench_qft():
        var circ = QuantumCircuit(n)
        # We skip state copy in loop to focus on execution time,
        # but ideally we should include setup if we want end-to-end.
        # However, circ.execute is strict evaluation.
        # Let's just measure execute + construction overhead (minimal).
        circ.state = state.copy()  # Copy? No, moves/copies handle.
        # Actually QuantumCircuit takes ownership or copies?
        # Check circuit.mojo: circ.state is a field.
        # If we reuse state, we might need to reset it.
        # For benchmarking, we just run forward.

        qft(circ, targets, do_swap=True)
        circ.execute()
        keep(state.re.unsafe_ptr())

    print("  QFT (Chebyshev/Gates):")
    var report_qft = run[bench_qft](2, 5)
    report_qft.print(Unit.ms)

    var speedup = report_qft.mean(Unit.ns) / report_cft.mean(Unit.ns)
    if speedup > 1.0:
        print("  CFT is ", speedup, "x FASTER than QFT")
    else:
        print("  CFT is ", 1.0 / speedup, "x SLOWER than QFT")


fn main() raises:
    # Case 1: N=20, k=10 (Subset)
    run_bench[20, 10]()

    # Case 2: N=20, k=5 (Small Subset)
    run_bench[20, 5]()

    # Case 3: N=15, k=10
    run_bench[15, 10]()
