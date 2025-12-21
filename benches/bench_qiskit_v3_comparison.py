#!/usr/bin/env python3
"""
Qiskit benchmark for value encoding to compare against v3 executor.
Tests N=3 to N=29 with adaptive iteration counts.
"""
import time
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT

def encode_value_qiskit(n: int, value: float) -> QuantumCircuit:
    """Create value encoding circuit in Qiskit (matches Mojo implementation)."""
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)
    
    # Hadamard on all qubits
    for i in range(n):
        qc.h(i)
    
    # IQFT
    # for j in range(n):
    #     target = n - 1 - j
    #     # Controlled phase rotations
    #     for k in range(j):
    #         control = n - 1 - k
    #         angle = -np.pi / (2 ** (j - k))
    #         qc.cp(angle, control, target)
    #     # Hadamard
    #     qc.h(target)
    
    # # Bit reversal (swap pairs)
    # for i in range(n // 2):
    #     qc.swap(i, n - 1 - i)
    iqft = QFT(n, inverse=True, do_swaps=True)
    qc.append(iqft, range(n))
    
    return qc


def benchmark_qiskit(n: int, value: float, iterations: int) -> float:
    """Benchmark Qiskit for given problem size."""
    times = []
    backend = Aer.get_backend('statevector_simulator')
    # Warmup
    qc = encode_value_qiskit(n, value)
    qc = transpile(qc, backend)
    job = backend.run(qc)
    result = job.result()
    
    for _ in range(iterations):
        t0 = time.perf_counter()

        job = backend.run(qc)
        result = job.result()
        # state = Statevector(qc)
        # _ = state.data  # Force evaluation
        t1 = time.perf_counter()
        
        times.append(t1 - t0)
    
    return np.mean(times)


def get_iterations(n: int) -> int:
    """Adaptive iteration count based on problem size."""
    if n <= 15:
        return 10
    elif n <= 20:
        return 5
    elif n <= 25:
        return 3
    else:
        return 2


def main():
    value = 0.5
    
    print("=" * 80)
    print("Qiskit Benchmark - Value Encoding")
    print("=" * 80)
    print()
    print(f"{'N':<5} | {'Iters':<6} | {'Qiskit Time (s)':<20}")
    print("-" * 80)
    
    results = []
    
    # Benchmark from N=3 to N=29
    for n in range(3, 30):
        iters = get_iterations(n)
        
        try:
            qiskit_time = benchmark_qiskit(n, value, iters)
            results.append((n, iters, qiskit_time))
            print(f"{n:<5} | {iters:<6} | {qiskit_time:<20.6f}")
        except Exception as e:
            print(f"{n:<5} | {iters:<6} | ERROR: {str(e)}")
            break
    
    print("=" * 80)
    print()
    print("Results saved. Compare with Mojo v3 benchmark:")
    print("  mojo run benches/bench_v3_vs_qiskit.mojo")
    
    # Save results to file
    with open("benches/qiskit_results.txt", "w") as f:
        f.write("N,Iterations,Qiskit_Time\n")
        for n, iters, time_val in results:
            f.write(f"{n},{iters},{time_val}\n")
    
    print("\nResults saved to: benches/qiskit_results.txt")


if __name__ == "__main__":
    main()
