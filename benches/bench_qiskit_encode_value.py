"""
Benchmark Qiskit value encoding for comparison with Mojo.
"""
import time
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

def encode_value_qiskit(n, value):
    """Value encoding using Qiskit circuit."""
    qc = QuantumCircuit(n)
    
    # Apply Hadamard to all qubits
    for j in range(n):
        qc.h(j)
    
    # Apply phase rotations
    for j in range(n):
        angle = 2 * np.pi / (2 ** (j + 1)) * value
        qc.p(angle, j)
    
    # Apply inverse QFT
    # Simplified IQFT for benchmarking
    for j in range(n-1, -1, -1):
        qc.h(j)
        for k in range(j):
            qc.cp(-np.pi / (2 ** (j - k)), k, j)
    
    # Swap qubits (bit reversal)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    
    return qc

def benchmark_qiskit(n=25, value=1.23, iterations=5):
    """Benchmark Qiskit value encoding."""
    print(f"Benchmarking Qiskit Value Encoding (N={n})...")
    
    # Create circuit
    qc = encode_value_qiskit(n, value)
    print(f"Total gates: {qc.size()}")
    
    # Create simulator
    simulator = AerSimulator(method='statevector')
    
    # Warmup
    for _ in range(2):
        result = simulator.run(qc).result()
        statevector = result.get_statevector()
    
    # Benchmark
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        result = simulator.run(qc).result()
        statevector = result.get_statevector()
        end = time.perf_counter()
        elapsed = (end - start) * 1000  # Convert to ms
        times.append(elapsed)
        print(f"Iteration {i+1}: {elapsed:.2f} ms")
    
    mean_time = np.mean(times)
    print(f"\nMean time: {mean_time:.2f} ms")
    print(f"Std dev: {np.std(times):.2f} ms")
    
    return mean_time

if __name__ == "__main__":
    qiskit_time = benchmark_qiskit(n=25, value=1.23, iterations=5)
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Qiskit Aer (N=25): {qiskit_time:.2f} ms")
    print("\nRun bench_encode_circuit_super_fast.mojo to compare with Mojo!")
