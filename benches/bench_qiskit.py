import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import sys

def create_value_encoding_circuit(n, v):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
    for j in range(n):
        qc.p(2 * np.pi / 2**(j+1) * v, j)
    # IQFT part (simplified for benchmark)
    # In my Mojo code: iqft_circuit(circuit, [n - 1 - j for j in range(n)], do_swap=True)
    # Qiskit's QFT is available in library
    from qiskit.circuit.library import QFT
    iqft = QFT(n, inverse=True, do_swaps=True)
    qc.append(iqft, range(n))
    return qc

def bench_qiskit(n, v, iters=5):
    backend = Aer.get_backend('statevector_simulator')
    # Set precision to single if possible? Aer defaults to double.
    # We use float64 in Mojo (Amplitude is Complex64 i.e. 2x float64).
    
    qc = create_value_encoding_circuit(n, v)
    qc = transpile(qc, backend)

    # Warmup
    job = backend.run(qc)
    result = job.result()
    
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        job = backend.run(qc)
        result = job.result()
        end = time.perf_counter()
        times.append((end - start) * 1000) # ms
    
    print(f"Qiskit Aer (N={n}): Mean {np.mean(times):.3f} ms, Std {np.std(times):.3f} ms")
    return np.mean(times)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    bench_qiskit(n, 1.23)
