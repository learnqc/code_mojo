"""
External benchmark helpers for comparing Mojo against other libraries.

Provides reusable functions for benchmarking:
- Qiskit (quantum circuit simulation)
- FFTW (Fast Fourier Transform)
- NumPy (general numerical operations)

All functions return execution time in milliseconds.
"""

import time
import numpy as np


# ============================================================================
# Qiskit Helpers
# ============================================================================

def benchmark_qiskit_circuit(circuit_builder, *args, iters: int = 5, transpile_mode: str = "cached") -> float:
    """Generic Qiskit circuit benchmark.
    
    Args:
        circuit_builder: Function that builds a Qiskit QuantumCircuit
        *args: Arguments to pass to circuit_builder
        iters: Number of iterations to average
        transpile_mode: "none" (raw execution), "cached" (transpile once), "in_loop" (include in timing), "slow" (Python Statevector)
        
    Returns:
        Average execution time in milliseconds
    """
    from qiskit_aer import Aer
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    
    backend = Aer.get_backend('statevector_simulator')
    qc = circuit_builder(*args)
    
    if transpile_mode == "slow":
        # Use Python Statevector simulator (the 68s baseline)
        # We must transpile/decompose for it to work if it has high-level gates,
        # but the simulation itself is slow.
        qc_t = transpile(qc, optimization_level=0)
        
        # Warmup
        _ = Statevector.from_instruction(qc_t).data
        
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            sv = Statevector.from_instruction(qc_t)
            _ = sv.data # Force evaluation
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            
    elif transpile_mode == "none":
        # Raw execution - no transpile() call at all
        _ = backend.run(qc).result()  # Warmup
        
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            job = backend.run(qc)
            result = job.result()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            
    elif transpile_mode == "cached":
        # Transpile once outside the loop
        qc_transpiled = transpile(qc, backend, optimization_level=3)
        _ = backend.run(qc_transpiled).result()  # Warmup
        
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            job = backend.run(qc_transpiled)
            result = job.result()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            
    elif transpile_mode == "in_loop":
        # Warmup (including transpile)
        qc_warm = transpile(qc, backend, optimization_level=3)
        _ = backend.run(qc_warm).result()
        
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            qc_t = transpile(qc, backend, optimization_level=3)
            job = backend.run(qc_t)
            result = job.result()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    else:
        raise ValueError(f"Unknown transpile_mode: {transpile_mode}")
    
    return sum(times) / len(times)


def build_qiskit_value_encoding(n: int, value: float):
    """Build value encoding circuit in Qiskit."""
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(n)
    max_val = 2 ** n
    normalized = value % max_val
    
    # Hadamard on all qubits
    for q in range(n):
        qc.h(q)
    
    # Phase rotations
    for q in range(n):
        angle = 2.0 * np.pi * normalized * (2 ** (n - 1 - q)) / max_val
        qc.p(angle, q)
    
    # Inverse QFT
    for j_inv in range(n - 1, -1, -1):
        q_target = n - 1 - j_inv
        qc.h(q_target)
        
        if j_inv > 0:
            for m in range(j_inv):
                q_control = n - 1 - m
                theta = -np.pi / (2 ** (j_inv - m))
                qc.cp(theta, q_target, q_control)
    
    return qc


def build_qiskit_prep(n: int, value: float):
    """Build only the preparation stage in Qiskit."""
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(n)
    
    # Hadamard on all qubits
    for i in range(n):
        qc.h(i)
    
    # Phase rotations
    for j in range(n):
        qc.p(2 * np.pi / 2 ** (j + 1) * value, j)
        
    return qc


def benchmark_qiskit_prep(n: int, value: float, iters: int = 5, mode: str = "cached") -> float:
    """Benchmark Qiskit prep stage."""
    return benchmark_qiskit_circuit(build_qiskit_prep, n, value, iters=iters, transpile_mode=mode)


def benchmark_qiskit_value_encoding(n: int, value: float, iters: int = 5, mode: str = "cached") -> float:
    """Benchmark Qiskit value encoding."""
    return benchmark_qiskit_circuit(build_qiskit_value_encoding, n, value, iters=iters, transpile_mode=mode)


def benchmark_qiskit_value_encoding_str(n: int, value: float, iters: int = 5) -> str:
    """Benchmark Qiskit value encoding, return as string for Mojo compatibility."""
    result = benchmark_qiskit_circuit(build_qiskit_value_encoding, n, value, iters=iters)
    return str(result)


def get_qiskit_statevector_data(n: int, value: float):
    """Run Qiskit value encoding and return raw statevector data (real, imag lists)."""
    from qiskit_aer import Aer
    from qiskit import transpile
    
    qc = build_qiskit_value_encoding(n, value)
    backend = Aer.get_backend('statevector_simulator')
    qc_transpiled = transpile(qc, backend)
    
    result = backend.run(qc_transpiled).result()
    sv = result.get_statevector()
    
    # Return as primitive lists for Mojo interop
    return sv.real.tolist(), sv.imag.tolist()


# ============================================================================
# FFTW Helpers
# ============================================================================

def benchmark_fftw(n: int, iters: int = 5) -> float:
    """Benchmark FFTW complex-to-complex FFT.
    
    Args:
        n: FFT size (power of 2)
        iters: Number of iterations to average
        
    Returns:
        Average execution time in milliseconds
    """
    try:
        import pyfftw
    except ImportError:
        print("Warning: pyfftw not installed, returning 0")
        return 0.0
    
    # Create input array
    size = 2 ** n
    a = pyfftw.empty_aligned(size, dtype='complex128')
    b = pyfftw.empty_aligned(size, dtype='complex128')
    
    # Initialize with random data
    a[:] = np.random.random(size) + 1j * np.random.random(size)
    
    # Create FFTW plan
    fft_object = pyfftw.FFTW(a, b, direction='FFTW_FORWARD')
    
    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fft_object()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return sum(times) / len(times)


# ============================================================================
# NumPy Helpers
# ============================================================================

def benchmark_numpy_fft(n: int, iters: int = 5) -> float:
    """Benchmark NumPy FFT.
    
    Args:
        n: FFT size (power of 2)
        iters: Number of iterations to average
        
    Returns:
        Average execution time in milliseconds
    """
    size = 2 ** n
    data = np.random.random(size) + 1j * np.random.random(size)
    
    # Warm up
    _ = np.fft.fft(data)
    
    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = np.fft.fft(data)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return sum(times) / len(times)


# ============================================================================
# Generic Benchmark Helper
# ============================================================================

def benchmark_python_function(func, *args, iters: int = 5, **kwargs) -> float:
    """Generic Python function benchmark.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments to func
        iters: Number of iterations to average
        **kwargs: Keyword arguments to func
        
    Returns:
        Average execution time in milliseconds
    """
    # Warm up
    _ = func(*args, **kwargs)
    
    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return sum(times) / len(times)


# ============================================================================
# Test/Demo
# ============================================================================

if __name__ == "__main__":
    print("Testing external benchmark helpers...")
    print()
    
    # Test Qiskit
    print("Qiskit value encoding:")
    for n in [10, 12]:
        time_ms = benchmark_qiskit_value_encoding(n, 42.0, iters=3)
        print(f"  n={n}: {time_ms:.2f} ms")
    print()
    
    # Test NumPy FFT
    print("NumPy FFT:")
    for n in [10, 15, 20]:
        time_ms = benchmark_numpy_fft(n, iters=5)
        print(f"  n={n} (size={2**n}): {time_ms:.2f} ms")
    print()
    
    # Test FFTW (if available)
    print("FFTW:")
    try:
        for n in [10, 15, 20]:
            time_ms = benchmark_fftw(n, iters=5)
            print(f"  n={n} (size={2**n}): {time_ms:.2f} ms")
    except Exception as e:
        print(f"  FFTW not available: {e}")


def benchmark_single_gate_qiskit(n: int, gate: str, target: int, angle: float = 0.0, iters: int = 10) -> float:
    """
    Benchmark a single gate in Qiskit.
    
    Args:
        n: Number of qubits
        gate: Gate name ('H', 'X', 'Z', 'P', 'RZ')
        target: Target qubit index
        angle: Rotation angle (for P and RZ gates)
        iters: Number of iterations
        
    Returns:
        Average execution time in milliseconds
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import Aer
    from qiskit import transpile
    import time
    
    # Create circuit
    qc = QuantumCircuit(n)
    
    # Add gate
    if gate == 'H':
        qc.h(target)
    elif gate == 'X':
        qc.x(target)
    elif gate == 'Z':
        qc.z(target)
    elif gate == 'Y':
        qc.y(target)
    elif gate == 'P':
        qc.p(angle, target)
    elif gate == 'RX':
        qc.rx(angle, target)
    elif gate == 'RY':
        qc.ry(angle, target)
    elif gate == 'RZ':
        qc.rz(angle, target)
    # Transpile
    backend = Aer.get_backend('statevector_simulator')
    qc_transpiled = transpile(qc, backend, optimization_level=3)
    
    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = backend.run(qc_transpiled).result()
        statevector = result.get_statevector()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # Convert to ms
    
    return sum(times) / len(times)


def benchmark_bit_reverse_numpy(n: int, iters: int = 10) -> float:
    """
    Benchmark bit reversal using NumPy.
    
    Args:
        n: Number of qubits (state size = 2^n)
        iters: Number of iterations
        
    Returns:
        Average execution time in milliseconds
    """
    import numpy as np
    import time
    
    size = 2 ** n
    
    # Create a state vector
    state = np.random.rand(size) + 1j * np.random.rand(size)
    
    # Precompute bit-reversed indices
    def bit_reverse_index(i, num_bits):
        result = 0
        for _ in range(num_bits):
            result = (result << 1) | (i & 1)
            i >>= 1
        return result
    
    indices = np.array([bit_reverse_index(i, n) for i in range(size)])
    
    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        reversed_state = state[indices]
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # Convert to ms
    
    return sum(times) / len(times)
