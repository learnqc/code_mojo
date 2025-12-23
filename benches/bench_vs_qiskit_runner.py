
import subprocess
import sys
import time
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFT
from encode_value_qiskit import encode_value_qiskit_circuit

# def encode_value_qiskit(n: int, value: float) -> QuantumCircuit:
#     """Create value encoding circuit in Qiskit."""
#     qr = QuantumRegister(n)
#     qc = QuantumCircuit(qr)
    
#     # Hadamard on all qubits
#     for j in range(n):
#         qc.h(j)

#     for j in range(n):
#         qc.p(2 * np.pi / (2 ** (n - j)) * value, j)
    
#     # Manual IQFT (Identical to Mojo: Target 0 to n-1, H then CPs)
#     for j in range(n):
#         # Target goes 0, 1, ..., n-1
#         # This matches Mojo's execution order when targets=[n-1...0] and loop is reversed
        
#         qc.h(j)
        
#         # CPs with "previous" targets in the sequence (which are higher indices in targets list)
#         # Mojo: for k in reversed(range(j)). targets[k] are > targets[j] if targets=[n-1...0]??
#         # WAIT. targets = [n-1, n-2, ... 0].
#         # j (loop index) goes n-1 down to 0. targets[j] goes 0 to n-1.
#         # Let's say outer loop index is 'idx'. targets[idx] is t.
#         # Inner loop 'k' from idx-1 down to 0. targets[k] is > t.
#         # So we want CP(t, >t).
        
#         for k in range(j + 1, n):
#             # angle matches Mojo: -pi / 2**(k-j)? 
#             # Mojo: -pi / 2**(idx - k_idx).
#             # If idx > k_idx, then 2**positive.
#             # In my python: j < k. So 2**(k-j).
#             angle = -np.pi / (2 ** (k - j))
#             qc.cp(angle, k, j)
            
#     # Swaps
#     for i in range(n // 2):
#         qc.swap(i, n - 1 - i)
            


#     return qc

def benchmark_qiskit(n: int, value: float, iterations: int) -> float:
    times = []
    backend = Aer.get_backend('statevector_simulator')
    
    qc = encode_value_qiskit_circuit(n, value)
    qc = transpile(qc, backend)
    
    # Warmup
    backend.run(qc).result()
    
    for _ in range(iterations):
        t0 = time.perf_counter()
        backend.run(qc).result()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    return np.mean(times)

def get_iterations(n: int) -> int:
    if n <= 15: return 10
    if n <= 20: return 5
    if n <= 25: return 3
    return 2

def verify_state_correctness(n: int, value: float, mojo_re: list, mojo_im: list):
    """Verify Mojo state against Qiskit state."""
    print(f"  Verifying N={n}...", end=" ", flush=True)
    
    # Qiskit State
    qc = encode_value_qiskit_circuit(n, value)
    backend = Aer.get_backend('statevector_simulator')
    qc = transpile(qc, backend)
    result = backend.run(qc).result()
    qiskit_sv = result.get_statevector()
    
    # Mojo State
    mojo_sv = np.array(mojo_re) + 1j * np.array(mojo_im)
    
    # Compare (Fidelity or Allclose)
    # Compare (Fidelity or Allclose)
    # Check Fidelity
    from qiskit.quantum_info import state_fidelity
    norm = np.linalg.norm(mojo_sv)
    print(f"(Mojo Norm: {norm:.6f})", end=" ")
    
    try:
        fid = state_fidelity(qiskit_sv, mojo_sv, validate=False)
    except Exception as e:
        print(f"State Fidelity Error: {e}")
        fid = 0.0
    
    if fid > 0.99999 and n != 3:
        print(f"✅ PASS (Fidelity: {fid:.6f})")
    else:
        if n == 3:
             print(f"DEBUG N=3 (v={value}) MISMATCH CHECK:")
        print(f"❌ FAIL (Fidelity: {fid:.6f})")
        print("Mojo (Probabilities):")
        for k in range(len(mojo_sv)):
             prob = abs(mojo_sv[k])**2
             print(f"|{k:0{n}b}>: {prob:.4f}")
        print("Qiskit (Probabilities):")
        for k in range(len(qiskit_sv)):
             prob = abs(qiskit_sv[k])**2
             print(f"|{k:0{n}b}>: {prob:.4f}")

def run_mojo_benchmark():
    print("Running Mojo benchmark...")
    result = subprocess.run(
        ["pixi", "run", "mojo", "benches/bench_encode_circuit_optimized.mojo"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Mojo benchmark failed:", result.stderr)
        return {}
    
    print("Processing results...")
    mojo_times = {}
    lines = result.stdout.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for timing data
        if "," in line and "N" not in line:
            parts = line.split(",")
            try:
                n = int(parts[0].strip())
                t = float(parts[1].strip())
                mojo_times[n] = t
            except:
                pass
        
        i += 1
            
    return mojo_times

def main():
    value = 4.7
    print("Benchmarking Qiskit vs Mojo (Value Encoding Circuit)...")
    
    # Run Mojo first
    mojo_data = run_mojo_benchmark()
    
    # Run Qiskit and Compare
    print("\n" + "=" * 95)
    print(f"{'N':<5} | {'Qiskit (s)':<15} | {'Mojo (s)':<15} | {'Speedup':<15} | {'Status':<15}")
    print("-" * 95)
    
    results = []
    
    # N=3 to 29
    for n in range(26, 30):
        iters = get_iterations(n)
        try:
            q_time = benchmark_qiskit(n, value, iters)
            
            m_time = mojo_data.get(n, float('nan'))
            
            speedup = q_time / m_time if m_time > 0 else 0.0
            status = "✅ Mojo Faster" if speedup > 1.0 else "❌ Qiskit Faster"
            
            print(f"{n:<5} | {q_time:<15.6f} | {m_time:<15.6f} | {speedup:<15.2f}x | {status}")
            results.append((n, q_time, m_time, speedup))
            
        except Exception as e:
            print(f"{n:<5} | {'ERROR':<15} | {'ERROR':<15} | {'-':<15} | {str(e)}")
            
    # Save to file
    with open("benches/benchmark_comparison_results.md", "w") as f:
        f.write("# Benchmark Results: Mojo vs Qiskit (Value Encoding)\n")
        f.write("| N | Qiskit (s) | Mojo (s) | Speedup |\n")
        f.write("|---|---|---|---|\n")
        for n, q, m, s in results:
            f.write(f"| {n} | {q:.6f} | {m:.6f} | {s:.2f}x |\n")

if __name__ == "__main__":
    main()
