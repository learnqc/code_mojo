import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT

def get_mojo_state():
    return None

    # for j in range(n):
    #     circuit.h(j)

    # for j in range(n):
    #     circuit.p(j, 2 * pi / 2 ** (j + 1) * v)

    # iqft_circuit(circuit, [n - 1 - j for j in range(n)], do_swap=True)

    # fn iqft(mut state: QuantumState, targets: List[Int], swap: Bool = False):
    # for j in reversed(range(len(targets))):
    #     transform(state, targets[j], H)
    #     for k in reversed(range(j)):
    #         c_transform(state, targets[j], targets[k], P(-pi / 2 ** (j - k)))

    # if swap:
    #     bit_reverse_state(state)


# fn iqft(mut circuit: QuantumCircuit, targets: List[Int], do_swap: Bool = True):
#     """
#     Adds IQFT gates to the circuit for the specified targets.
#     Matches the logic in butterfly.core.state.iqft.

#     Args:
#         circuit: The QuantumCircuit to add gates to.
#         targets: List of target qubit indices.
#         do_swap: If True, adds an efficient bit-reversal operation to reverse qubit order.
#     """
#     for j in reversed(range(len(targets))):
#         circuit.h(targets[j])
#         for k in reversed(range(j)):
#             # Note: targets[j] is control, targets[k] is target in state.mojo implementation
#             circuit.cp(targets[k], targets[j], -pi / (2 ** (j - k)))

#     if do_swap:
#         if len(targets) == circuit.num_qubits:
#             circuit.bit_reverse()
#         else:
#             # Partial swap: swap targets[i] with targets[k-1-i]
#             var k = len(targets)
#             for i in range(k // 2):
#                 circuit.swap(targets[i], targets[k - 1 - i])

def encode_value_qiskit_circuit(n: int, value: float, swap: bool = False) -> QuantumCircuit:
    """Create value encoding circuit in Qiskit (matches Mojo implementation)."""
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    
    # Hadamard on all qubits
    for i in range(n):
        qc.h(i)
    for j in range(n):
        if swap:
            qc.p(2 * np.pi / 2 ** (n-j) * value, j)
        else:
            qc.p(2 * np.pi / 2 ** (j + 1) * value, j)
    # IQFT
    # for j in reversed(range(len(targets))):
    #     # Hadamard
    #     qc.h(targets[j])
    #     # Controlled phase rotations
    #     for k in reversed(range(j)):
    #         qc.cp(-np.pi / 2 ** (j - k), targets[j], targets[k])
    
    # # Bit reversal (swap pairs)
    # for i in range(n // 2):
    #     qc.swap(i, n - 1 - i)
    iqft = QFT(n, inverse=True, do_swaps=swap)
    qc.append(iqft, range(n))
    
    return qc

# [ 0.09857767+0.03636722j  
# 0.07477825+0.06912431j  
# 0.04852498+0.10525883j
# 0.00641297+0.16322104j
# -0.12894844+0.34953004j
# 0.58402831-0.63179826j
# 0.18795316-0.08664759j
# 0.12867309-0.00505558j]

def main():
    n = 3
    value = 4.7
    
    print(f"Comparing States for N={n}, v={value}")
          
    # Get Qiskit State using EXISTING function
    print("Generating Qiskit state...")
    qc = encode_value_qiskit_circuit(n, value, True)
    qiskit_sv = Statevector.from_instruction(qc).data
    print("qiskit_sv:", qiskit_sv)
    
    print("\nState Vector Comparison:")
    
    # Get Mojo State
    mojo_sv = get_mojo_state()
    if mojo_sv is None:
        return
    
    print(f"{'Idx':<5} | {'Mojo (Complex)':<25} | {'Qiskit (Complex)':<25} | {'Mojo Pr':<10} | {'Qiskit Pr':<10}")
    print("-" * 90)
    
    err = 0.0
    
    for k in range(len(mojo_sv)):
        m_val = mojo_sv[k]
        q_val = qiskit_sv[k]
        
        m_prob = abs(m_val)**2
        q_prob = abs(q_val)**2
        
        err += abs(m_prob - q_prob)
        
        # Format complex numbers
        m_str = f"{m_val.real:.4f}{m_val.imag:+.4f}j"
        q_str = f"{q_val.real:.4f}{q_val.imag:+.4f}j"
        
        print(f"{k:<5} | {m_str:<25} | {q_str:<25} | {m_prob:<10.4f} | {q_prob:<10.4f}")
        
    print("-" * 90)
    print(f"Total Probability Error: {err:.6f}")
    
    from qiskit.quantum_info import state_fidelity
    fid = state_fidelity(qiskit_sv, mojo_sv, validate=False)
    print(f"Fidelity: {fid:.6f}")

if __name__ == "__main__":
    main()
