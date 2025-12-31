import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit import transpile

def get_qiskit_prep_state(n: int, value: float):
    """Returns the statevector after the preparation stage of value encoding."""
    qc = QuantumCircuit(n)
    
    # Hadamard on all qubits
    for i in range(n):
        qc.h(i)
    
    # Phase rotations
    for j in range(n):
        qc.p(2 * np.pi / 2 ** (j + 1) * value, j)
        
    backend = Aer.get_backend('statevector_simulator')
    qc = transpile(qc, backend)
    result = backend.run(qc).result()
    sv = result.get_statevector().data
    
    return sv.real.tolist(), sv.imag.tolist()

def get_qiskit_full_state(n: int, value: float):
    """Returns the statevector after the full value encoding (prep + iqft)."""
    qc = QuantumCircuit(n)
    
    # Prep
    for i in range(n):
        qc.h(i)
    for j in range(n):
        qc.p(2 * np.pi / 2 ** (j + 1) * value, j)
    
    # IQFT (without swaps, matching Mojo's encode_value_circuits_runtime swap=False)
    # Mojo order: reversed(range(n)) for targets=[n-1, ..., 0]
    # In Mojo: 
    # for j in reversed(range(n)):
    #     h(targets[j])
    #     for k in reversed(range(j)):
    #         cp(targets[k], targets[j], -pi / 2**(j-k))
    
    targets = [n - 1 - i for i in range(n)]
    
    for j in range(n - 1, -1, -1):
        qc.h(targets[j])
        for k in range(j - 1, -1, -1):
            # Mojo: control=targets[k], target=targets[j]
            # Since IQFT CP gates are symmetric, target/control order shouldn't matter 
            # for probability but might affect phase if convention differs.
            qc.cp(-np.pi / (2 ** (j - k)), targets[k], targets[j])
            
    backend = Aer.get_backend('statevector_simulator')
    qc = transpile(qc, backend)
    result = backend.run(qc).result()
    sv = result.get_statevector().data
    
    return sv.real.tolist(), sv.imag.tolist()
