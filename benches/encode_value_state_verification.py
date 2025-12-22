import subprocess
import numpy as np
import re
from qiskit.quantum_info import Statevector, state_fidelity
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from encode_value_qiskit import encode_value_qiskit_circuit

def get_mojo_states():
    print("Running Mojo benchmark (encode_value_butterfly.mojo)...")
    result = subprocess.run(
        ["pixi", "run", "mojo", "benches/encode_value_butterfly.mojo"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Mojo execution failed:", result.stderr)
        return {}

    output = result.stdout
    states = {}
    current_n = None
    current_re = []
    current_im = []
    
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # Check for delimiter
        # === N= 3  ===
        # Allow lenient spacing
        if "=== N=" in line:
            # Save previous if exists
            if current_n is not None and current_re:
                states[current_n] = np.array(current_re) + 1j * np.array(current_im)
            
            # Parse new N
            try:
                # expecting "=== N= X ==="
                # remove === and N=
                clean = line.replace("===", "").replace("N=", "").strip()
                current_n = int(clean)
                current_re = []
                current_im = []
            except ValueError:
                print(f"Failed to parse N from delimiter: {line}")
                current_n = None
            continue

        # Ignore warning lines if any
        if "warning:" in line or "Args:" in line or "^" in line:
            continue
            
        if current_n is not None:
            # Parse complex number
            if line.endswith('i'):
                val_str = line[:-1]
                parts = val_str.split(' + ')
                if len(parts) == 2:
                    try:
                        re_val = float(parts[0])
                        im_val = float(parts[1])
                        current_re.append(re_val)
                        current_im.append(im_val)
                    except ValueError:
                        pass
    
    # Save last block
    if current_n is not None and current_re:
        states[current_n] = np.array(current_re) + 1j * np.array(current_im)
        
    return states

def main():
    value = 4.7
    mojo_states = get_mojo_states()
    
    if not mojo_states:
        print("No Mojo states found.")
        return

    all_passed = True
    
    for n, mojo_sv in mojo_states.items():
        print(f"\n{'='*40}")
        print(f"Verifying N={n}, v={value}")
        print(f"{'='*40}")
        
        # Get Qiskit State
        qc = encode_value_qiskit_circuit(n, value, swap=True)
        qiskit_sv = Statevector.from_instruction(qc).data
        
        # Compare
        fid = state_fidelity(qiskit_sv, mojo_sv, validate=False)
        print(f"Fidelity: {fid:.6f}")
        
        if fid > 0.999:
            print("✅ MATCH")
        else:
            print("❌ MISMATCH")
            all_passed = False
            
            # Print details on mismatch
            print(f"{'Idx':<5} | {'Mojo':<25} | {'Qiskit':<25} | {'Diff':<10}")
            limit = min(8, len(mojo_sv))
            for k in range(limit):
                m_prob = abs(mojo_sv[k])**2
                q_prob = abs(qiskit_sv[k])**2
                diff = abs(m_prob - q_prob)
                print(f"{k:<5} | {m_prob:<25.4f} | {q_prob:<25.4f} | {diff:<10.4f}")

    if all_passed:
        print("\n🎉 ALL STATES MATCH!")
    else:
        print("\n⚠️ SOME STATES FAILED.")

if __name__ == "__main__":
    main()
