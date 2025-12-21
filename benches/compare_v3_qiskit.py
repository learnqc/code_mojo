#!/usr/bin/env python3
"""
Create comparison table between v3 and Qiskit results.
"""

# Qiskit results (from qiskit_results_all_gates.txt)
qiskit_results = {
    3: 0.000343,
    4: 0.000356,
    5: 0.000479,
    6: 0.000658,
    7: 0.000834,
    8: 0.001217,
    9: 0.001420,
    10: 0.003183,
    11: 0.005009,
    12: 0.003736,
    13: 0.005579,
    14: 0.015846,
    15: 0.029544,
    16: 0.058545,
    17: 0.166494,
    18: 0.323772,
    19: 0.691661,
    20: 0.650839,
    21: 2.905565,
    22: 6.471364,
    23: 14.805743,
    24: 31.650514,
    25: 68.014326,
}

# V3 results (from benchmark output)
v3_results = {
    3: 1.23e-05,
    4: 1.48e-05,
    5: 2.29e-05,
    6: 2.13e-05,
    7: 2.03e-05,
    8: 2.45e-05,
    9: 2.72e-05,
    10: 3.65e-05,
    11: 0.0002099,
    12: 0.0003281,
    13: 0.0006422,
    14: 0.0008098,
    15: 0.0014478,
    16: 0.0019406,
    17: 0.003804,
    18: 0.0063592,
    19: 0.015569,
    20: 0.0289704,
    21: 0.11898233333333333,
    22: 0.11815066666666667,
    23: 0.307501,
    24: 0.549985,
    25: 1.313151,
    26: 2.5021715,
    27: 5.481583,
    28: 11.0972815,
    29: 34.452513,
}

print("=" * 90)
print("V3 Executor vs Qiskit - Value Encoding Benchmark Comparison")
print("=" * 90)
print()
print(f"{'N':<5} | {'V3 Time (s)':<15} | {'Qiskit Time (s)':<15} | {'Speedup':<10}")
print("-" * 90)

for n in range(3, 26):  # Up to N=25 where we have Qiskit data
    v3_time = v3_results[n]
    qiskit_time = qiskit_results[n]
    speedup = qiskit_time / v3_time
    
    print(f"{n:<5} | {v3_time:<15.6f} | {qiskit_time:<15.6f} | {speedup:<10.1f}x")

print("=" * 90)
print()
print("Summary:")
print(f"  Average speedup (N=3-25): {sum(qiskit_results[n]/v3_results[n] for n in range(3,26))/23:.1f}x")
print(f"  Best speedup: {max(qiskit_results[n]/v3_results[n] for n in range(3,26)):.1f}x (N={max(range(3,26), key=lambda n: qiskit_results[n]/v3_results[n])})")
print(f"  At N=25: {qiskit_results[25]/v3_results[25]:.1f}x faster than Qiskit")
print()
print("V3 results for N=26-29 (no Qiskit comparison):")
for n in range(26, 30):
    print(f"  N={n}: {v3_results[n]:.2f}s")
