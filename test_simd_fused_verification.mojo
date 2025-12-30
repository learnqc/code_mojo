"""
Verify simd_fused correctness against simd_v2 baseline.
"""

from butterfly.core.state import QuantumState
from butterfly.core.execute_simd_fused import execute_simd_fused
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime


fn main() raises:
    print("simd_fused Correctness Verification")
    print("=" * 80)
    print("Comparing simd_fused vs simd_v2 (verified baseline)")
    print()

    print("Verification Phase")
    print("-" * 80)

    var all_passed = True

    for n in List[Int](15, 16, 17, 18, 19, 20):
        for circuit_type in List[String]("prep", "iqft"):
            var circuit_idx = 0 if circuit_type == "prep" else 1

            print(
                "Verifying n=" + String(n) + ", circuit=" + circuit_type + "..."
            )

            var circuits = encode_value_circuits_runtime(n, 4.7, swap=False)

            # Execute with simd_v2 (correct baseline)
            var state_correct = QuantumState(n)
            circuits[circuit_idx].execute_simd_v2_dynamic(state_correct)

            # Execute with simd_fused
            var state_fused = QuantumState(n)
            execute_simd_fused(state_fused, circuits[circuit_idx])

            # Compare
            var diff = 0.0
            for i in range(state_correct.size()):
                var dr = state_correct.re[i] - state_fused.re[i]
                var di = state_correct.im[i] - state_fused.im[i]
                diff += dr * dr + di * di

            if diff < 1e-10:
                print("  ✓ PASS (diff=" + String(diff) + ")")
            else:
                print("  ✗ FAIL (diff=" + String(diff) + ")")
                all_passed = False

    print()
    if all_passed:
        print("✓ All verifications passed!")
        print("simd_fused is correct and ready for benchmarking.")
    else:
        print("✗ Some verifications failed!")
        print("simd_fused has correctness issues that need to be fixed.")
