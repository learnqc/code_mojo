from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit, execute_simd_v2
from butterfly.core.execute_as_grid import execute_as_grid
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from butterfly.utils.visualization import print_state
from testing import assert_almost_equal


fn compare_states(s1: QuantumState, s2: QuantumState) raises:
    """Verify bit-perfect equivalence."""
    for i in range(len(s1)):
        assert_almost_equal(s1.re[i], s2.re[i], atol=1e-10)
        assert_almost_equal(s1.im[i], s2.im[i], atol=1e-10)


fn main() raises:
    print("Verifying execute_as_grid on value encoding circuit[0]")
    print("=" * 60)

    alias n = 8
    alias N = 1 << n
    var v = 4.7

    var circuits = encode_value_circuits_runtime(n, v, True)
    var prep_circuit = circuits[0].copy()

    # 1. Baseline: SIMD V2
    var state_v2 = QuantumState(n)
    execute_simd_v2[n](state_v2, prep_circuit)

    # 2. Baseline: Fused V3
    var state_v3 = QuantumState(n)
    execute_fused_v3[N](state_v3, prep_circuit)

    # 3. Target: Execute as grid (col_bits = 4)
    var state_grid = QuantumState(n)
    execute_as_grid(state_grid, prep_circuit, 4)

    # Verify
    print("Comparing Grid vs V2...")
    compare_states(state_grid, state_v2)
    print("✓ Grid vs V2 match")

    print("Comparing Grid vs V3...")
    compare_states(state_grid, state_v3)
    print("✓ Grid vs V3 match")

    print("\n✓ Verification successful for all strategies.")
