"""
Radix-4 Fusion and Cache-Blocking Executor (v3).
Minimizes memory passes by grouping local gates and using Radix-4 kernels for global stages.
"""

from butterfly.core.state import QuantumState, bit_reverse_state, simd_width
from butterfly.core.types import FloatType, Type, Gate, Amplitude
from butterfly.core.circuit import QuantumTransformation
from butterfly.algos.unitary_kernels import Matrix4x4, Matrix8x8, acc_mul
from butterfly.algos.fused_gates import (
    compute_kron_product,
    compute_kron_product_3,
)
from algorithm import parallelize, vectorize
from memory import UnsafePointer

alias BLOCK_LOG = 11
alias BLOCK_SIZE = 1 << BLOCK_LOG


struct TransformationGroup(Copyable, Movable):
    """A list of transformations that can be executed in a single pass."""

    var transformations: List[QuantumTransformation]
    var is_local: Bool  # If true, these can be block-executed
    var is_radix4: Bool  # If true, these are fused into a 4x4 matrix
    var is_radix8: Bool  # If true, these are fused into an 8x8 matrix

    fn __init__(
        out self,
        is_local: Bool = False,
        is_radix4: Bool = False,
        is_radix8: Bool = False,
    ):
        self.transformations = List[QuantumTransformation]()
        self.is_local = is_local
        self.is_radix4 = is_radix4
        self.is_radix8 = is_radix8

    fn __copyinit__(out self, existing: Self):
        self.transformations = List[QuantumTransformation]()
        for i in range(len(existing.transformations)):
            self.transformations.append(existing.transformations[i].copy())
        self.is_local = existing.is_local
        self.is_radix4 = existing.is_radix4
        self.is_radix8 = existing.is_radix8

    fn __moveinit__(out self, deinit existing: Self):
        self.transformations = existing.transformations^
        self.is_local = existing.is_local
        self.is_radix4 = existing.is_radix4
        self.is_radix8 = existing.is_radix8

    fn copy(self) -> Self:
        var new_g = TransformationGroup(
            self.is_local, self.is_radix4, self.is_radix8
        )
        for i in range(len(self.transformations)):
            new_g.transformations.append(self.transformations[i].copy())
        return new_g^


struct CircuitAnalyzer:
    """Partitions a circuit into execution groups."""

    var groups: List[TransformationGroup]
    var use_radix8: Bool

    fn __init__(
        out self,
        transformations: List[QuantumTransformation],
        use_radix8: Bool = False,
    ):
        self.groups = List[TransformationGroup]()
        self.use_radix8 = use_radix8
        self.analyze(transformations)

    fn is_local(self, t: QuantumTransformation) -> Bool:
        var involved = t.get_involved_qubits()
        for i in range(len(involved)):
            if involved[i] >= BLOCK_LOG:
                return False
        return True

    fn analyze(mut self, transformations: List[QuantumTransformation]):
        var i = 0
        var n = len(transformations)

        while i < n:
            var t = transformations[i].copy()

            if t.is_permutation:
                var g = TransformationGroup()
                g.transformations.append(t^)
                self.groups.append(g^)
                i += 1
                continue

            if self.is_local(t):
                # Try to form Radix-8 triplet for local gates (if enabled)
                if self.use_radix8 and i + 2 < n:
                    var t2 = transformations[i + 1].copy()
                    var t3 = transformations[i + 2].copy()
                    if (
                        not t2.is_permutation
                        and not t3.is_permutation
                        and self.is_local(t2)
                        and self.is_local(t3)
                        and not t.is_controlled()
                        and not t2.is_controlled()
                        and not t3.is_controlled()
                        and t.target != t2.target
                        and t.target != t3.target
                        and t2.target != t3.target
                    ):
                        var g = TransformationGroup(
                            is_local=True, is_radix4=False, is_radix8=True
                        )
                        g.transformations.append(t^)
                        g.transformations.append(t2^)
                        g.transformations.append(t3^)
                        self.groups.append(g^)
                        i += 3
                        continue

                # Start a local group (cache-blocked execution)
                var g = TransformationGroup(is_local=True)
                g.transformations.append(t^)
                i += 1
                while i < n:
                    var next_t_existing = transformations[i].copy()
                    if next_t_existing.is_permutation or not self.is_local(
                        next_t_existing
                    ):
                        break
                    g.transformations.append(next_t_existing^)
                    i += 1
                self.groups.append(g^)
            else:
                # Global gate - try to pair into Radix-4
                if i + 1 < n:
                    var next_t = transformations[i + 1].copy()
                    if (
                        not next_t.is_permutation
                        and not self.is_local(next_t)
                        and not t.is_controlled()
                        and not next_t.is_controlled()
                        and t.target != next_t.target
                    ):
                        var g = TransformationGroup(
                            is_local=False, is_radix4=True
                        )
                        g.transformations.append(t^)
                        g.transformations.append(next_t^)
                        self.groups.append(g^)
                        i += 2
                        continue

                # Fallback: Single Global group
                var g = TransformationGroup(is_local=False, is_radix4=False)
                g.transformations.append(t^)
                self.groups.append(g^)
                i += 1


fn execute_fused_v3[
    N: Int
](
    mut state: QuantumState,
    transformations: List[QuantumTransformation],
    use_radix8: Bool = False,
):
    """Main entry point for Radix-4 Cache-Blocking execution.

    Args:
        state: The quantum state to transform.
        transformations: List of transformations to apply.
        use_radix8: If True, use Radix-8 fusion for local gate triplets (default: False).
    """
    var analyzer = CircuitAnalyzer(transformations, use_radix8)

    for i in range(len(analyzer.groups)):
        var g_ref = analyzer.groups[i].copy()

        if g_ref.is_radix8:
            execute_radix8_local_group(state, g_ref.transformations)
        elif g_ref.is_local:
            execute_local_group(state, g_ref.transformations)
        elif g_ref.is_radix4:
            execute_radix4_group[N](state, g_ref.transformations)
        else:
            execute_global_group[N](state, g_ref.transformations)


fn execute_local_group(
    mut state: QuantumState, transformations: List[QuantumTransformation]
):
    """Execute a group of local gates using cache-blocking."""
    var size = state.size()
    var num_blocks = size >> BLOCK_LOG
    var ptr_re = UnsafePointer[FloatType](state.re.unsafe_ptr().address)
    var ptr_im = UnsafePointer[FloatType](state.im.unsafe_ptr().address)

    @parameter
    fn block_worker(block_idx: Int):
        var start = block_idx << BLOCK_LOG
        for i in range(len(transformations)):
            var t_copy = transformations[i].copy()
            apply_local_transform(
                ptr_re, ptr_im, start, t_copy.target, t_copy.gate
            )

    parallelize[block_worker](num_blocks)


fn apply_local_transform(
    mut ptr_re: UnsafePointer[FloatType],
    mut ptr_im: UnsafePointer[FloatType],
    start: Int,
    target: Int,
    gate: Gate,
):
    """Applies a local transform to a specific cache block."""
    var stride = 1 << target
    var count = BLOCK_SIZE // 2

    var g00 = gate[0][0]
    var g01 = gate[0][1]
    var g10 = gate[1][0]
    var g11 = gate[1][1]

    @parameter
    fn v_butterfly[w: Int](i: Int):
        var p_re = UnsafePointer[FloatType](ptr_re.address)
        var p_im = UnsafePointer[FloatType](ptr_im.address)

        var group = i // stride
        var offset = i % stride
        var idx0 = start + group * 2 * stride + offset
        var idx1 = idx0 + stride

        var r0 = ptr_re.load[width=w](idx0)
        var i0 = ptr_im.load[width=w](idx0)
        var r1 = ptr_re.load[width=w](idx1)
        var i1 = ptr_im.load[width=w](idx1)

        var res0_re = SIMD[DType.float64, w](0.0)
        var res0_im = SIMD[DType.float64, w](0.0)
        acc_mul(res0_re, res0_im, g00, r0, i0)
        acc_mul(res0_re, res0_im, g01, r1, i1)

        var res1_re = SIMD[DType.float64, w](0.0)
        var res1_im = SIMD[DType.float64, w](0.0)
        acc_mul(res1_re, res1_im, g10, r0, i0)
        acc_mul(res1_re, res1_im, g11, r1, i1)

        p_re.store(idx0, res0_re)
        p_im.store(idx0, res0_im)
        p_re.store(idx1, res1_re)
        p_im.store(idx1, res1_im)

    vectorize[v_butterfly, simd_width](count)


fn execute_radix8_local_group(
    mut state: QuantumState, transformations: List[QuantumTransformation]
):
    """Execute a triplet of local gates using Radix-8 fusion."""
    # Should have exactly 3 transformations
    if len(transformations) != 3:
        return

    var t0 = transformations[0].copy()
    var t1 = transformations[1].copy()
    var t2 = transformations[2].copy()

    # Sort by target (descending for Kronecker product ordering)
    var targets = List[Int](capacity=3)
    var gates = List[Gate](capacity=3)
    targets.append(t0.target)
    targets.append(t1.target)
    targets.append(t2.target)
    gates.append(t0.gate)
    gates.append(t1.gate)
    gates.append(t2.gate)

    # Bubble sort
    for _ in range(2):
        for j in range(2):
            if targets[j] < targets[j + 1]:
                var tmp_t = targets[j]
                targets[j] = targets[j + 1]
                targets[j + 1] = tmp_t
                var tmp_g = gates[j]
                gates[j] = gates[j + 1]
                gates[j + 1] = tmp_g

    var q_high = targets[0]
    var q_mid = targets[1]
    var q_low = targets[2]

    # Compute fused 8x8 matrix: M = g_high ⊗ g_mid ⊗ g_low
    var M = compute_kron_product_3(gates[0], gates[1], gates[2])

    # Apply using transform_matrix8 from fused_gates
    from butterfly.algos.fused_gates import transform_matrix8

    transform_matrix8(state, q_high, q_mid, q_low, M)


fn execute_global_group[
    N: Int
](mut state: QuantumState, transformations: List[QuantumTransformation]):
    """Execute a group of global gates."""
    from butterfly.core.execute_simd_v2_dispatch import (
        execute_transformations_simd_v2,
    )

    execute_transformations_simd_v2[N](state, transformations)


fn execute_radix4_group[
    N: Int
](mut state: QuantumState, transformations: List[QuantumTransformation]):
    """Execute two global gates fused into a Radix-4 passage."""
    var t0 = transformations[0].copy()
    var t1 = transformations[1].copy()

    var q_high = t0.target
    var q_low = t1.target
    var gate_high = t0.gate
    var gate_low = t1.gate

    if q_low > q_high:
        var tmp_q = q_high
        q_high = q_low
        q_low = tmp_q
        var tmp_g = gate_high
        gate_high = gate_low
        gate_low = tmp_g

    var M = compute_fused_matrix(gate_high, gate_low)
    generic_radix4_kernel[N](state, q_high, q_low, M)


fn compute_fused_matrix(u_high: Gate, u_low: Gate) -> Matrix4x4:
    var m = Matrix4x4(
        InlineArray[Amplitude, 4](
            Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0)
        ),
        InlineArray[Amplitude, 4](
            Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0)
        ),
        InlineArray[Amplitude, 4](
            Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0)
        ),
        InlineArray[Amplitude, 4](
            Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0)
        ),
    )
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    var row = 2 * i + k
                    var col = 2 * j + l
                    m[row][col] = u_high[i][j] * u_low[k][l]
    return m


fn generic_radix4_kernel[
    N: Int
](mut state: QuantumState, q_high: Int, q_low: Int, M: Matrix4x4):
    var ptr_re = UnsafePointer[FloatType](state.re.unsafe_ptr().address)
    var ptr_im = UnsafePointer[FloatType](state.im.unsafe_ptr().address)

    var s_high = 1 << q_high
    var s_low = 1 << q_low
    var count = (1 << N) // 4

    var low_mask = (1 << q_low) - 1
    var mid_mask = ((1 << q_high) - 1) ^ low_mask
    var high_mask = ~((1 << q_high) - 1)

    @parameter
    fn v_radix4[w: Int](i: Int):
        var p_re = UnsafePointer[FloatType](ptr_re.address)
        var p_im = UnsafePointer[FloatType](ptr_im.address)

        var idx = (
            ((i & high_mask) << 2) | ((i & mid_mask) << 1) | (i & low_mask)
        )

        var idx0 = idx
        var idx1 = idx | s_low
        var idx2 = idx | s_high
        var idx3 = idx | s_high | s_low

        var r0 = ptr_re.load[width=w](idx0)
        var i0 = ptr_im.load[width=w](idx0)
        var r1 = ptr_re.load[width=w](idx1)
        var i1 = ptr_im.load[width=w](idx1)
        var r2 = ptr_re.load[width=w](idx2)
        var i2 = ptr_im.load[width=w](idx2)
        var r3 = ptr_re.load[width=w](idx3)
        var i3 = ptr_im.load[width=w](idx3)

        var z_re0 = SIMD[DType.float64, w](0.0)
        var z_im0 = SIMD[DType.float64, w](0.0)
        var z_re1 = SIMD[DType.float64, w](0.0)
        var z_im1 = SIMD[DType.float64, w](0.0)
        var z_re2 = SIMD[DType.float64, w](0.0)
        var z_im2 = SIMD[DType.float64, w](0.0)
        var z_re3 = SIMD[DType.float64, w](0.0)
        var z_im3 = SIMD[DType.float64, w](0.0)

        acc_mul(z_re0, z_im0, M[0][0], r0, i0)
        acc_mul(z_re0, z_im0, M[0][1], r1, i1)
        acc_mul(z_re0, z_im0, M[0][2], r2, i2)
        acc_mul(z_re0, z_im0, M[0][3], r3, i3)

        acc_mul(z_re1, z_im1, M[1][0], r0, i0)
        acc_mul(z_re1, z_im1, M[1][1], r1, i1)
        acc_mul(z_re1, z_im1, M[1][2], r2, i2)
        acc_mul(z_re1, z_im1, M[1][3], r3, i3)

        acc_mul(z_re2, z_im2, M[2][0], r0, i0)
        acc_mul(z_re2, z_im2, M[2][1], r1, i1)
        acc_mul(z_re2, z_im2, M[2][2], r2, i2)
        acc_mul(z_re2, z_im2, M[2][3], r3, i3)

        acc_mul(z_re3, z_im3, M[3][0], r0, i0)
        acc_mul(z_re3, z_im3, M[3][1], r1, i1)
        acc_mul(z_re3, z_im3, M[3][2], r2, i2)
        acc_mul(z_re3, z_im3, M[3][3], r3, i3)

        p_re.store(idx0, z_re0)
        p_im.store(idx0, z_im0)
        p_re.store(idx1, z_re1)
        p_im.store(idx1, z_im1)
        p_re.store(idx2, z_re2)
        p_im.store(idx2, z_im2)
        p_re.store(idx3, z_re3)
        p_im.store(idx3, z_im3)

    @parameter
    fn global_worker(idx: Int):
        v_radix4[1](idx)

    parallelize[global_worker](count)
