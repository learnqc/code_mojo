from butterfly.core.state import QuantumState
from butterfly.utils.bit_utils import insert_zero_bit
from butterfly.core.types import FloatType, Type, Amplitude, Gate
from butterfly.algos.unitary_kernels import (
    Matrix4x4,
    Matrix8x8,
    Matrix16x16,
    acc_mul,
    unitary_radix4_kernel,
)
from sys.info import simd_width_of
from memory import UnsafePointer
from algorithm import vectorize, parallelize

alias simd_width = simd_width_of[Type]()


@always_inline
fn compute_kron_product(g1: Gate, g2: Gate) -> Matrix4x4:
    """
    Computes the Kronecker product M = g1 x g2.
    Here g1 is the gate for the MORE significant qubit (higher index),
    and g2 is for the LESS significant qubit (lower index).
    M_(2*i+k, 2*j+l) = g1_ij * g2_kl
    """
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
                    m[row][col] = g1[i][j] * g2[k][l]
    return m


fn transform_fused(
    mut state: QuantumState, t1: Int, g1: Gate, t2: Int, g2: Gate
):
    """
    Applies two single-qubit gates U1 (on t1) and U2 (on t2) in a single pass.
    This effectively applies M = U1 x U2 (if t1 > t2) or U2 x U1 (if t2 > t1).
    """
    var n = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    # Determine qubit ordering for Kronecker product
    # We want the matrix such that applying it to vector (00, 01, 10, 11) works.
    # If t_high > t_low:
    #   00: 0_high 0_low
    #   01: 0_high 1_low
    #   10: 1_high 0_low
    #   11: 1_high 1_low
    # This corresponds to standard Kron product ordering g_high \u2297 g_low.

    var M: Matrix4x4
    var low_pos: Int
    var high_pos: Int

    if t1 > t2:
        M = compute_kron_product(g1, g2)
        high_pos = t1
        low_pos = t2
    else:
        M = compute_kron_product(g2, g1)
        high_pos = t2
        low_pos = t1

    var num_quads = n >> 2

    # Parallelize
    alias chunk_size = 256  # Process 256 quads per chunk (1024 amplitudes)
    var num_chunks = num_quads // chunk_size

    @parameter
    fn worker(chunk_idx: Int):
        var start = chunk_idx * chunk_size
        var end = start + chunk_size

        # We need a kernel that can handle arbitrary strides (scattered quads).
        # unitary_radix4_kernel expects stride, 2*stride, 3*stride spacing.
        # This is ONLY true if low_pos and high_pos are adjacent or related in a specific way?
        # No, 'stride' in unitary_radix4_kernel is a single value.
        # Our indices are:
        # idx0 = insert_zero(k, low, high)
        # idx1 = idx0 + (1<<low)
        # idx2 = idx0 + (1<<high)
        # idx3 = idx0 + (1<<low) + (1<<high)
        # This is NOT a linear stride unless 1<<high == 2 * (1<<low) i.e. adjacent?
        # Even then, it's: idx, idx+S, idx+2S, idx+S+2S = idx+3S.
        # Yes! If high = low + 1, then stride_high = 2*stride_low.
        # Then indices are: 0, S, 2S, 3S.
        # So unitary_radix4_kernel ONLY works for ADJACENT qubits.

        # For arbitrary qubits, we need a general kernel.
        # Let's implement a 'scattered' kernel here or use a localized one.

        for k in range(start, end):
            var idx = insert_zero_bit(k, low_pos)
            var idx0 = insert_zero_bit(idx, high_pos)

            var stride1 = 1 << low_pos
            var stride2 = 1 << high_pos

            var idx1 = idx0 | stride1
            var idx2 = idx0 | stride2
            var idx3 = idx1 | stride2

            # Load 4 values (scalar loads because gaps are arbitrary)
            # Or can we vectorize?
            # If we process SIMD 'k' values, we generate SIMD 'idx0's.
            # Then we can gather 'idx0', 'idx1', 'idx2', 'idx3'.

            # Let's try vectorized loop within the chunk
            # TODO: Add SIMD optimization later if needed. For now, scalar loop over k,
            # but we can manually inline the matrix mul for correctness first.

            var re0 = ptr_re[idx0]
            var im0 = ptr_im[idx0]
            var re1 = ptr_re[idx1]
            var im1 = ptr_im[idx1]
            var re2 = ptr_re[idx2]
            var im2 = ptr_im[idx2]
            var re3 = ptr_re[idx3]
            var im3 = ptr_im[idx3]

            # Apply M
            # 0
            var n_re0 = re0 * M[0][0].re - im0 * M[0][0].im
            var n_im0 = re0 * M[0][0].im + im0 * M[0][0].re
            n_re0 += re1 * M[0][1].re - im1 * M[0][1].im
            n_im0 += re1 * M[0][1].im + im1 * M[0][1].re
            n_re0 += re2 * M[0][2].re - im2 * M[0][2].im
            n_im0 += re2 * M[0][2].im + im2 * M[0][2].re
            n_re0 += re3 * M[0][3].re - im3 * M[0][3].im
            n_im0 += re3 * M[0][3].im + im3 * M[0][3].re

            # 1
            var n_re1 = re0 * M[1][0].re - im0 * M[1][0].im
            var n_im1 = re0 * M[1][0].im + im0 * M[1][0].re
            n_re1 += re1 * M[1][1].re - im1 * M[1][1].im
            n_im1 += re1 * M[1][1].im + im1 * M[1][1].re
            n_re1 += re2 * M[1][2].re - im2 * M[1][2].im
            n_im1 += re2 * M[1][2].im + im2 * M[1][2].re
            n_re1 += re3 * M[1][3].re - im3 * M[1][3].im
            n_im1 += re3 * M[1][3].im + im3 * M[1][3].re

            # 2
            var n_re2 = re0 * M[2][0].re - im0 * M[2][0].im
            var n_im2 = re0 * M[2][0].im + im0 * M[2][0].re
            n_re2 += re1 * M[2][1].re - im1 * M[2][1].im
            n_im2 += re1 * M[2][1].im + im1 * M[2][1].re
            n_re2 += re2 * M[2][2].re - im2 * M[2][2].im
            n_im2 += re2 * M[2][2].im + im2 * M[2][2].re
            n_re2 += re3 * M[2][3].re - im3 * M[2][3].im
            n_im2 += re3 * M[2][3].im + im3 * M[2][3].re

            # 3
            var n_re3 = re0 * M[3][0].re - im0 * M[3][0].im
            var n_im3 = re0 * M[3][0].im + im0 * M[3][0].re
            n_re3 += re1 * M[3][1].re - im1 * M[3][1].im
            n_im3 += re1 * M[3][1].im + im1 * M[3][1].re
            n_re3 += re2 * M[3][2].re - im2 * M[3][2].im
            n_im3 += re2 * M[3][2].im + im2 * M[3][2].re
            n_re3 += re3 * M[3][3].re - im3 * M[3][3].im
            n_im3 += re3 * M[3][3].im + im3 * M[3][3].re

            ptr_re[idx0] = n_re0
            ptr_im[idx0] = n_im0
            ptr_re[idx1] = n_re1
            ptr_im[idx1] = n_im1
            ptr_re[idx2] = n_re2
            ptr_im[idx2] = n_im2
            ptr_re[idx3] = n_re3
            ptr_im[idx3] = n_im3

    parallelize[worker](num_chunks)

    # Handle remainder
    for k in range(num_chunks * chunk_size, num_quads):
        var idx = insert_zero_bit(k, low_pos)
        var idx0 = insert_zero_bit(idx, high_pos)

        var stride1 = 1 << low_pos
        var stride2 = 1 << high_pos

        var idx1 = idx0 | stride1
        var idx2 = idx0 | stride2
        var idx3 = idx1 | stride2

        var re0 = ptr_re[idx0]
        var im0 = ptr_im[idx0]
        var re1 = ptr_re[idx1]
        var im1 = ptr_im[idx1]
        var re2 = ptr_re[idx2]
        var im2 = ptr_im[idx2]
        var re3 = ptr_re[idx3]
        var im3 = ptr_im[idx3]

        var n_re0 = re0 * M[0][0].re - im0 * M[0][0].im
        var n_im0 = re0 * M[0][0].im + im0 * M[0][0].re
        n_re0 += re1 * M[0][1].re - im1 * M[0][1].im
        n_im0 += re1 * M[0][1].im + im1 * M[0][1].re
        n_re0 += re2 * M[0][2].re - im2 * M[0][2].im
        n_im0 += re2 * M[0][2].im + im2 * M[0][2].re
        n_re0 += re3 * M[0][3].re - im3 * M[0][3].im
        n_im0 += re3 * M[0][3].im + im3 * M[0][3].re

        var n_re1 = re0 * M[1][0].re - im0 * M[1][0].im
        var n_im1 = re0 * M[1][0].im + im0 * M[1][0].re
        n_re1 += re1 * M[1][1].re - im1 * M[1][1].im
        n_im1 += re1 * M[1][1].im + im1 * M[1][1].re
        n_re1 += re2 * M[1][2].re - im2 * M[1][2].im
        n_im1 += re2 * M[1][2].im + im2 * M[1][2].re
        n_re1 += re3 * M[1][3].re - im3 * M[1][3].im
        n_im1 += re3 * M[1][3].im + im3 * M[1][3].re

        var n_re2 = re0 * M[2][0].re - im0 * M[2][0].im
        var n_im2 = re0 * M[2][0].im + im0 * M[2][0].re
        n_re2 += re1 * M[2][1].re - im1 * M[2][1].im
        n_im2 += re1 * M[2][1].im + im1 * M[2][1].re
        n_re2 += re2 * M[2][2].re - im2 * M[2][2].im
        n_im2 += re2 * M[2][2].im + im2 * M[2][2].re
        n_re2 += re3 * M[2][3].re - im3 * M[2][3].im
        n_im2 += re3 * M[2][3].im + im3 * M[2][3].re

        var n_re3 = re0 * M[3][0].re - im0 * M[3][0].im
        var n_im3 = re0 * M[3][0].im + im0 * M[3][0].re
        n_re3 += re1 * M[3][1].re - im1 * M[3][1].im
        n_im3 += re1 * M[3][1].im + im1 * M[3][1].re
        n_re3 += re2 * M[3][2].re - im2 * M[3][2].im
        n_im3 += re2 * M[3][2].im + im2 * M[3][2].re
        n_re3 += re3 * M[3][3].re - im3 * M[3][3].im
        n_im3 += re3 * M[3][3].im + im3 * M[3][3].re

        ptr_re[idx0] = n_re0
        ptr_im[idx0] = n_im0
        ptr_re[idx1] = n_re1
        ptr_im[idx1] = n_im1
        ptr_re[idx2] = n_re2
        ptr_im[idx2] = n_im2
        ptr_re[idx3] = n_re3
        ptr_im[idx3] = n_im3


fn compute_kron_product_3(g1: Gate, g2: Gate, g3: Gate) -> Matrix8x8:
    """Matrix 8x8 = g1 x g2 x g3. g1=highest, g3=lowest."""
    # This can be implemented via 2 kron products
    var m4 = compute_kron_product(g1, g2)
    var row = InlineArray[Amplitude, 8](
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
    )
    var res = Matrix8x8(row, row, row, row, row, row, row, row)
    for i in range(4):
        for j in range(4):
            for k in range(2):
                for l in range(2):
                    res[2 * i + k][2 * j + l] = m4[i][j] * g3[k][l]
    return res


fn transform_fused_3(
    mut state: QuantumState,
    t1: Int,
    g1: Gate,
    t2: Int,
    g2: Gate,
    t3: Int,
    g3: Gate,
):
    """
    Applies 3 single-qubit gates in one pass.
    """
    var ta = t1
    var tb = t2
    var tc = t3
    var ga = g1
    var gb = g2
    var gc = g3

    # Bubble sort 3 items
    if ta > tb:
        var tmp = ta
        ta = tb
        tb = tmp
        var tmpg = ga
        ga = gb
        gb = tmpg
    if tb > tc:
        var tmp = tb
        tb = tc
        tc = tmp
        var tmpg = gb
        gb = gc
        gc = tmpg
    if ta > tb:
        var tmp = ta
        ta = tb
        tb = tmp
        var tmpg = ga
        ga = gb
        gb = tmpg

    # Now ta < tb < tc.
    var M = compute_kron_product_3(gc, gb, ga)
    transform_matrix8(state, tc, tb, ta, M)


fn transform_matrix4(mut state: QuantumState, t1: Int, t2: Int, mat: Matrix4x4):
    """
    Applies an arbitrary 4x4 matrix 'mat' to qubits t1 and t2.
    'mat' is assumed to act on the basis ordered by qubit index (Higher Index, Lower Index).
    i.e., if t1 > t2: mat acts on |t1 t2>. If t2 > t1: mat acts on |t2 t1>.
    """
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var n = state.size()
    var num_quads = n >> 2

    # Identify high and low for loop structure
    var high_pos = t1
    var low_pos = t2
    if t2 > t1:
        high_pos = t2
        low_pos = t1

    var stride_low = 1 << low_pos
    var stride_high = 1 << high_pos

    # If low_pos is large enough, we can use contiguous SIMD loads
    if stride_low >= simd_width:
        # Pre-broadcast matrix elements to SIMD vectors
        var m00_re = SIMD[DType.float64, simd_width](mat[0][0].re)
        var m00_im = SIMD[DType.float64, simd_width](mat[0][0].im)
        var m01_re = SIMD[DType.float64, simd_width](mat[0][1].re)
        var m01_im = SIMD[DType.float64, simd_width](mat[0][1].im)
        var m02_re = SIMD[DType.float64, simd_width](mat[0][2].re)
        var m02_im = SIMD[DType.float64, simd_width](mat[0][2].im)
        var m03_re = SIMD[DType.float64, simd_width](mat[0][3].re)
        var m03_im = SIMD[DType.float64, simd_width](mat[0][3].im)

        var m10_re = SIMD[DType.float64, simd_width](mat[1][0].re)
        var m10_im = SIMD[DType.float64, simd_width](mat[1][0].im)
        var m11_re = SIMD[DType.float64, simd_width](mat[1][1].re)
        var m11_im = SIMD[DType.float64, simd_width](mat[1][1].im)
        var m12_re = SIMD[DType.float64, simd_width](mat[1][2].re)
        var m12_im = SIMD[DType.float64, simd_width](mat[1][2].im)
        var m13_re = SIMD[DType.float64, simd_width](mat[1][3].re)
        var m13_im = SIMD[DType.float64, simd_width](mat[1][3].im)

        var m20_re = SIMD[DType.float64, simd_width](mat[2][0].re)
        var m20_im = SIMD[DType.float64, simd_width](mat[2][0].im)
        var m21_re = SIMD[DType.float64, simd_width](mat[2][1].re)
        var m21_im = SIMD[DType.float64, simd_width](mat[2][1].im)
        var m22_re = SIMD[DType.float64, simd_width](mat[2][2].re)
        var m22_im = SIMD[DType.float64, simd_width](mat[2][2].im)
        var m23_re = SIMD[DType.float64, simd_width](mat[2][3].re)
        var m23_im = SIMD[DType.float64, simd_width](mat[2][3].im)

        var m30_re = SIMD[DType.float64, simd_width](mat[3][0].re)
        var m30_im = SIMD[DType.float64, simd_width](mat[3][0].im)
        var m31_re = SIMD[DType.float64, simd_width](mat[3][1].re)
        var m31_im = SIMD[DType.float64, simd_width](mat[3][1].im)
        var m32_re = SIMD[DType.float64, simd_width](mat[3][2].re)
        var m32_im = SIMD[DType.float64, simd_width](mat[3][2].im)
        var m33_re = SIMD[DType.float64, simd_width](mat[3][3].re)
        var m33_im = SIMD[DType.float64, simd_width](mat[3][3].im)

        var num_outer = n >> (high_pos + 1)
        var num_mid = 1 << (high_pos - low_pos - 1)
        var num_inner = 1 << low_pos

        @parameter
        fn worker_simd(outer_idx: Int):
            var base_outer = outer_idx << (high_pos + 1)
            for m in range(num_mid):
                var base_mid = base_outer | (m << (low_pos + 1))

                @parameter
                fn inner_simd[w: Int](i: Int):
                    var idx0 = base_mid | i
                    var idx1 = idx0 | stride_low
                    var idx2 = idx0 | stride_high
                    var idx3 = idx1 | stride_high

                    # SIMD Loads
                    var re0 = ptr_re.load[width=w](idx0)
                    var im0 = ptr_im.load[width=w](idx0)
                    var re1 = ptr_re.load[width=w](idx1)
                    var im1 = ptr_im.load[width=w](idx1)
                    var re2 = ptr_re.load[width=w](idx2)
                    var im2 = ptr_im.load[width=w](idx2)
                    var re3 = ptr_re.load[width=w](idx3)
                    var im3 = ptr_im.load[width=w](idx3)

                    # Rebind the matrix vectors to size 'w' if they are different
                    # (In this case w will be simd_width, but we must use w for the type)
                    # We can use SIMD[dtype, w](scalar) for broadcast since the values are constant
                    var cur_m00_re = SIMD[DType.float64, w](m00_re[0])
                    var cur_m00_im = SIMD[DType.float64, w](m00_im[0])
                    var cur_m01_re = SIMD[DType.float64, w](m01_re[0])
                    var cur_m01_im = SIMD[DType.float64, w](m01_im[0])
                    var cur_m02_re = SIMD[DType.float64, w](m02_re[0])
                    var cur_m02_im = SIMD[DType.float64, w](m02_im[0])
                    var cur_m03_re = SIMD[DType.float64, w](m03_re[0])
                    var cur_m03_im = SIMD[DType.float64, w](m03_im[0])

                    var cur_m10_re = SIMD[DType.float64, w](m10_re[0])
                    var cur_m10_im = SIMD[DType.float64, w](m10_im[0])
                    var cur_m11_re = SIMD[DType.float64, w](m11_re[0])
                    var cur_m11_im = SIMD[DType.float64, w](m11_im[0])
                    var cur_m12_re = SIMD[DType.float64, w](m12_re[0])
                    var cur_m12_im = SIMD[DType.float64, w](m12_im[0])
                    var cur_m13_re = SIMD[DType.float64, w](m13_re[0])
                    var cur_m13_im = SIMD[DType.float64, w](m13_im[0])

                    var cur_m20_re = SIMD[DType.float64, w](m20_re[0])
                    var cur_m20_im = SIMD[DType.float64, w](m20_im[0])
                    var cur_m21_re = SIMD[DType.float64, w](m21_re[0])
                    var cur_m21_im = SIMD[DType.float64, w](m21_im[0])
                    var cur_m22_re = SIMD[DType.float64, w](m22_re[0])
                    var cur_m22_im = SIMD[DType.float64, w](m22_im[0])
                    var cur_m23_re = SIMD[DType.float64, w](m23_re[0])
                    var cur_m23_im = SIMD[DType.float64, w](m23_im[0])

                    var cur_m30_re = SIMD[DType.float64, w](m30_re[0])
                    var cur_m30_im = SIMD[DType.float64, w](m30_im[0])
                    var cur_m31_re = SIMD[DType.float64, w](m31_re[0])
                    var cur_m31_im = SIMD[DType.float64, w](m31_im[0])
                    var cur_m32_re = SIMD[DType.float64, w](m32_re[0])
                    var cur_m32_im = SIMD[DType.float64, w](m32_im[0])
                    var cur_m33_re = SIMD[DType.float64, w](m33_re[0])
                    var cur_m33_im = SIMD[DType.float64, w](m33_im[0])

                    # Complex Matrix Mul (Vectorized)
                    # Output 0
                    var n_re0 = re0 * cur_m00_re - im0 * cur_m00_im
                    var n_im0 = re0 * cur_m00_im + im0 * cur_m00_re
                    n_re0 += re1 * cur_m01_re - im1 * cur_m01_im
                    n_im0 += re1 * cur_m01_im + im1 * cur_m01_re
                    n_re0 += re2 * cur_m02_re - im2 * cur_m02_im
                    n_im0 += re2 * cur_m02_im + im2 * cur_m02_re
                    n_re0 += re3 * cur_m03_re - im3 * cur_m03_im
                    n_im0 += re3 * cur_m03_im + im3 * cur_m03_re

                    # Output 1
                    var n_re1 = re0 * cur_m10_re - im0 * cur_m10_im
                    var n_im1 = re0 * cur_m10_im + im0 * cur_m10_re
                    n_re1 += re1 * cur_m11_re - im1 * cur_m11_im
                    n_im1 += re1 * cur_m11_im + im1 * cur_m11_re
                    n_re1 += re2 * cur_m12_re - im2 * cur_m12_im
                    n_im1 += re2 * cur_m12_im + im2 * cur_m12_re
                    n_re1 += re3 * cur_m13_re - im3 * cur_m13_im
                    n_im1 += re3 * cur_m13_im + im3 * cur_m13_re

                    # Output 2
                    var n_re2 = re0 * cur_m20_re - im0 * cur_m20_im
                    var n_im2 = re0 * cur_m20_im + im0 * cur_m20_re
                    n_re2 += re1 * cur_m21_re - im1 * cur_m21_im
                    n_im2 += re1 * cur_m21_im + im1 * cur_m21_re
                    n_re2 += re2 * cur_m22_re - im2 * cur_m22_im
                    n_im2 += re2 * cur_m22_im + im2 * cur_m22_re
                    n_re2 += re3 * cur_m23_re - im3 * cur_m23_im
                    n_im2 += re3 * cur_m23_im + im3 * cur_m23_re

                    # Output 3
                    var n_re3 = re0 * cur_m30_re - im0 * cur_m30_im
                    var n_im3 = re0 * cur_m30_im + im0 * cur_m30_re
                    n_re3 += re1 * cur_m31_re - im1 * cur_m31_im
                    n_im3 += re1 * cur_m31_im + im1 * cur_m31_re
                    n_re3 += re2 * cur_m32_re - im2 * cur_m32_im
                    n_im3 += re2 * cur_m32_im + im2 * cur_m32_re
                    n_re3 += re3 * cur_m33_re - im3 * cur_m33_im
                    n_im3 += re3 * cur_m33_im + im3 * cur_m33_re

                    # Stores
                    ptr_re.store(idx0, n_re0)
                    ptr_im.store(idx0, n_im0)
                    ptr_re.store(idx1, n_re1)
                    ptr_im.store(idx1, n_im1)
                    ptr_re.store(idx2, n_re2)
                    ptr_im.store(idx2, n_im2)
                    ptr_re.store(idx3, n_re3)
                    ptr_im.store(idx3, n_im3)

                vectorize[inner_simd, simd_width](num_inner)

        parallelize[worker_simd](num_outer)
        return

    # Fallback for small low_pos (scalar)
    alias chunk_size = 256
    var num_chunks = num_quads // chunk_size

    @parameter
    fn worker(chunk_idx: Int):
        var start = chunk_idx * chunk_size
        var end = start + chunk_size

        for k in range(start, end):
            var idx = insert_zero_bit(k, low_pos)
            var idx0 = insert_zero_bit(idx, high_pos)

            var idx1 = idx0 | stride_low
            var idx2 = idx0 | stride_high
            var idx3 = idx1 | stride_high

            var re0 = ptr_re[idx0]
            var im0 = ptr_im[idx0]
            var re1 = ptr_re[idx1]
            var im1 = ptr_im[idx1]
            var re2 = ptr_re[idx2]
            var im2 = ptr_im[idx2]
            var re3 = ptr_re[idx3]
            var im3 = ptr_im[idx3]

            var n_re0 = re0 * mat[0][0].re - im0 * mat[0][0].im
            var n_im0 = re0 * mat[0][0].im + im0 * mat[0][0].re
            n_re0 += re1 * mat[0][1].re - im1 * mat[0][1].im
            n_im0 += re1 * mat[0][1].im + im1 * mat[0][1].re
            n_re0 += re2 * mat[0][2].re - im2 * mat[0][2].im
            n_im0 += re2 * mat[0][2].im + im2 * mat[0][2].re
            n_re0 += re3 * mat[0][3].re - im3 * mat[0][3].im
            n_im0 += re3 * mat[0][3].im + im3 * mat[0][3].re

            var n_re1 = re0 * mat[1][0].re - im0 * mat[1][0].im
            var n_im1 = re0 * mat[1][0].im + im0 * mat[1][0].re
            n_re1 += re1 * mat[1][1].re - im1 * mat[1][1].im
            n_im1 += re1 * mat[1][1].im + im1 * mat[1][1].re
            n_re1 += re2 * mat[1][2].re - im2 * mat[1][2].im
            n_im1 += re2 * mat[1][2].im + im2 * mat[1][2].re
            n_re1 += re3 * mat[1][3].re - im3 * mat[1][3].im
            n_im1 += re3 * mat[1][3].im + im3 * mat[1][3].re

            var n_re2 = re0 * mat[2][0].re - im0 * mat[2][0].im
            var n_im2 = re0 * mat[2][0].im + im0 * mat[2][0].re
            n_re2 += re1 * mat[2][1].re - im1 * mat[2][1].im
            n_im2 += re1 * mat[2][1].im + im1 * mat[2][1].re
            n_re2 += re2 * mat[2][2].re - im2 * mat[2][2].im
            n_im2 += re2 * mat[2][2].im + im2 * mat[2][2].re
            n_re2 += re3 * mat[2][3].re - im3 * mat[2][3].im
            n_im2 += re3 * mat[2][3].im + im3 * mat[2][3].re

            var n_re3 = re0 * mat[3][0].re - im0 * mat[3][0].im
            var n_im3 = re0 * mat[3][0].im + im0 * mat[3][0].re
            n_re3 += re1 * mat[3][1].re - im1 * mat[3][1].im
            n_im3 += re1 * mat[3][1].im + im1 * mat[3][1].re
            n_re3 += re2 * mat[3][2].re - im2 * mat[3][2].im
            n_im3 += re2 * mat[3][2].im + im2 * mat[3][2].re
            n_re3 += re3 * mat[3][3].re - im3 * mat[3][3].im
            n_im3 += re3 * mat[3][3].im + im3 * mat[3][3].re

            ptr_re[idx0] = n_re0
            ptr_im[idx0] = n_im0
            ptr_re[idx1] = n_re1
            ptr_im[idx1] = n_im1
            ptr_re[idx2] = n_re2
            ptr_im[idx2] = n_im2
            ptr_re[idx3] = n_re3
            ptr_im[idx3] = n_im3

    parallelize[worker](num_chunks)

    # Remainder
    for k in range(num_chunks * chunk_size, num_quads):
        var idx = insert_zero_bit(k, low_pos)
        var idx0 = insert_zero_bit(idx, high_pos)
        var idx1 = idx0 | stride_low
        var idx2 = idx0 | stride_high
        var idx3 = idx1 | stride_high
        var re0 = ptr_re[idx0]
        var im0 = ptr_im[idx0]
        var re1 = ptr_re[idx1]
        var im1 = ptr_im[idx1]
        var re2 = ptr_re[idx2]
        var im2 = ptr_im[idx2]
        var re3 = ptr_re[idx3]
        var im3 = ptr_im[idx3]
        var n_re0 = re0 * mat[0][0].re - im0 * mat[0][0].im
        var n_im0 = re0 * mat[0][0].im + im0 * mat[0][0].re
        n_re0 += re1 * mat[0][1].re - im1 * mat[0][1].im
        n_im0 += re1 * mat[0][1].im + im1 * mat[0][1].re
        n_re0 += re2 * mat[0][2].re - im2 * mat[0][2].im
        n_im0 += re2 * mat[0][2].im + im2 * mat[0][2].re
        n_re0 += re3 * mat[0][3].re - im3 * mat[0][3].im
        n_im0 += re3 * mat[0][3].im + im3 * mat[0][3].re
        var n_re1 = re0 * mat[1][0].re - im0 * mat[1][0].im
        var n_im1 = re0 * mat[1][0].im + im0 * mat[1][0].re
        n_re1 += re1 * mat[1][1].re - im1 * mat[1][1].im
        n_im1 += re1 * mat[1][1].im + im1 * mat[1][1].re
        n_re1 += re2 * mat[1][2].re - im2 * mat[1][2].im
        n_im1 += re2 * mat[1][2].im + im2 * mat[1][2].re
        n_re1 += re3 * mat[1][3].re - im3 * mat[1][3].im
        n_im1 += re3 * mat[1][3].im + im3 * mat[1][3].re
        var n_re2 = re0 * mat[2][0].re - im0 * mat[2][0].im
        var n_im2 = re0 * mat[2][0].im + im0 * mat[2][0].re
        n_re2 += re1 * mat[2][1].re - im1 * mat[2][1].im
        n_im2 += re1 * mat[2][1].im + im1 * mat[2][1].re
        n_re2 += re2 * mat[2][2].re - im2 * mat[2][2].im
        n_im2 += re2 * mat[2][2].im + im2 * mat[2][2].re
        n_re2 += re3 * mat[2][3].re - im3 * mat[2][3].im
        n_im2 += re3 * mat[2][3].im + im3 * mat[2][3].re
        var n_re3 = re0 * mat[3][0].re - im0 * mat[3][0].im
        var n_im3 = re0 * mat[3][0].im + im0 * mat[3][0].re
        n_re3 += re1 * mat[3][1].re - im1 * mat[3][1].im
        n_im3 += re1 * mat[3][1].im + im1 * mat[3][1].re
        n_re3 += re2 * mat[3][2].re - im2 * mat[3][2].im
        n_im3 += re2 * mat[3][2].im + im2 * mat[3][2].re
        n_re3 += re3 * mat[3][3].re - im3 * mat[3][3].im
        n_im3 += re3 * mat[3][3].im + im3 * mat[3][3].re
        ptr_re[idx0] = n_re0
        ptr_im[idx0] = n_im0
        ptr_re[idx1] = n_re1
        ptr_im[idx1] = n_im1
        ptr_re[idx2] = n_re2
        ptr_im[idx2] = n_im2
        ptr_re[idx3] = n_re3
        ptr_im[idx3] = n_im3


fn transform_matrix8(
    mut state: QuantumState,
    target_high: Int,
    target_mid: Int,
    target_low: Int,
    mat: Matrix8x8,
):
    """
    Applies an 8x8 unitary matrix to three qubits.
    """
    var n = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var q_high = max(target_high, max(target_mid, target_low))
    var q_low = min(target_high, min(target_mid, target_low))
    var q_mid = target_high + target_mid + target_low - q_high - q_low

    var stride_low = 1 << q_low
    var stride_mid = 1 << q_mid
    var stride_high = 1 << q_high

    var num_octets = n >> 3

    if stride_low >= simd_width:
        # Broadcasting 64 matrix elements is expensive but avoids loads in the loop.
        # However, for brevity and to avoid register pressure, let's see.
        # Actually, let's do it for performance.

        # We'll use a macro-like approach for the matrix mul part to keep it clean.

        var num_outer = n >> (q_high + 1)
        var num_mid_level = 1 << (q_high - q_mid - 1)
        var num_low_level = 1 << (q_mid - q_low - 1)
        var num_inner = 1 << q_low

        @parameter
        fn worker_simd(outer_idx: Int):
            var base_outer = outer_idx << (q_high + 1)
            for m in range(num_mid_level):
                var base_mid = base_outer | (m << (q_mid + 1))
                for l in range(num_low_level):
                    var base_low = base_mid | (l << (q_low + 1))

                    @parameter
                    fn inner_simd[w: Int](i: Int):
                        var idx0 = base_low | i
                        var idx1 = idx0 | stride_low
                        var idx2 = idx0 | stride_mid
                        var idx3 = idx1 | stride_mid
                        var idx4 = idx0 | stride_high
                        var idx5 = idx1 | stride_high
                        var idx6 = idx2 | stride_high
                        var idx7 = idx3 | stride_high

                        # Loads
                        var re0 = ptr_re.load[width=w](idx0)
                        var im0 = ptr_im.load[width=w](idx0)
                        var re1 = ptr_re.load[width=w](idx1)
                        var im1 = ptr_im.load[width=w](idx1)
                        var re2 = ptr_re.load[width=w](idx2)
                        var im2 = ptr_im.load[width=w](idx2)
                        var re3 = ptr_re.load[width=w](idx3)
                        var im3 = ptr_im.load[width=w](idx3)
                        var re4 = ptr_re.load[width=w](idx4)
                        var im4 = ptr_im.load[width=w](idx4)
                        var re5 = ptr_re.load[width=w](idx5)
                        var im5 = ptr_im.load[width=w](idx5)
                        var re6 = ptr_re.load[width=w](idx6)
                        var im6 = ptr_im.load[width=w](idx6)
                        var re7 = ptr_re.load[width=w](idx7)
                        var im7 = ptr_im.load[width=w](idx7)

                        for row in range(8):
                            var n_re: SIMD[DType.float64, w] = 0
                            var n_im: SIMD[DType.float64, w] = 0

                            # Row-column dot products
                            # (re+i*im) * (m_re+i*m_im) = (re*m_re - im*m_im) + i*(re*m_im + im*m_re)

                            # Row 0..7 dot avec re0..7/im0..7
                            # This loop unrolling is handled by Mojo's compiler

                            var m_re0 = mat[row][0].re
                            var m_im0 = mat[row][0].im
                            n_re += re0 * m_re0 - im0 * m_im0
                            n_im += re0 * m_im0 + im0 * m_re0

                            var m_re1 = mat[row][1].re
                            var m_im1 = mat[row][1].im
                            n_re += re1 * m_re1 - im1 * m_im1
                            n_im += re1 * m_im1 + im1 * m_re1

                            var m_re2 = mat[row][2].re
                            var m_im2 = mat[row][2].im
                            n_re += re2 * m_re2 - im2 * m_im2
                            n_im += re2 * m_im2 + im2 * m_re2

                            var m_re3 = mat[row][3].re
                            var m_im3 = mat[row][3].im
                            n_re += re3 * m_re3 - im3 * m_im3
                            n_im += re3 * m_im3 + im3 * m_re3

                            var m_re4 = mat[row][4].re
                            var m_im4 = mat[row][4].im
                            n_re += re4 * m_re4 - im4 * m_im4
                            n_im += re4 * m_im4 + im4 * m_re4

                            var m_re5 = mat[row][5].re
                            var m_im5 = mat[row][5].im
                            n_re += re5 * m_re5 - im5 * m_im5
                            n_im += re5 * m_im5 + im5 * m_re5

                            var m_re6 = mat[row][6].re
                            var m_im6 = mat[row][6].im
                            n_re += re6 * m_re6 - im6 * m_im6
                            n_im += re6 * m_im6 + im6 * m_re6

                            var m_re7 = mat[row][7].re
                            var m_im7 = mat[row][7].im
                            n_re += re7 * m_re7 - im7 * m_im7
                            n_im += re7 * m_im7 + im7 * m_re7

                            if row == 0:
                                ptr_re.store(idx0, n_re)
                                ptr_im.store(idx0, n_im)
                            elif row == 1:
                                ptr_re.store(idx1, n_re)
                                ptr_im.store(idx1, n_im)
                            elif row == 2:
                                ptr_re.store(idx2, n_re)
                                ptr_im.store(idx2, n_im)
                            elif row == 3:
                                ptr_re.store(idx3, n_re)
                                ptr_im.store(idx3, n_im)
                            elif row == 4:
                                ptr_re.store(idx4, n_re)
                                ptr_im.store(idx4, n_im)
                            elif row == 5:
                                ptr_re.store(idx5, n_re)
                                ptr_im.store(idx5, n_im)
                            elif row == 6:
                                ptr_re.store(idx6, n_re)
                                ptr_im.store(idx6, n_im)
                            elif row == 7:
                                ptr_re.store(idx7, n_re)
                                ptr_im.store(idx7, n_im)

                    vectorize[inner_simd, simd_width](num_inner)

        parallelize[worker_simd](num_outer)
        return

    # Fallback (Scalar)
    alias chunk_size = 256
    var num_chunks = num_octets // chunk_size

    @parameter
    fn worker(chunk_idx: Int):
        var start = chunk_idx * chunk_size
        var end = start + chunk_size
        for k in range(start, end):
            var idx = insert_zero_bit(k, q_low)
            idx = insert_zero_bit(idx, q_mid)
            var idx0 = insert_zero_bit(idx, q_high)

            var idx1 = idx0 | stride_low
            var idx2 = idx0 | stride_mid
            var idx3 = idx1 | stride_mid
            var idx4 = idx0 | stride_high
            var idx5 = idx1 | stride_high
            var idx6 = idx2 | stride_high
            var idx7 = idx3 | stride_high

            var re0 = ptr_re[idx0]
            var im0 = ptr_im[idx0]
            var re1 = ptr_re[idx1]
            var im1 = ptr_im[idx1]
            var re2 = ptr_re[idx2]
            var im2 = ptr_im[idx2]
            var re3 = ptr_re[idx3]
            var im3 = ptr_im[idx3]
            var re4 = ptr_re[idx4]
            var im4 = ptr_im[idx4]
            var re5 = ptr_re[idx5]
            var im5 = ptr_im[idx5]
            var re6 = ptr_re[idx6]
            var im6 = ptr_im[idx6]
            var re7 = ptr_re[idx7]
            var im7 = ptr_im[idx7]

            for row in range(8):
                var sum_re: FloatType = 0
                var sum_im: FloatType = 0

                # Row 0 dot
                var m_re = mat[row][0].re
                var m_im = mat[row][0].im
                sum_re += re0 * m_re - im0 * m_im
                sum_im += re0 * m_im + im0 * m_re

                m_re = mat[row][1].re
                m_im = mat[row][1].im
                sum_re += re1 * m_re - im1 * m_im
                sum_im += re1 * m_im + im1 * m_re

                m_re = mat[row][2].re
                m_im = mat[row][2].im
                sum_re += re2 * m_re - im2 * m_im
                sum_im += re2 * m_im + im2 * m_re

                m_re = mat[row][3].re
                m_im = mat[row][3].im
                sum_re += re3 * m_re - im3 * m_im
                sum_im += re3 * m_im + im3 * m_re

                m_re = mat[row][4].re
                m_im = mat[row][4].im
                sum_re += re4 * m_re - im4 * m_im
                sum_im += re4 * m_im + im4 * m_re

                m_re = mat[row][5].re
                m_im = mat[row][5].im
                sum_re += re5 * m_re - im5 * m_im
                sum_im += re5 * m_im + im5 * m_re

                m_re = mat[row][6].re
                m_im = mat[row][6].im
                sum_re += re6 * m_re - im6 * m_im
                sum_im += re6 * m_im + im6 * m_re

                m_re = mat[row][7].re
                m_im = mat[row][7].im
                sum_re += re7 * m_re - im7 * m_im
                sum_im += re7 * m_im + im7 * m_re

                if row == 0:
                    ptr_re[idx0] = sum_re
                    ptr_im[idx0] = sum_im
                elif row == 1:
                    ptr_re[idx1] = sum_re
                    ptr_im[idx1] = sum_im
                elif row == 2:
                    ptr_re[idx2] = sum_re
                    ptr_im[idx2] = sum_im
                elif row == 3:
                    ptr_re[idx3] = sum_re
                    ptr_im[idx3] = sum_im
                elif row == 4:
                    ptr_re[idx4] = sum_re
                    ptr_im[idx4] = sum_im
                elif row == 5:
                    ptr_re[idx5] = sum_re
                    ptr_im[idx5] = sum_im
                elif row == 6:
                    ptr_re[idx6] = sum_re
                    ptr_im[idx6] = sum_im
                elif row == 7:
                    ptr_re[idx7] = sum_re
                    ptr_im[idx7] = sum_im

    parallelize[worker](num_chunks)
    for k in range(num_chunks * chunk_size, num_octets):
        var idx = insert_zero_bit(k, q_low)
        idx = insert_zero_bit(idx, q_mid)
        var idx0 = insert_zero_bit(idx, q_high)
        var idx1 = idx0 | stride_low
        var idx2 = idx0 | stride_mid
        var idx3 = idx1 | stride_mid
        var idx4 = idx0 | stride_high
        var idx5 = idx1 | stride_high
        var idx6 = idx2 | stride_high
        var idx7 = idx3 | stride_high

        var re0 = ptr_re[idx0]
        var im0 = ptr_im[idx0]
        var re1 = ptr_re[idx1]
        var im1 = ptr_im[idx1]
        var re2 = ptr_re[idx2]
        var im2 = ptr_im[idx2]
        var re3 = ptr_re[idx3]
        var im3 = ptr_im[idx3]
        var re4 = ptr_re[idx4]
        var im4 = ptr_im[idx4]
        var re5 = ptr_re[idx5]
        var im5 = ptr_im[idx5]
        var re6 = ptr_re[idx6]
        var im6 = ptr_im[idx6]
        var re7 = ptr_re[idx7]
        var im7 = ptr_im[idx7]

        for row in range(8):
            var sum_re: FloatType = 0
            var sum_im: FloatType = 0

            var m_re = mat[row][0].re
            var m_im = mat[row][0].im
            sum_re += re0 * m_re - im0 * m_im
            sum_im += re0 * m_im + im0 * m_re

            m_re = mat[row][1].re
            m_im = mat[row][1].im
            sum_re += re1 * m_re - im1 * m_im
            sum_im += re1 * m_im + im1 * m_re

            m_re = mat[row][2].re
            m_im = mat[row][2].im
            sum_re += re2 * m_re - im2 * m_im
            sum_im += re2 * m_im + im2 * m_re

            m_re = mat[row][3].re
            m_im = mat[row][3].im
            sum_re += re3 * m_re - im3 * m_im
            sum_im += re3 * m_im + im3 * m_re

            m_re = mat[row][4].re
            m_im = mat[row][4].im
            sum_re += re4 * m_re - im4 * m_im
            sum_im += re4 * m_im + im4 * m_re

            m_re = mat[row][5].re
            m_im = mat[row][5].im
            sum_re += re5 * m_re - im5 * m_im
            sum_im += re5 * m_im + im5 * m_re

            m_re = mat[row][6].re
            m_im = mat[row][6].im
            sum_re += re6 * m_re - im6 * m_im
            sum_im += re6 * m_im + im6 * m_re

            m_re = mat[row][7].re
            m_im = mat[row][7].im
            sum_re += re7 * m_re - im7 * m_im
            sum_im += re7 * m_im + im7 * m_re

            if row == 0:
                ptr_re[idx0] = sum_re
                ptr_im[idx0] = sum_im
            elif row == 1:
                ptr_re[idx1] = sum_re
                ptr_im[idx1] = sum_im
            elif row == 2:
                ptr_re[idx2] = sum_re
                ptr_im[idx2] = sum_im
            elif row == 3:
                ptr_re[idx3] = sum_re
                ptr_im[idx3] = sum_im
            elif row == 4:
                ptr_re[idx4] = sum_re
                ptr_im[idx4] = sum_im
            elif row == 5:
                ptr_re[idx5] = sum_re
                ptr_im[idx5] = sum_im
            elif row == 6:
                ptr_re[idx6] = sum_re
                ptr_im[idx6] = sum_im
            elif row == 7:
                ptr_re[idx7] = sum_re
                ptr_im[idx7] = sum_im


fn transform_matrix16(
    mut state: QuantumState,
    q3: Int,
    q2: Int,
    q1: Int,
    q0: Int,
    mat: Matrix16x16,
):
    """
    Applies a 16x16 unitary matrix to four qubits.
    q3 > q2 > q1 > q0.
    """
    var n = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var stride0 = 1 << q0
    var stride1 = 1 << q1
    var stride2 = 1 << q2
    var stride3 = 1 << q3

    var num_hex = n >> 4

    if stride0 >= simd_width:
        var num_outer = n >> (q3 + 1)
        var n32 = 1 << (q3 - q2 - 1)
        var n21 = 1 << (q2 - q1 - 1)
        var n10 = 1 << (q1 - q0 - 1)
        var num_inner = 1 << q0

        @parameter
        fn worker_simd(outer_idx: Int):
            var base_outer = outer_idx << (q3 + 1)
            for i3 in range(n32):
                var base3 = base_outer | (i3 << (q2 + 1))
                for i2 in range(n21):
                    var base2 = base3 | (i2 << (q1 + 1))
                    for i1 in range(n10):
                        var base1 = base2 | (i1 << (q0 + 1))

                        @parameter
                        fn inner_simd[w: Int](i: Int):
                            var idx0 = base1 | i
                            # 16 indices
                            var idx1 = idx0 | stride0
                            var idx2 = idx0 | stride1
                            var idx3 = idx1 | stride1
                            var idx4 = idx0 | stride2
                            var idx5 = idx1 | stride2
                            var idx6 = idx2 | stride2
                            var idx7 = idx3 | stride2
                            var idx8 = idx0 | stride3
                            var idx9 = idx1 | stride3
                            var idx10 = idx2 | stride3
                            var idx11 = idx3 | stride3
                            var idx12 = idx4 | stride3
                            var idx13 = idx5 | stride3
                            var idx14 = idx6 | stride3
                            var idx15 = idx7 | stride3

                            # Process rows in batches of 4 to avoid register pressure
                            for batch in range(4):
                                var r0 = batch * 4
                                var r1 = r0 + 1
                                var r2 = r0 + 2
                                var r3 = r0 + 3

                                var n_re0: SIMD[DType.float64, w] = 0
                                var n_im0: SIMD[DType.float64, w] = 0
                                var n_re1: SIMD[DType.float64, w] = 0
                                var n_im1: SIMD[DType.float64, w] = 0
                                var n_re2: SIMD[DType.float64, w] = 0
                                var n_im2: SIMD[DType.float64, w] = 0
                                var n_re3: SIMD[DType.float64, w] = 0
                                var n_im3: SIMD[DType.float64, w] = 0

                                # Part 1: First 8 inputs
                                var re0 = ptr_re.load[width=w](idx0)
                                var im0 = ptr_im.load[width=w](idx0)
                                var re1 = ptr_re.load[width=w](idx1)
                                var im1 = ptr_im.load[width=w](idx1)
                                var re2 = ptr_re.load[width=w](idx2)
                                var im2 = ptr_im.load[width=w](idx2)
                                var re3 = ptr_re.load[width=w](idx3)
                                var im3 = ptr_im.load[width=w](idx3)
                                var re4 = ptr_re.load[width=w](idx4)
                                var im4 = ptr_im.load[width=w](idx4)
                                var re5 = ptr_re.load[width=w](idx5)
                                var im5 = ptr_im.load[width=w](idx5)
                                var re6 = ptr_re.load[width=w](idx6)
                                var im6 = ptr_im.load[width=w](idx6)
                                var re7 = ptr_re.load[width=w](idx7)
                                var im7 = ptr_im.load[width=w](idx7)

                                acc_mul(n_re0, n_im0, mat[r0][0], re0, im0)
                                acc_mul(n_re0, n_im0, mat[r0][1], re1, im1)
                                acc_mul(n_re0, n_im0, mat[r0][2], re2, im2)
                                acc_mul(n_re0, n_im0, mat[r0][3], re3, im3)
                                acc_mul(n_re0, n_im0, mat[r0][4], re4, im4)
                                acc_mul(n_re0, n_im0, mat[r0][5], re5, im5)
                                acc_mul(n_re0, n_im0, mat[r0][6], re6, im6)
                                acc_mul(n_re0, n_im0, mat[r0][7], re7, im7)

                                acc_mul(n_re1, n_im1, mat[r1][0], re0, im0)
                                acc_mul(n_re1, n_im1, mat[r1][1], re1, im1)
                                acc_mul(n_re1, n_im1, mat[r1][2], re2, im2)
                                acc_mul(n_re1, n_im1, mat[r1][3], re3, im3)
                                acc_mul(n_re1, n_im1, mat[r1][4], re4, im4)
                                acc_mul(n_re1, n_im1, mat[r1][5], re5, im5)
                                acc_mul(n_re1, n_im1, mat[r1][6], re6, im6)
                                acc_mul(n_re1, n_im1, mat[r1][7], re7, im7)

                                acc_mul(n_re2, n_im2, mat[r2][0], re0, im0)
                                acc_mul(n_re2, n_im2, mat[r2][1], re1, im1)
                                acc_mul(n_re2, n_im2, mat[r2][2], re2, im2)
                                acc_mul(n_re2, n_im2, mat[r2][3], re3, im3)
                                acc_mul(n_re2, n_im2, mat[r2][4], re4, im4)
                                acc_mul(n_re2, n_im2, mat[r2][5], re5, im5)
                                acc_mul(n_re2, n_im2, mat[r2][6], re6, im6)
                                acc_mul(n_re2, n_im2, mat[r2][7], re7, im7)

                                acc_mul(n_re3, n_im3, mat[r3][0], re0, im0)
                                acc_mul(n_re3, n_im3, mat[r3][1], re1, im1)
                                acc_mul(n_re3, n_im3, mat[r3][2], re2, im2)
                                acc_mul(n_re3, n_im3, mat[r3][3], re3, im3)
                                acc_mul(n_re3, n_im3, mat[r3][4], re4, im4)
                                acc_mul(n_re3, n_im3, mat[r3][5], re5, im5)
                                acc_mul(n_re3, n_im3, mat[r3][6], re6, im6)
                                acc_mul(n_re3, n_im3, mat[r3][7], re7, im7)

                                # Part 2: Next 8 inputs
                                var re8 = ptr_re.load[width=w](idx8)
                                var im8 = ptr_im.load[width=w](idx8)
                                var re9 = ptr_re.load[width=w](idx9)
                                var im9 = ptr_im.load[width=w](idx9)
                                var re10 = ptr_re.load[width=w](idx10)
                                var im10 = ptr_im.load[width=w](idx10)
                                var re11 = ptr_re.load[width=w](idx11)
                                var im11 = ptr_im.load[width=w](idx11)
                                var re12 = ptr_re.load[width=w](idx12)
                                var im12 = ptr_im.load[width=w](idx12)
                                var re13 = ptr_re.load[width=w](idx13)
                                var im13 = ptr_im.load[width=w](idx13)
                                var re14 = ptr_re.load[width=w](idx14)
                                var im14 = ptr_im.load[width=w](idx14)
                                var re15 = ptr_re.load[width=w](idx15)
                                var im15 = ptr_im.load[width=w](idx15)

                                acc_mul(n_re0, n_im0, mat[r0][8], re8, im8)
                                acc_mul(n_re0, n_im0, mat[r0][9], re9, im9)
                                acc_mul(n_re0, n_im0, mat[r0][10], re10, im10)
                                acc_mul(n_re0, n_im0, mat[r0][11], re11, im11)
                                acc_mul(n_re0, n_im0, mat[r0][12], re12, im12)
                                acc_mul(n_re0, n_im0, mat[r0][13], re13, im13)
                                acc_mul(n_re0, n_im0, mat[r0][14], re14, im14)
                                acc_mul(n_re0, n_im0, mat[r0][15], re15, im15)

                                acc_mul(n_re1, n_im1, mat[r1][8], re8, im8)
                                acc_mul(n_re1, n_im1, mat[r1][9], re9, im9)
                                acc_mul(n_re1, n_im1, mat[r1][10], re10, im10)
                                acc_mul(n_re1, n_im1, mat[r1][11], re11, im11)
                                acc_mul(n_re1, n_im1, mat[r1][12], re12, im12)
                                acc_mul(n_re1, n_im1, mat[r1][13], re13, im13)
                                acc_mul(n_re1, n_im1, mat[r1][14], re14, im14)
                                acc_mul(n_re1, n_im1, mat[r1][15], re15, im15)

                                acc_mul(n_re2, n_im2, mat[r2][8], re8, im8)
                                acc_mul(n_re2, n_im2, mat[r2][9], re9, im9)
                                acc_mul(n_re2, n_im2, mat[r2][10], re10, im10)
                                acc_mul(n_re2, n_im2, mat[r2][11], re11, im11)
                                acc_mul(n_re2, n_im2, mat[r2][12], re12, im12)
                                acc_mul(n_re2, n_im2, mat[r2][13], re13, im13)
                                acc_mul(n_re2, n_im2, mat[r2][14], re14, im14)
                                acc_mul(n_re2, n_im2, mat[r2][15], re15, im15)

                                acc_mul(n_re3, n_im3, mat[r3][8], re8, im8)
                                acc_mul(n_re3, n_im3, mat[r3][9], re9, im9)
                                acc_mul(n_re3, n_im3, mat[r3][10], re10, im10)
                                acc_mul(n_re3, n_im3, mat[r3][11], re11, im11)
                                acc_mul(n_re3, n_im3, mat[r3][12], re12, im12)
                                acc_mul(n_re3, n_im3, mat[r3][13], re13, im13)
                                acc_mul(n_re3, n_im3, mat[r3][14], re14, im14)
                                acc_mul(n_re3, n_im3, mat[r3][15], re15, im15)

                                # Store results for this batch
                                if batch == 0:
                                    ptr_re.store(idx0, n_re0)
                                    ptr_im.store(idx0, n_im0)
                                    ptr_re.store(idx1, n_re1)
                                    ptr_im.store(idx1, n_im1)
                                    ptr_re.store(idx2, n_re2)
                                    ptr_im.store(idx2, n_im2)
                                    ptr_re.store(idx3, n_re3)
                                    ptr_im.store(idx3, n_im3)
                                elif batch == 1:
                                    ptr_re.store(idx4, n_re0)
                                    ptr_im.store(idx4, n_im0)
                                    ptr_re.store(idx5, n_re1)
                                    ptr_im.store(idx5, n_im1)
                                    ptr_re.store(idx6, n_re2)
                                    ptr_im.store(idx6, n_im2)
                                    ptr_re.store(idx7, n_re3)
                                    ptr_im.store(idx7, n_im3)
                                elif batch == 2:
                                    ptr_re.store(idx8, n_re0)
                                    ptr_im.store(idx8, n_im0)
                                    ptr_re.store(idx9, n_re1)
                                    ptr_im.store(idx9, n_im1)
                                    ptr_re.store(idx10, n_re2)
                                    ptr_im.store(idx10, n_im2)
                                    ptr_re.store(idx11, n_re3)
                                    ptr_im.store(idx11, n_im3)
                                else:
                                    ptr_re.store(idx12, n_re0)
                                    ptr_im.store(idx12, n_im0)
                                    ptr_re.store(idx13, n_re1)
                                    ptr_im.store(idx13, n_im1)
                                    ptr_re.store(idx14, n_re2)
                                    ptr_im.store(idx14, n_im2)
                                    ptr_re.store(idx15, n_re3)
                                    ptr_im.store(idx15, n_im3)

                        vectorize[inner_simd, simd_width](num_inner)

        parallelize[worker_simd](num_outer)
        return

    # Fallback (Scalar/Slow)
    for k in range(num_hex):
        var idx = insert_zero_bit(k, q0)
        idx = insert_zero_bit(idx, q1)
        idx = insert_zero_bit(idx, q2)
        var idx0 = insert_zero_bit(idx, q3)

        # 16 indices
        var indices = InlineArray[Int, 16](
            idx0,
            idx0 | stride0,
            idx0 | stride1,
            idx0 | stride0 | stride1,
            idx0 | stride2,
            idx0 | stride0 | stride2,
            idx0 | stride1 | stride2,
            idx0 | stride0 | stride1 | stride2,
            idx0 | stride3,
            idx0 | stride0 | stride3,
            idx0 | stride1 | stride3,
            idx0 | stride0 | stride1 | stride3,
            idx0 | stride2 | stride3,
            idx0 | stride0 | stride2 | stride3,
            idx0 | stride1 | stride2 | stride3,
            idx0 | stride0 | stride1 | stride2 | stride3,
        )

        var re = InlineArray[FloatType, 16](
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        var im = InlineArray[FloatType, 16](
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        for i in range(16):
            re[i] = ptr_re[indices[i]]
            im[i] = ptr_im[indices[i]]

        for r in range(16):
            var s_re: FloatType = 0
            var s_im: FloatType = 0
            for c in range(16):
                s_re += re[c] * mat[r][c].re - im[c] * mat[r][c].im
                s_im += re[c] * mat[r][c].im + im[c] * mat[r][c].re
            ptr_re[indices[r]] = s_re
            ptr_im[indices[r]] = s_im
