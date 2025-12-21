from butterfly.core.types import FloatType, Type, Amplitude, Gate
from sys.info import simd_width_of

alias simd_width = simd_width_of[Type]()
from memory import UnsafePointer
from algorithm import vectorize

alias Matrix4x4 = InlineArray[InlineArray[Amplitude, 4], 4]
alias Matrix8x8 = InlineArray[InlineArray[Amplitude, 8], 8]
alias Matrix16x16 = InlineArray[InlineArray[Amplitude, 16], 16]


fn compute_kron_product(u: Gate) -> Matrix4x4:
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
                    m[row][col] = u[i][j] * u[k][l]
    return m


@always_inline
fn matmul_matrix4x4(m1: Matrix4x4, m2: Matrix4x4) -> Matrix4x4:
    """Computes m1 * m2."""
    var res = Matrix4x4(
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
    for i in range(4):
        for j in range(4):
            var sum_re: FloatType = 0
            var sum_im: FloatType = 0
            for k in range(4):
                sum_re += m1[i][k].re * m2[k][j].re - m1[i][k].im * m2[k][j].im
                sum_im += m1[i][k].re * m2[k][j].im + m1[i][k].im * m2[k][j].re
            res[i][j] = Amplitude(sum_re, sum_im)
    return res


@always_inline
fn matmul_matrix8x8(m1: Matrix8x8, m2: Matrix8x8) -> Matrix8x8:
    """Computes m1 * m2."""
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
    for i in range(8):
        for j in range(8):
            var sum_re: FloatType = 0
            var sum_im: FloatType = 0
            for k in range(8):
                sum_re += m1[i][k].re * m2[k][j].re - m1[i][k].im * m2[k][j].im
                sum_im += m1[i][k].re * m2[k][j].im + m1[i][k].im * m2[k][j].re
            res[i][j] = Amplitude(sum_re, sum_im)
    return res


@always_inline
fn matmul_matrix16x16(m1: Matrix16x16, m2: Matrix16x16) -> Matrix16x16:
    """Computes m1 * m2."""
    var row = InlineArray[Amplitude, 16](
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
    )
    var res = Matrix16x16(
        row,
        row,
        row,
        row,
        row,
        row,
        row,
        row,
        row,
        row,
        row,
        row,
        row,
        row,
        row,
        row,
    )
    for i in range(16):
        for j in range(16):
            var sum_re: FloatType = 0
            var sum_im: FloatType = 0
            for k in range(16):
                sum_re += m1[i][k].re * m2[k][j].re - m1[i][k].im * m2[k][j].im
                sum_im += m1[i][k].re * m2[k][j].im + m1[i][k].im * m2[k][j].re
            res[i][j] = Amplitude(sum_re, sum_im)
    return res


@always_inline
fn acc_mul[
    w: Int
](
    mut acc_re: SIMD[DType.float64, w],
    mut acc_im: SIMD[DType.float64, w],
    scalar: Amplitude,
    vec_re: SIMD[DType.float64, w],
    vec_im: SIMD[DType.float64, w],
):
    var a = scalar.re
    var b = scalar.im
    var c = vec_re
    var d = vec_im
    acc_re += a * c - b * d
    acc_im += a * d + b * c


fn unitary_radix4_kernel(
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    M: Matrix4x4,
    start: Int,
    stride: Int,
):
    """
    Applies the 4x4 unitary matrix M to the 4 vectors at:
    start, start+stride, start+2*stride, start+3*stride.
    Processing 'stride' elements (vectorized).
    """
    var p_re = UnsafePointer[FloatType](ptr_re.address)
    var p_im = UnsafePointer[FloatType](ptr_im.address)

    @parameter
    fn v_kernel[w: Int](idx: Int):
        var k = idx
        var idx0 = start + k
        var idx1 = start + stride + k
        var idx2 = start + 2 * stride + k
        var idx3 = start + 3 * stride + k

        var re0 = p_re.load[width=w](idx0)
        var im0 = p_im.load[width=w](idx0)
        var re1 = p_re.load[width=w](idx1)
        var im1 = p_im.load[width=w](idx1)
        var re2 = p_re.load[width=w](idx2)
        var im2 = p_im.load[width=w](idx2)
        var re3 = p_re.load[width=w](idx3)
        var im3 = p_im.load[width=w](idx3)

        var z_re0 = SIMD[DType.float64, w](0.0)
        var z_im0 = SIMD[DType.float64, w](0.0)
        var z_re1 = SIMD[DType.float64, w](0.0)
        var z_im1 = SIMD[DType.float64, w](0.0)
        var z_re2 = SIMD[DType.float64, w](0.0)
        var z_im2 = SIMD[DType.float64, w](0.0)
        var z_re3 = SIMD[DType.float64, w](0.0)
        var z_im3 = SIMD[DType.float64, w](0.0)

        acc_mul(z_re0, z_im0, M[0][0], re0, im0)
        acc_mul(z_re0, z_im0, M[0][1], re1, im1)
        acc_mul(z_re0, z_im0, M[0][2], re2, im2)
        acc_mul(z_re0, z_im0, M[0][3], re3, im3)

        acc_mul(z_re1, z_im1, M[1][0], re0, im0)
        acc_mul(z_re1, z_im1, M[1][1], re1, im1)
        acc_mul(z_re1, z_im1, M[1][2], re2, im2)
        acc_mul(z_re1, z_im1, M[1][3], re3, im3)

        acc_mul(z_re2, z_im2, M[2][0], re0, im0)
        acc_mul(z_re2, z_im2, M[2][1], re1, im1)
        acc_mul(z_re2, z_im2, M[2][2], re2, im2)
        acc_mul(z_re2, z_im2, M[2][3], re3, im3)

        acc_mul(z_re3, z_im3, M[3][0], re0, im0)
        acc_mul(z_re3, z_im3, M[3][1], re1, im1)
        acc_mul(z_re3, z_im3, M[3][2], re2, im2)
        acc_mul(z_re3, z_im3, M[3][3], re3, im3)

        p_re.store(idx0, z_re0)
        p_im.store(idx0, z_im0)
        p_re.store(idx1, z_re1)
        p_im.store(idx1, z_im1)
        p_re.store(idx2, z_re2)
        p_im.store(idx2, z_im2)
        p_re.store(idx3, z_re3)
        p_im.store(idx3, z_im3)

    vectorize[v_kernel, simd_width](stride)
