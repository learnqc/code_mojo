from complex import ComplexSIMD
from sys.info import simd_width_of
from sys import is_defined
import math


comptime Type = DType.float32 if is_defined["FLOAT32"]() else DType.float64
alias FloatType = Scalar[Type]

alias simd_width = simd_width_of[Type]()

alias Complex = ComplexSIMD[Type, 1]
alias Amplitude = Complex

alias Gate = InlineArray[InlineArray[Complex, 2], 2]

alias pi: FloatType = math.pi

alias `0`: Complex = Complex(0, 0)
alias `1`: Complex = Complex(1, 0)
alias `i`: Complex = Complex(0, 1)

alias sq_half_re: FloatType = math.sqrt(0.5).cast[Type]()
alias sq_half: Complex = Complex(sq_half_re, 0)
alias sq2: Complex = Complex(math.sqrt(2.0).cast[Type](), 0)
