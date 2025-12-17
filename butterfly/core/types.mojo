from complex import ComplexSIMD
import math

alias Type = DType.float64
alias FloatType = Scalar[Type]
alias Amplitude = ComplexSIMD[Type, 1]

alias Gate = InlineArray[InlineArray[Amplitude, 2], 2]

alias pi: FloatType = math.pi

alias `0`: Amplitude = Amplitude(0, 0)
alias `1`: Amplitude = Amplitude(1, 0)
alias `i`: Amplitude = Amplitude(0, 1)

alias sq_half: Amplitude = Amplitude(math.sqrt(0.5).cast[Type](), 0)
alias sq2: Amplitude = Amplitude(math.sqrt(2.0).cast[Type](), 0)

alias GridState = List[List[Amplitude]]
alias ArrayState = InlineArray[Amplitude]
