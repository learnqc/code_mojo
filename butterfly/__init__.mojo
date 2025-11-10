alias Type = DType.float32
alias FloatType = SIMD[Type, 1]

from complex import ComplexSIMD
alias Amplitude = ComplexSIMD[Type, 1]
alias State = List[Amplitude]
alias ArrayState = InlineArray[Amplitude]
alias GridState = List[State]

alias Gate = InlineArray[InlineArray[Amplitude, 2], 2]

import math
alias pi: FloatType = math.pi

alias `0`: Amplitude = Amplitude(0, 0)
alias `1`: Amplitude = Amplitude(1, 0)
alias `i`: Amplitude = Amplitude(0, 1)

alias sq2: Amplitude = Amplitude(math.sqrt(0.5).cast[Type](), 0)