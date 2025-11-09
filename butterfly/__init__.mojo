alias Type = DType.float32
alias FloatType = SIMD[Type, 1]

from complex import ComplexSIMD
alias Amplitude = ComplexSIMD[Type, 1]
alias State = List[Amplitude]
alias ArrayState = InlineArray[Amplitude]

import math
alias pi: FloatType = math.pi