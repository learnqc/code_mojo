from math import cos, sin, log2, log10, atan2, floor
from butterfly.core.types import *


fn cis(theta: FloatType) -> Amplitude:
    return Amplitude(cos(theta), sin(theta))


alias X: Gate = [[`0`, `1`], [`1`, `0`]]
alias Z: Gate = [[`1`, `0`], [`0`, -`1`]]


fn P(theta: FloatType) -> Gate:
    gate: Gate = [[`1`, `0`], [`0`, cis(theta)]]
    return gate^


alias H: Gate = [[sq_half, sq_half], [sq_half, -sq_half]]


fn RZ(theta: FloatType) -> Gate:
    gate: Gate = [
        [Amplitude(cos(theta / 2), -sin(theta / 2)), `0`],
        [`0`, Amplitude(cos(theta / 2), sin(theta / 2))],
    ]
    return gate^


alias Y: Gate = [[`0`, -`i`], [`i`, `0`]]


fn RX(theta: FloatType) -> Gate:
    gate: Gate = [
        [Amplitude(cos(theta / 2), 0), Amplitude(0, -sin(theta / 2))],
        [Amplitude(0, -sin(theta / 2)), Amplitude(cos(theta / 2), 0)],
    ]
    return gate^


fn RY(theta: FloatType) -> Gate:
    gate: Gate = [
        [Amplitude(cos(theta / 2), 0), Amplitude(-sin(theta / 2), 0)],
        [Amplitude(sin(theta / 2), 0), Amplitude(cos(theta / 2), 0)],
    ]
    return gate^
