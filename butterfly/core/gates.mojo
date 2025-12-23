from math import cos, sin, log2, log10, atan2, floor
from butterfly.core.types import *
from collections import InlineArray


@always_inline
fn is_bit_set(mask: Int, i: Int) -> Bool:
    """Return True if the i-th bit of mask is set."""
    return (mask >> i) & 1 == 1


@always_inline
fn is_bit_not_set(mask: Int, i: Int) -> Bool:
    """Return True if the i-th bit of mask is not set."""
    return (mask >> i) & 1 == 0


fn cis(theta: FloatType) -> Amplitude:
    return Amplitude(cos(theta), sin(theta))


alias X: Gate = [[`0`, `1`], [`1`, `0`]]
alias Y: Gate = [[`0`, Amplitude(0, -1)], [Amplitude(0, 1), `0`]]
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


fn U3(theta: FloatType, phi: FloatType, lam: FloatType) -> Gate:
    gate: Gate = [
        [
            Amplitude(cos(theta / 2), 0),
            -cis(lam) * Amplitude(sin(theta / 2), 0),
        ],
        [
            cis(phi) * Amplitude(sin(theta / 2), 0),
            cis(phi + lam) * Amplitude(cos(theta / 2), 0),
        ],
    ]
    return gate^


fn U2(phi: FloatType, lam: FloatType) -> Gate:
    return U3(pi / 2, phi, lam)


fn U1(lam: FloatType) -> Gate:
    return U3(0, 0, lam)


fn CX(control: Int, target: Int) -> Gate:
    return X


fn CZ(control: Int, target: Int) -> Gate:
    return Z


fn CP(theta: FloatType, control: Int, target: Int) -> Gate:
    return P(theta)


fn CR(theta: FloatType, control: Int, target: Int) -> Gate:
    return RZ(theta)


fn CRX(theta: FloatType, control: Int, target: Int) -> Gate:
    return RX(theta)


fn CRY(theta: FloatType, control: Int, target: Int) -> Gate:
    return RY(theta)


fn CRZ(theta: FloatType, control: Int, target: Int) -> Gate:
    return RZ(theta)


@always_inline
fn is_h(g: Gate) -> Bool:
    """Returns True if the gate is Hadamard."""
    var sq_half_val = sq_half.re[0]
    return (
        (g[0][0].re[0] - sq_half_val) ** 2 < 1e-24
        and (g[0][1].re[0] - sq_half_val) ** 2 < 1e-24
        and (g[1][0].re[0] - sq_half_val) ** 2 < 1e-24
        and (g[1][1].re[0] + sq_half_val) ** 2 < 1e-24
        and g[0][0].im[0] ** 2 < 1e-24
        and g[0][1].im[0] ** 2 < 1e-24
        and g[1][0].im[0] ** 2 < 1e-24
        and g[1][1].im[0] ** 2 < 1e-24
    )


@always_inline
fn is_z(g: Gate) -> Bool:
    """Returns True if the gate is Pauli-Z."""
    return (
        abs(g[0][0].re[0] - 1.0) < 1e-24
        and abs(g[0][0].im[0]) < 1e-24
        and abs(g[0][1].re[0]) < 1e-24
        and abs(g[0][1].im[0]) < 1e-24
        and abs(g[1][0].re[0]) < 1e-24
        and abs(g[1][0].im[0]) < 1e-24
        and abs(g[1][1].re[0] + 1.0) < 1e-24
        and abs(g[1][1].im[0]) < 1e-24
    )


@always_inline
fn is_x(g: Gate) -> Bool:
    """Returns True if the gate is Pauli-X."""
    return (
        abs(g[0][0].re[0]) < 1e-24
        and abs(g[0][0].im[0]) < 1e-24
        and abs(g[0][1].re[0] - 1.0) < 1e-24
        and abs(g[0][1].im[0]) < 1e-24
        and abs(g[1][0].re[0] - 1.0) < 1e-24
        and abs(g[1][0].im[0]) < 1e-24
        and abs(g[1][1].re[0]) < 1e-24
        and abs(g[1][1].im[0]) < 1e-24
    )


@always_inline
fn is_y(g: Gate) -> Bool:
    """Returns True if the gate is Pauli-Y."""
    return (
        abs(g[0][0].re[0]) < 1e-24
        and abs(g[0][0].im[0]) < 1e-24
        and abs(g[0][1].re[0]) < 1e-24
        and abs(g[0][1].im[0] + 1.0) < 1e-24
        and abs(g[1][0].re[0]) < 1e-24
        and abs(g[1][0].im[0] - 1.0) < 1e-24
        and abs(g[1][1].re[0]) < 1e-24
        and abs(g[1][1].im[0]) < 1e-24
    )


@always_inline
fn is_p(g: Gate) -> Bool:
    """Returns True if the gate is a phase gate P(theta)."""
    return (
        abs(g[0][0].re[0] - 1.0) < 1e-24
        and abs(g[0][0].im[0]) < 1e-24
        and abs(g[0][1].re[0]) < 1e-24
        and abs(g[0][1].im[0]) < 1e-24
        and abs(g[1][0].re[0]) < 1e-24
        and abs(g[1][0].im[0]) < 1e-24
        # g[1][1] can be any phase exp(i*theta)
    )


@always_inline
fn get_phase_angle(g: Gate) -> Float64:
    """Extract theta from a phase gate G = [[1,0],[0, exp(i*theta)]]."""
    return atan2(g[1][1].im[0], g[1][1].re[0])
