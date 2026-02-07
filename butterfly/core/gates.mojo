from  butterfly.core.types import *
from math import cos, sin, pi
from butterfly.utils.common import cis

alias X: Gate = [[`0`, `1`], [`1`, `0`]]
alias Y: Gate = [[`0`, Complex(0, -1)], [Complex(0, 1), `0`]]
alias Z: Gate = [[`1`, `0`], [`0`, -`1`]]


fn P(theta: FloatType) -> Gate:
    gate: Gate = [[`1`, `0`], [`0`, cis(theta)]]
    return gate^


alias H: Gate = [[sq_half, sq_half], [sq_half, -sq_half]]


fn RZ(theta: FloatType) -> Gate:
    gate: Gate = [
        [Complex(cos(theta / 2), -sin(theta / 2)), `0`],
        [`0`, Complex(cos(theta / 2), sin(theta / 2))],
    ]
    return gate^


fn RX(theta: FloatType) -> Gate:
    gate: Gate = [
        [Complex(cos(theta / 2), 0), Complex(0, -sin(theta / 2))],
        [Complex(0, -sin(theta / 2)), Complex(cos(theta / 2), 0)],
    ]
    return gate^


fn RY(theta: FloatType) -> Gate:
    gate: Gate = [
        [Complex(cos(theta / 2), 0), Complex(-sin(theta / 2), 0)],
        [Complex(sin(theta / 2), 0), Complex(cos(theta / 2), 0)],
    ]
    return gate^


fn U3(theta: FloatType, phi: FloatType, lam: FloatType) -> Gate:
    gate: Gate = [
        [
            Complex(cos(theta / 2), 0),
            -cis(lam) * Complex(sin(theta / 2), 0),
        ],
        [
            cis(phi) * Complex(sin(theta / 2), 0),
            cis(phi + lam) * Complex(cos(theta / 2), 0),
        ],
    ]
    return gate^


fn U2(phi: FloatType, lam: FloatType) -> Gate:
    return U3(pi / 2, phi, lam)


fn U1(lam: FloatType) -> Gate:
    return U3(0, 0, lam)

struct GateKind(Copyable, Movable, ImplicitlyCopyable):
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    fn __str__(self) -> String:
        return gate_kind_name(self)

    @staticmethod
    fn from_int(value: Int) raises -> GateKind:
        if not is_valid_gate_kind(value):
            raise Error("Unknown gate kind: " + String(value))
        return GateKind(value)

    alias H = GateKind(0)
    alias P = GateKind(1)
    alias X = GateKind(2)
    alias Y = GateKind(3)
    alias Z = GateKind(4)
    alias RX = GateKind(5)
    alias RY = GateKind(6)
    alias RZ = GateKind(7)
    alias CUSTOM = GateKind(8)


fn is_valid_gate_kind(value: Int) -> Bool:
    return value >= GateKind.H.value and value <= GateKind.CUSTOM.value


fn gate_kind_name(kind: GateKind) -> String:
    if kind == GateKind.H:
        return "H"
    if kind == GateKind.P:
        return "P"
    if kind == GateKind.X:
        return "X"
    if kind == GateKind.Y:
        return "Y"
    if kind == GateKind.Z:
        return "Z"
    if kind == GateKind.RX:
        return "RX"
    if kind == GateKind.RY:
        return "RY"
    if kind == GateKind.RZ:
        return "RZ"
    if kind == GateKind.CUSTOM:
        return "CUSTOM"
    return "UNKNOWN"


alias gate_names:InlineArray[String, 9] = ["H", "P", "X", "Y", "Z", "RX", "RY", "RZ", "CUSTOM"]

from utils.variant import Variant

alias gate_types: InlineArray[Variant[Gate, fn(FloatType) -> Gate], 9] = [
    H,
    P,
    X,
    Y,
    Z,
    RX,
    RY,
    RZ,
    H,
]

struct GateInfo(ImplicitlyCopyable, Copyable, Movable):
    var kind: GateKind
    var name: String
    var gate: Gate
    var arg: Optional[FloatType]


    fn __init__(
        out self,
        kind: GateKind,
        arg: Optional[FloatType] = None,
    ):
        self.kind = kind
        self.name = gate_names[kind.value]
        var gate_type = gate_types[kind.value]
        if gate_type.isa[Gate]():
            self.gate = gate_type[Gate]
            self.arg = None
        else:
            var gate_fn = gate_type[fn(FloatType) -> Gate]
            self.gate = gate_fn(arg.value())
            self.arg = arg

    fn __init__(
        out self,
        kind: GateKind,
        gate: Gate,
        name: String = "CUSTOM",
    ):
        self.kind = kind
        self.name = name
        self.gate = gate
        self.arg = None

    fn __str__(self) -> String:
        var str = self.name
        if self.arg:
            str += "(" + String(self.arg.value()) + ")"
        return str

alias H_Gate = GateInfo(GateKind.H)
alias X_Gate = GateInfo(GateKind.X)
alias Y_Gate = GateInfo(GateKind.Y)
alias Z_Gate = GateInfo(GateKind.Z)

fn P_Gate(theta: FloatType) -> GateInfo:
    return GateInfo(GateKind.P, theta)

fn RX_Gate(theta: FloatType) -> GateInfo:
    return GateInfo(GateKind.RX, theta)

fn RY_Gate(theta: FloatType) -> GateInfo:
    return GateInfo(GateKind.RY, theta)
    
fn RZ_Gate(theta: FloatType) -> GateInfo:
    return GateInfo(GateKind.RZ, theta)

fn Custom_Gate(gate: Gate, name: String = "CUSTOM") -> GateInfo:
    return GateInfo(GateKind.CUSTOM, gate, name)
