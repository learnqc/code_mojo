from butterfly.core.circuit import QuantumCircuit, QuantumRegister
from butterfly.core.state import QuantumState
from butterfly.core.types import pi, Amplitude, FloatType, Gate
from butterfly.core.gates import H, X, RX, RY, cis
from math import sqrt, cos, sin, asin
import math


fn gate_to_unitary(gate: Gate) -> List[Amplitude]:
    """Convert a 2x2 Gate matrix to a flat List[Amplitude] for unitary operations.

    Returns elements in row-major order: [g00, g01, g10, g11]
    """
    var u = List[Amplitude](capacity=4)
    u.append(gate[0][0])
    u.append(gate[0][1])
    u.append(gate[1][0])
    u.append(gate[1][1])
    return u^


fn complex_sincd(n: Int, v: Float64) -> List[Amplitude]:
    """Calculate the complex sinc distribution for QAE verification."""
    var N = 1 << n
    var c = List[Amplitude]()
    for k in range(N):
        var p_val: Float64 = 1.0
        for j in range(n):
            var angle = (v - Float64(k)) * pi / Float64(1 << (j + 1))
            p_val *= math.cos(angle)

        var cis_arg = (Float64(N - 1) / Float64(N)) * (v - Float64(k)) * pi
        var val = Amplitude(p_val, 0) * cis(cis_arg)
        c.append(val)
    return c^


fn abs_sq(a: Amplitude) -> Float64:
    """Calculate |a|^2 for an amplitude."""
    return a.re**2 + a.im**2


fn all_close(
    state1: QuantumState, state2: QuantumState, tol: Float64 = 1e-6
) -> Bool:
    """Check if two quantum states are approximately equal."""
    if len(state1) != len(state2):
        return False

    for i in range(len(state1)):
        var diff_re = state1[i].re - state2[i].re
        var diff_im = state1[i].im - state2[i].im
        if diff_re < 0:
            diff_re = -diff_re
        if diff_im < 0:
            diff_im = -diff_im
        if diff_re > tol or diff_im > tol:
            return False
    return True


fn test_unitary_consistency() raises:
    """Test that gate-based and unitary-based circuits produce identical results.
    """
    var n = 3
    var theta = 4.7 * pi / Float64(1 << (n - 1))
    var inverse = True

    var N = 1 << n

    # Store states for gate-based and unitary-based execution
    var state_gate: QuantumState
    var state_unitary: QuantumState

    # Gate-based implementation
    print("Running gate-based circuit...")
    var qc_gate = QuantumCircuit(n + 1)
    var q_gate = qc_gate.add_register("q", n)
    var a_gate = qc_gate.add_register("a", 1)

    # Preparation on ancilla
    qc_gate.x(a_gate[0])
    qc_gate.rx(a_gate[0], -pi / 2)

    # Hadamards on state register
    for i in range(n):
        qc_gate.h(q_gate[i])

    # Controlled RY gates
    for i in range(n):
        for _ in range(1 << i):
            qc_gate.cry(a_gate[0], q_gate[i], 2 * theta)

    # Inverse operations on ancilla
    if inverse:
        qc_gate.rx(a_gate[0], pi / 2)
        qc_gate.x(a_gate[0])

    # IQFT on state register
    qc_gate.iqft(q_gate)

    qc_gate.execute()
    state_gate = qc_gate.state

    # Unitary-based implementation
    print("Running unitary-based circuit...")
    var qc_unitary = QuantumCircuit(n + 1)
    var q_unitary = qc_unitary.add_register("q", n)
    var a_unitary = qc_unitary.add_register("a", 1)

    # Preparation on ancilla using unitaries
    qc_unitary.append_u(gate_to_unitary(X), a_unitary)
    qc_unitary.unitary(gate_to_unitary(RX(-pi / 2)), a_unitary[0])

    # Hadamards on state register using unitaries
    for i in range(n):
        qc_unitary.unitary(gate_to_unitary(H), q_unitary[i])

    # Controlled RY gates using unitaries
    for i in range(n):
        for _ in range(1 << i):
            qc_unitary.c_append_u(
                gate_to_unitary(RY(2 * theta)), q_unitary[i], a_unitary
            )

    # Inverse operations on ancilla using unitaries
    if inverse:
        qc_unitary.unitary(gate_to_unitary(RX(pi / 2)), a_unitary[0])
        qc_unitary.unitary(gate_to_unitary(X), a_unitary[0])

    # IQFT on state register
    qc_unitary.iqft(q_unitary)

    qc_unitary.execute()
    state_unitary = qc_unitary.state

    # Verify states match
    print("Comparing gate-based vs unitary-based states...")
    if not all_close(state_gate, state_unitary):
        print("ERROR: Gate-based and unitary-based states do not match!")
        raise Error("State mismatch between gate and unitary implementations")

    print("✓ States match!")

    # Verify against theoretical distribution
    if inverse:
        print("Verifying against theoretical complex_sincd distribution...")
        var s = complex_sincd(n, theta / (2 * pi) * Float64(N))

        # Compare first N amplitudes
        for i in range(N):
            var diff_re = state_gate[i].re - s[i].re
            var diff_im = state_gate[i].im - s[i].im
            if diff_re < 0:
                diff_re = -diff_re
            if diff_im < 0:
                diff_im = -diff_im

            if diff_re > 1e-5 or diff_im > 1e-5:
                print("Mismatch at index", i)
                print("  Expected:", s[i].re, "+", s[i].im, "i")
                print(
                    "  Got:     ", state_gate[i].re, "+", state_gate[i].im, "i"
                )
                raise Error("State does not match theoretical distribution")

        print("✓ Theoretical distribution matches!")
    else:
        print("Verifying non-inverse case against theoretical distribution...")
        var s = complex_sincd(n, theta / (2 * pi) * Float64(N))

        # First N amplitudes should be 1/sqrt(2) * 1j * s
        # Last N amplitudes should be 1/sqrt(2) * s
        var factor = 1.0 / sqrt(2.0)

        for i in range(N):
            # First half: 1/sqrt(2) * 1j * s[i] = 1/sqrt(2) * (s[i].im*i + s[i].re*i*i)
            #                                    = 1/sqrt(2) * (-s[i].re + s[i].im*i)
            var expected_re = -factor * s[i].im
            var expected_im = factor * s[i].re

            var diff_re = state_gate[i].re - expected_re
            var diff_im = state_gate[i].im - expected_im
            if diff_re < 0:
                diff_re = -diff_re
            if diff_im < 0:
                diff_im = -diff_im

            if diff_re > 1e-5 or diff_im > 1e-5:
                print("Mismatch at index", i, "(first half)")
                raise Error("State does not match theoretical distribution")

        for i in range(N):
            # Second half: 1/sqrt(2) * s[i]
            var expected_re = factor * s[i].re
            var expected_im = factor * s[i].im

            var diff_re = state_gate[N + i].re - expected_re
            var diff_im = state_gate[N + i].im - expected_im
            if diff_re < 0:
                diff_re = -diff_re
            if diff_im < 0:
                diff_im = -diff_im

            if diff_re > 1e-5 or diff_im > 1e-5:
                print("Mismatch at index", N + i, "(second half)")
                raise Error("State does not match theoretical distribution")

        print("✓ Theoretical distribution matches!")

    print("\n✅ Unitary consistency test passed!")


fn main() raises:
    test_unitary_consistency()
