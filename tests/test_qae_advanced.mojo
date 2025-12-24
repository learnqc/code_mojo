from butterfly.core.circuit import QuantumCircuit, QuantumRegister
from butterfly.algos.amplitude_estimation import amplitude_estimation_circuit
from butterfly.algos.grover import phase_oracle_match
from butterfly.core.types import pi, Amplitude, FloatType
from butterfly.core.gates import cis
import math


fn prepare_uniform(n: Int) -> QuantumCircuit:
    var qc = QuantumCircuit(n)
    _ = qc.add_register("q", n)
    for i in range(n):
        qc.h(i)
    return qc^


fn complex_sincd(n: Int, v: Float64) -> List[Amplitude]:
    var N = 1 << n
    var c = List[Amplitude]()
    for k in range(N):
        var p_val: Float64 = 1.0
        for j in range(n):
            var angle = (v - Float64(k)) * pi / Float64(1 << (j + 1))
            p_val *= math.cos(angle)

        var cis_arg = (Float64(N - 1) / Float64(N)) * (v - Float64(k)) * pi
        # cis(x) returns Amplitude(cos x, sin x)
        var val = Amplitude(p_val, 0) * cis(cis_arg)
        c.append(val)
    return c^


fn abs_sq(a: Amplitude) -> Float64:
    return a.re**2 + a.im**2


fn main() raises:
    test_qae_suite()
    print("Advanced QAE tests passed!")


fn test_qae_suite() raises:
    var n_prec = 4  # User 'n'
    var n_state = 3  # User 'm'

    var N_prec = 1 << n_prec
    var N_state = 1 << n_state

    # Range 1 to 2^m - 1.
    for l in range(1, N_state):
        var items = List[Int]()
        for i in range(l):
            items.append(i)

        print("\nitems count:", l)

        var prepare = prepare_uniform(n_state)
        var oracle = phase_oracle_match(n_state, items)

        var qc = amplitude_estimation_circuit(
            n_prec, prepare, oracle, swap=False
        )
        var state_qc = qc.execute()

        # Calculate marginal probabilities on the 'c' (precision) register.
        var probs = List[Float64](capacity=N_prec)
        for _ in range(N_prec):
            probs.append(0.0)

        for k in range(N_state):
            for j in range(N_prec):
                var idx = k * N_prec + j
                var amp = state_qc[idx]
                probs[j] += amp.re**2 + amp.im**2

        # Theoretical v
        var term = math.sqrt(Float64(l) / Float64(N_state))
        var v_val = (Float64(N_prec) / pi) * math.asin(term)

        # Probs1 calculation (folded spectrum)
        var half_len = N_prec // 2
        var probs1 = List[Float64]()
        # Item 0
        probs1.append(probs[0] + probs[half_len])

        for k in range(1, half_len):
            # idx1: k - half_len. (wraps negative)
            var idx1 = k - half_len
            if idx1 < 0:
                idx1 = idx1 + N_prec  # -7 -> 9

            var idx2 = k + half_len  # 1+8 = 9

            # User python: probs[idx1] + probs[idx2].
            # Note in python: -7 is 9. +8 is 9.
            # So it sums same element twice?
            # Let's assume indices are meant to be 'k' relative to center.
            # If user wanted symmetric sum: probs[N-k] + probs[k].
            # But formula `probs[k - len//2]` suggests shifting.
            # Let's trust literal interpretation:

            probs1.append(probs[idx1] + probs[idx2])

        # Compare with complex_sincd(n-1, v)
        var sinc_ref = complex_sincd(n_prec - 1, v_val)

        for k in range(len(probs1)):
            var val_exp = probs1[k]
            var val_theory = abs_sq(sinc_ref[k])

            var diff = val_exp - val_theory
            if diff < 0:
                diff = -diff

            if diff > 0.05:
                print(
                    "SincD mismatch at k:",
                    k,
                    "Exp:",
                    val_exp,
                    "Theory:",
                    val_theory,
                )

        # Estimate count (Using Sin^2 as user did, but checking error)
        var max_p = -1.0
        var best_v = -1

        for k in range(half_len, N_prec):
            if probs[k] > max_p:
                max_p = probs[k]
                best_v = k

        var angle_est = Float64(best_v) * pi / Float64(N_prec)
        var count_est = Int(Float64(N_state) * math.cos(angle_est) ** 2)

        print("Real l:", l, "Est:", count_est)

        var err = count_est - l
        if err < 0:
            err = -err
        # Allow small deviation
        if err > 1:
            print("Estimation deviated!")
