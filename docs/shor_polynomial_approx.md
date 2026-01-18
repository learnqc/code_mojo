# Shor via Polynomial Encodings (Exact, Approx, Sampled)

This note records the current **polynomial-based** Shor experiments in this repo:

- Exact multilinear polynomial for `f(x) = 2^x mod N` (truth-table / Möbius).
- Approximation via truncation (degree / top‑k).
- Sampling‑based approximation (Monte Carlo coefficients, no full truth table).

**Caution**: These are **not** reversible modular‑exponentiation circuits.  
They are phase‑encoding visualizations / experiments.

---

## 1) Exact polynomial encoding (truth table)

**Idea**  
Compute the exact multilinear coefficients (ANF) for `f(x) = 2^x mod N`:

- Evaluate `f(x)` for all `x in [0, 2^n)`.
- Apply Möbius transform to get ANF coefficients.
- Build a phase‑encoding circuit from those terms.

**Files**
- `tests/test_animate_shor_poly_modexp.mojo`

**Key steps**
- `build_modexp_terms(...)` builds ANF from full truth table.
- `build_polynomial_circuit(exp_bits, value_bits, terms)` encodes the phases.
- IQFT is applied to the exponent register.
- `estimate_order_from_state(...)` + `factors_from_order(...)` report results.

---

## 2) Approximation by truncation (degree / top‑k)

**Idea**  
Use the exact ANF coefficients, then **keep only a subset**:

- Limit maximum term degree (e.g., degree ≤ 3).
- Optionally keep only the top‑`k` largest‑magnitude terms.

**Files**
- `tests/test_animate_shor_poly_modexp_approx.mojo`

**Key knobs**
- `degree_cap`
- `max_terms`

**Notes**
This still requires the **full truth table**, but the circuit is much smaller.

---

## 3) Sampling‑based approximation (no full truth table)

**Idea**  
Estimate ANF coefficients via Monte Carlo samples of `f(x)`:

```
coeff(mask) = 2^n * E_x[ (-1)^(|x|-|mask|) * f(x) * 1_{mask ⊆ x} ]
```

**Files**
- `tests/test_animate_shor_poly_modexp_sampled.mojo`

**Key knobs**
- `degree_cap` (max term degree)
- `max_terms` (top‑k)
- `samples` (Monte Carlo samples)
- `seed`

**Notes**
This avoids the full truth table but is approximate and instance‑dependent.

---

## 4) Value‑bits sweep (empirical robustness)

**Idea**  
Sweep `value_bits` from 1 to `ceil(log2 N)` while keeping `exp_bits = 2 * value_bits_max`.  
Track which `a` values still recover factors even with fewer value qubits.

**File**
- `tests/test_shor_value_bits_sweep.mojo`

**Output**
- Per‑`a` sweep with success/failure.
- `true_order` for reference.
- CSV‑style dump for easy plotting.

---

## 5) Experiments summary (what to remember)

- These are **phase‑encoding** experiments, not reversible modular arithmetic.
- Success is **instance‑dependent** (some bases `a` work, others fail).
- Smaller `value_bits` can still reveal the period for some `a`.
- Approximation quality depends strongly on term budget and sampling.

---

## Quick commands

```bash
# Exact polynomial encoding
mojo tests/test_animate_shor_poly_modexp.mojo

# Approx by truncation (degree / top‑k)
mojo tests/test_animate_shor_poly_modexp_approx.mojo

# Sampling‑based approximation (no truth table)
mojo tests/test_animate_shor_poly_modexp_sampled.mojo

# Sweep value_bits
mojo tests/test_shor_value_bits_sweep.mojo
```

---

## Design warnings (repeatable)

- These are **not Shor‑correct** unless you replace the polynomial oracle with a
  reversible modular‑exponentiation circuit.
- Use them for **visualization and heuristics**, not guarantees.
