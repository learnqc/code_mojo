from __future__ import annotations

from math import pi

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector

_TQ_CACHE: dict[tuple[int, float, str, int, str], object] = {}
_RAND_CACHE: dict[tuple[int, int, int, int, str], object] = {}


def _build_value_encoding_circuit(n: int, value: float, stage: str) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
    for j in range(n):
        theta = 2 * pi / (2 ** (j + 1)) * value
        qc.p(theta, j)
    if stage == "full":
        iqft = QFT(n, do_swaps=False, inverse=True)
        qc.append(iqft, range(n))
    return qc


def _statevector_from_circuit(qc: QuantumCircuit, opt_level: int) -> Statevector:
    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method="statevector")
        tqc = transpile(qc, backend, optimization_level=opt_level)
        tqc.save_statevector()
        result = backend.run(tqc, shots=1).result()
        return result.get_statevector(tqc)
    except Exception:
        pass

    tqc = transpile(qc, optimization_level=opt_level)
    return Statevector.from_instruction(tqc)


def _fast_statevector_run(qc: QuantumCircuit, opt_level: int) -> None:
    _ = _statevector_from_circuit(qc, opt_level)


def run_value_encoding(n: int, value: float, stage: str, opt_level: int) -> None:
    qc = _build_value_encoding_circuit(n, value, stage)
    _fast_statevector_run(qc, opt_level)


def transpile_value_encoding(
    n: int, value: float, stage: str, opt_level: int
) -> None:
    qc = _build_value_encoding_circuit(n, value, stage)
    key = (n, value, stage, opt_level, "aer")
    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method="statevector")
        tqc = transpile(qc, backend, optimization_level=opt_level)
        tqc.save_statevector()
        _TQ_CACHE[key] = (backend, tqc)
        return
    except Exception:
        pass

    tqc = transpile(qc, optimization_level=opt_level)
    _TQ_CACHE[(n, value, stage, opt_level, "qiskit")] = tqc


def run_cached_statevector(
    n: int, value: float, stage: str, opt_level: int
) -> None:
    key_aer = (n, value, stage, opt_level, "aer")
    if key_aer in _TQ_CACHE:
        backend, tqc = _TQ_CACHE[key_aer]
        result = backend.run(tqc, shots=1).result()
        _ = result.get_statevector(tqc)
        return

    key_qiskit = (n, value, stage, opt_level, "qiskit")
    if key_qiskit in _TQ_CACHE:
        tqc = _TQ_CACHE[key_qiskit]
        _ = Statevector.from_instruction(tqc)
        return

    # Fallback if cache is missing.
    run_value_encoding(n, value, stage, opt_level)


def _build_single_gate_circuit(
    n: int, gate: str, theta: float, target: int
) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    gate_u = gate.upper()
    if gate_u == "H":
        qc.h(target)
    elif gate_u == "X":
        qc.x(target)
    elif gate_u == "Y":
        qc.y(target)
    elif gate_u == "Z":
        qc.z(target)
    elif gate_u == "P":
        qc.p(theta, target)
    elif gate_u == "RX":
        qc.rx(theta, target)
    elif gate_u == "RY":
        qc.ry(theta, target)
    elif gate_u == "RZ":
        qc.rz(theta, target)
    else:
        raise ValueError(f"Unsupported gate: {gate}")
    return qc


def transpile_single_gate(
    n: int, gate: str, theta: float, target: int, opt_level: int
) -> None:
    qc = _build_single_gate_circuit(n, gate, theta, target)
    key = (n, gate, theta, target, opt_level, "aer")
    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method="statevector")
        tqc = transpile(qc, backend, optimization_level=opt_level)
        tqc.save_statevector()
        _TQ_CACHE[key] = (backend, tqc)
        return
    except Exception:
        pass

    tqc = transpile(qc, optimization_level=opt_level)
    _TQ_CACHE[(n, gate, theta, target, opt_level, "qiskit")] = tqc


def run_single_gate_cached(
    n: int, gate: str, theta: float, target: int, opt_level: int
) -> None:
    key_aer = (n, gate, theta, target, opt_level, "aer")
    if key_aer in _TQ_CACHE:
        backend, tqc = _TQ_CACHE[key_aer]
        result = backend.run(tqc, shots=1).result()
        _ = result.get_statevector(tqc)
        return

    key_qiskit = (n, gate, theta, target, opt_level, "qiskit")
    if key_qiskit in _TQ_CACHE:
        tqc = _TQ_CACHE[key_qiskit]
        _ = Statevector.from_instruction(tqc)
        return

    # Fallback if cache is missing.
    qc = _build_single_gate_circuit(n, gate, theta, target)
    _fast_statevector_run(qc, opt_level)


def sample_statevector(
    n: int, value: float, stage: str, opt_level: int, indices: list[int]
) -> list[complex]:
    qc = _build_value_encoding_circuit(n, value, stage)
    sv = _statevector_from_circuit(qc, opt_level)
    return [sv.data[i] for i in indices]


class _Lcg:
    MOD = 2147483647
    MUL = 48271

    def __init__(self, seed: int) -> None:
        s = seed % self.MOD
        if s <= 0:
            s += self.MOD - 1
        self.state = s

    def next_raw(self) -> int:
        self.state = (self.state * self.MUL) % self.MOD
        return self.state

    def next_int(self, max_val: int) -> int:
        if max_val <= 0:
            return 0
        return self.next_raw() % max_val

    def next_angle(self) -> float:
        return (self.next_raw() / self.MOD) * (2 * pi) - pi


def _build_random_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    rng = _Lcg(seed)

    for _ in range(depth):
        is_controlled = (rng.next_raw() % 100) >= 70
        if not is_controlled:
            gate_id = rng.next_int(8)
            target = rng.next_int(n)
            if gate_id == 0:
                qc.h(target)
            elif gate_id == 1:
                qc.x(target)
            elif gate_id == 2:
                qc.y(target)
            elif gate_id == 3:
                qc.z(target)
            elif gate_id == 4:
                qc.p(rng.next_angle(), target)
            elif gate_id == 5:
                qc.rx(rng.next_angle(), target)
            elif gate_id == 6:
                qc.ry(rng.next_angle(), target)
            elif gate_id == 7:
                qc.rz(rng.next_angle(), target)
        else:
            gate_id = rng.next_int(7)
            control = rng.next_int(n)
            target = rng.next_int(n)
            if target == control:
                target = (target + 1) % n
            if gate_id == 0:
                qc.cx(control, target)
            elif gate_id == 1:
                qc.cy(control, target)
            elif gate_id == 2:
                qc.cz(control, target)
            elif gate_id == 3:
                qc.cp(rng.next_angle(), control, target)
            elif gate_id == 4:
                qc.crx(rng.next_angle(), control, target)
            elif gate_id == 5:
                qc.cry(rng.next_angle(), control, target)
            elif gate_id == 6:
                qc.crz(rng.next_angle(), control, target)
    return qc


def transpile_random_circuit(
    n: int, depth: int, seed: int, opt_level: int
) -> None:
    qc = _build_random_circuit(n, depth, seed)
    key = (n, depth, seed, opt_level, "aer")
    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method="statevector")
        tqc = transpile(qc, backend, optimization_level=opt_level)
        tqc.save_statevector()
        _RAND_CACHE[key] = (backend, tqc)
        return
    except Exception:
        pass

    tqc = transpile(qc, optimization_level=opt_level)
    _RAND_CACHE[(n, depth, seed, opt_level, "qiskit")] = tqc


def run_cached_random_circuit(
    n: int, depth: int, seed: int, opt_level: int
) -> None:
    key_aer = (n, depth, seed, opt_level, "aer")
    if key_aer in _RAND_CACHE:
        backend, tqc = _RAND_CACHE[key_aer]
        result = backend.run(tqc, shots=1).result()
        _ = result.get_statevector(tqc)
        return

    key_qiskit = (n, depth, seed, opt_level, "qiskit")
    if key_qiskit in _RAND_CACHE:
        tqc = _RAND_CACHE[key_qiskit]
        _ = Statevector.from_instruction(tqc)
        return

    # Fallback if cache is missing.
    qc = _build_random_circuit(n, depth, seed)
    _fast_statevector_run(qc, opt_level)


def sample_statevector_random(
    n: int, depth: int, seed: int, opt_level: int, indices: list[int]
) -> list[complex]:
    qc = _build_random_circuit(n, depth, seed)
    sv = _statevector_from_circuit(qc, opt_level)
    return [sv.data[i] for i in indices]
