from butterfly.core.quantum_circuit import (
    Circuit,
    ClassicalTransform,
    MeasurementTransform,
    QuantumTransformation,
)
from butterfly.core.circuit import (
    GateTransformation,
    SwapTransformation,
    QubitReversalTransformation,
)
from butterfly.core.gates import H_Gate
from butterfly.core.types import FloatType
from collections import List


fn circuit_to_string(circuit: Circuit) -> String:
    @always_inline
    fn format_arg(value: FloatType) -> String:
        return String(round(value, 2))

    @always_inline
    fn format_gate(tr: GateTransformation) -> String:
        var s = tr.gate_info.name
        if tr.gate_info.arg:
            s += "(" + format_arg(tr.gate_info.arg.value()) + ")"
        return s

    @always_inline
    fn format_classical(tr: ClassicalTransform) -> String:
        return tr.name

    @always_inline
    fn format_transformation(tr: QuantumTransformation) -> String:
        if tr.isa[GateTransformation]():
            return format_gate(tr[GateTransformation])
        if tr.isa[SwapTransformation]():
            return "SWAP"
        if tr.isa[QubitReversalTransformation]():
            return "QREV"
        if tr.isa[MeasurementTransform]():
            return "MEASURE"
        return format_classical(tr[ClassicalTransform])

    var out = "QuantumCircuitn=" + String(circuit.num_qubits)
    out += ", registers=["
    for i in range(len(circuit.registers)):
        if i > 0:
            out += ","
        var reg = circuit.registers[i].copy()
        out += (
            reg.name
            + ":"
            + String(reg.start)
            + "+"
            + String(reg.length)
        )
    out += "])\n"
    for i in range(len(circuit.transformations)):
        var tr = circuit.transformations[i].copy()
        var line = String(i) + ": " + format_transformation(tr)
        if tr.isa[GateTransformation]():
            var gate_tr = tr[GateTransformation].copy()
            line += " t=" + String(gate_tr.target)
            if len(gate_tr.controls) > 0:
                line += " c=["
                for j in range(len(gate_tr.controls)):
                    if j > 0:
                        line += ","
                    line += String(gate_tr.controls[j])
                line += "]"
        elif tr.isa[SwapTransformation]():
            var swap_tr = tr[SwapTransformation].copy()
            line += " a=" + String(swap_tr.a) + " b=" + String(swap_tr.b)
        elif tr.isa[QubitReversalTransformation]():
            var qrev_tr = tr[QubitReversalTransformation].copy()
            if len(qrev_tr.targets) > 0:
                line += " targets=["
                for j in range(len(qrev_tr.targets)):
                    if j > 0:
                        line += ","
                    line += String(qrev_tr.targets[j])
                line += "]"
        elif tr.isa[MeasurementTransform]():
            var meas_tr = tr[MeasurementTransform].copy()
            if len(meas_tr.targets) > 0:
                line += " targets=["
                for j in range(len(meas_tr.targets)):
                    if j > 0:
                        line += ","
                    line += String(meas_tr.targets[j])
                line += "]"
            if len(meas_tr.values) > 0:
                line += " values=["
                for j in range(len(meas_tr.values)):
                    if j > 0:
                        line += ","
                    if meas_tr.values[j]:
                        line += String(meas_tr.values[j].value())
                    else:
                        line += "_"
                line += "]"
            if meas_tr.seed:
                line += " seed=" + String(meas_tr.seed.value())
        else:
            var cl_tr = tr[ClassicalTransform].copy()
            if len(cl_tr.targets) > 0:
                line += " targets=["
                for j in range(len(cl_tr.targets)):
                    if j > 0:
                        line += ","
                    line += String(cl_tr.targets[j])
                line += "]"
        out += line + "\n"
    return out


fn print_circuit(circuit: Circuit):
    print(circuit_to_string(circuit))


fn circuit_to_ascii(circuit: Circuit) -> String:
    """Render an ASCII circuit diagram."""
    @always_inline
    fn format_arg(value: FloatType) -> String:
        return String(round(value, 2))

    @always_inline
    fn center_label(label: String, width: Int) -> String:
        var left = (width - len(label)) // 2
        var right = width - len(label) - left
        var out = ""
        for _ in range(left):
            out += " "
        out += label
        for _ in range(right):
            out += " "
        return out

    @always_inline
    fn format_gate(tr: GateTransformation) -> String:
        var s = tr.gate_info.name
        if tr.gate_info.arg:
            s += "(" + format_arg(tr.gate_info.arg.value()) + ")"
        return s

    @always_inline
    fn format_classical(tr: ClassicalTransform) -> String:
        return tr.name

    var index_width = len(String(circuit.num_qubits - 1))
    var lines = List[String](capacity=circuit.num_qubits)
    for q in range(circuit.num_qubits):
        var qlabel = String(q)
        while len(qlabel) < index_width:
            qlabel = " " + qlabel
        lines.append("q" + qlabel + ": ")

    for tr in circuit.transformations:
        var label = ""
        var gate_tr = GateTransformation([], 0, H_Gate)
        var is_gate = False
        var is_swap = False
        var is_qrev = False
        var is_measure = False
        var swap_tr = SwapTransformation(0, 0)
        var qrev_targets = List[Int]()
        var has_qrev_targets = False
        var cl_targets = List[Int]()
        var has_cl_targets = False
        var meas_targets = List[Int]()
        var has_meas_targets = False
        if tr.isa[GateTransformation]():
            gate_tr = tr[GateTransformation].copy()
            label = format_gate(gate_tr)
            is_gate = True
        elif tr.isa[SwapTransformation]():
            swap_tr = tr[SwapTransformation].copy()
            label = "SWAP"
            is_swap = True
        elif tr.isa[QubitReversalTransformation]():
            var qrev_tr = tr[QubitReversalTransformation].copy()
            label = "QREV"
            qrev_targets = qrev_tr.targets.copy()
            has_qrev_targets = len(qrev_targets) > 0
            is_qrev = True
        elif tr.isa[MeasurementTransform]():
            var meas_tr = tr[MeasurementTransform].copy()
            label = "MEAS"
            meas_targets = meas_tr.targets.copy()
            has_meas_targets = len(meas_targets) > 0
            is_measure = True
        else:
            var cl_tr = tr[ClassicalTransform].copy()
            label = format_classical(cl_tr)
            cl_targets = cl_tr.targets.copy()
            has_cl_targets = len(cl_targets) > 0
        var box_width = len(label) + 2
        if box_width < 5:
            box_width = 5
        var seg_len = box_width + 4
        if seg_len % 2 == 0:
            seg_len += 1
            box_width += 1
        var label_width = box_width - 2
        var box = "[" + center_label(label, label_width) + "]"
        var box_start = (seg_len - box_width) // 2

        var mid = seg_len // 2
        var min_q = 0
        var max_q = circuit.num_qubits - 1
        if is_gate:
            min_q = gate_tr.target
            max_q = gate_tr.target
            for c in gate_tr.controls:
                if c < min_q:
                    min_q = c
                if c > max_q:
                    max_q = c
        elif is_swap:
            min_q = swap_tr.a if swap_tr.a < swap_tr.b else swap_tr.b
            max_q = swap_tr.b if swap_tr.b > swap_tr.a else swap_tr.a

        for q in range(circuit.num_qubits):
            var segment = "-" * seg_len
            if is_gate and q == gate_tr.target:
                segment = (
                    segment[:box_start]
                    + box
                    + segment[box_start + box_width:]
                )
            elif is_swap and (q == swap_tr.a or q == swap_tr.b):
                segment = (
                    segment[:box_start]
                    + box
                    + segment[box_start + box_width:]
                )
            elif is_swap and q > min_q and q < max_q:
                segment = segment[:mid] + "+" + segment[mid + 1:]
            elif is_qrev:
                var show_box = not has_qrev_targets
                if has_qrev_targets:
                    for t in qrev_targets:
                        if t == q:
                            show_box = True
                            break
                if show_box:
                    segment = (
                        segment[:box_start]
                        + box
                        + segment[box_start + box_width:]
                    )
            elif is_measure:
                var show_box = not has_meas_targets
                if has_meas_targets:
                    for t in meas_targets:
                        if t == q:
                            show_box = True
                            break
                if show_box:
                    segment = (
                        segment[:box_start]
                        + box
                        + segment[box_start + box_width:]
                    )
            elif not is_gate and not is_swap:
                var show_box = not has_cl_targets
                if has_cl_targets:
                    for t in cl_targets:
                        if t == q:
                            show_box = True
                            break
                if show_box:
                    segment = (
                        segment[:box_start]
                        + box
                        + segment[box_start + box_width:]
                    )
            else:
                var is_control = False
                for c in gate_tr.controls:
                    if c == q:
                        is_control = True
                        break
                if is_control:
                    segment = segment[:mid] + "o" + segment[mid + 1:]
                elif len(gate_tr.controls) > 0 and q > min_q and q < max_q:
                    segment = segment[:mid] + "+" + segment[mid + 1:]
            lines[q] = lines[q] + segment + " "

    var out = ""
    for q in range(circuit.num_qubits):
        out += lines[q] + "\n"
    return out


fn print_circuit_ascii(circuit: Circuit):
    print(circuit_to_ascii(circuit))
