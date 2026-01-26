from collections import List
from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.algos.shor import (
    estimate_order_from_state,
    factors_from_order,
    gcd,
    order_finding_circuit,
    shor_factor_simulated,
)
from butterfly.algos.shor_polynomial import build_shor_polynomial_circuit
from butterfly.utils.circuit_print import print_circuit_ascii
from butterfly.utils.visualization import (
    animate_execution,
    animate_execution_table,
    print_state,
    print_state_grid_colored_cells,
)

from tools.circuit_core import (
    Session,
    apply_gate,
    compute_state,
    ensure_circuit,
    require_int_flexible,
)
from butterfly.core.executors import execute
from butterfly.utils.context import ExecContext

from tools.tool_spec import tools_summary


fn sanitize_token(token: String) -> String:
    var t = String(token.strip())
    while t.startswith("("):
        t = String(t[1:])
    while t.endswith(")") or t.endswith(","):
        t = String(t[: len(t) - 1])
    return String(t)


fn looks_like_angle(token: String) -> Bool:
    if token.find("pi") >= 0:
        return True
    if token.find(".") >= 0:
        return True
    return False


fn looks_like_index(token: String) -> Bool:
    var t = String(token.strip())
    if t.startswith("-"):
        return False
    for i in range(len(t)):
        var c = t[i]
        if c < "0" or c > "9":
            return False
    return String(t) != ""


fn parse_bool_from_string(value: String, default: Bool) -> Bool:
    var val = value.lower()
    if val == "true" or val == "1" or val == "yes" or val == "on":
        return True
    if val == "false" or val == "0" or val == "no" or val == "off":
        return False
    return default


fn row_bits_from_rows(rows: Int) -> Int:
    if rows <= 0:
        return -1
    var bits = 0
    var value = rows
    while value > 1:
        if value % 2 != 0:
            return -1
        value = value // 2
        bits += 1
    return bits


fn default_col_bits(num_qubits: Int) -> Int:
    if num_qubits <= 0:
        return 0
    var row_bits = num_qubits // 2
    if num_qubits % 2 == 0:
        return row_bits
    return row_bits + 1


fn bit_length(value: Int) -> Int:
    if value <= 0:
        return 0
    var bits = 0
    var v = value
    while v > 0:
        v = v // 2
        bits += 1
    return bits


fn parse_args_list(raw: String) -> List[String]:
    var parts = raw.split(" ")
    var out = List[String]()
    for i in range(len(parts)):
        var token = String(parts[i]).strip()
        if String(token) != "":
            out.append(String(token))
    return out^


fn get_arg(
    args: List[Tuple[String, String]],
    key: String,
) -> Optional[String]:
    for i in range(len(args)):
        var (k, v) = args[i]
        if k == key:
            return Optional[String](v)
    return None


fn get_arg_list(
    args: List[Tuple[String, String]],
    key: String,
) -> List[String]:
    var out = List[String]()
    var raw = get_arg(args, key)
    if raw:
        return parse_args_list(raw.value())
    return out^


fn execute_tool(
    mut session: Session,
    name: String,
    args: List[Tuple[String, String]],
) raises -> Tuple[String, Bool]:
    if name == "create_circuit":
        var qubits_val = get_arg(args, "qubits")
        if not qubits_val:
            qubits_val = get_arg(args, "N")
        var qubits = 0
        if qubits_val:
            try:
                qubits = require_int_flexible(qubits_val.value())
            except:
                return ("Invalid qubit count.", True)
        if qubits <= 0:
            return ("Invalid qubit count.", True)
        session.circuit = QuantumCircuit(qubits)
        session.has_circuit = True
        return ("Created circuit with " + String(qubits) + " qubits.", False)
    if name == "add_gate":
        var gate_val = get_arg(args, "gate")
        if not gate_val:
            return ("Missing gate name.", True)
        var gate = gate_val.value().lower()
        var arg_list = get_arg_list(args, "args")
        for i in range(len(arg_list)):
            arg_list[i] = sanitize_token(arg_list[i])
        if (gate == "p" or gate == "rx" or gate == "ry" or gate == "rz") and len(arg_list) == 2:
            if looks_like_angle(arg_list[0]) and looks_like_index(arg_list[1]):
                var tmp = arg_list[0]
                arg_list[0] = arg_list[1]
                arg_list[1] = tmp
        try:
            apply_gate(session, gate, arg_list)
        except:
            return ("Failed to apply gate.", True)
        return ("Added gate: " + gate, False)
    if name == "show_circuit":
        if not ensure_circuit(session):
            return ("No circuit.", False)
        print_circuit_ascii(session.circuit)
        return ("Displayed circuit.", True)
    if name == "show_state":
        if not ensure_circuit(session):
            return ("No circuit.", False)
        var state = compute_state(session)
        print_state(state, short=True, use_color=True)
        return ("Displayed state.", True)
    if name == "show_grid":
        if not ensure_circuit(session):
            return ("No circuit.", False)
        var col_bits = 2
        var col_bits_val = get_arg(args, "col_bits")
        if col_bits_val:
            col_bits = require_int_flexible(col_bits_val.value())
        var use_log = True
        var log_val = get_arg(args, "log")
        if log_val:
            use_log = parse_bool_from_string(log_val.value(), True)
        var show_bin = True
        var bin_val = get_arg(args, "bin")
        if bin_val:
            show_bin = parse_bool_from_string(bin_val.value(), True)
        var origin_bottom = False
        var origin_val = get_arg(args, "origin_bottom")
        if origin_val:
            origin_bottom = parse_bool_from_string(origin_val.value(), False)
        var state = compute_state(session)
        print_state_grid_colored_cells(
            state,
            col_bits,
            use_log=use_log,
            origin_bottom=origin_bottom,
            show_bin_labels=show_bin,
        )
        return ("Displayed grid.", True)
    if name == "show_grid_rows":
        if not ensure_circuit(session):
            return ("No circuit.", False)
        var rows_val = get_arg(args, "rows")
        if not rows_val:
            return ("Missing rows.", True)
        var rows: Int
        try:
            rows = require_int_flexible(rows_val.value())
        except:
            return ("Invalid rows value.", True)
        var row_bits = row_bits_from_rows(rows)
        if row_bits < 0:
            return ("Rows must be a power of two.", True)
        var col_bits = session.circuit.num_qubits - row_bits
        if col_bits < 0:
            return ("Too many rows for this circuit.", True)
        var origin_bottom = False
        var origin_val = get_arg(args, "origin_bottom")
        if origin_val:
            origin_bottom = parse_bool_from_string(origin_val.value(), False)
        var state = compute_state(session)
        print_state_grid_colored_cells(
            state,
            col_bits,
            use_log=True,
            origin_bottom=origin_bottom,
            show_bin_labels=True,
        )
        return ("Displayed grid.", True)
    if name == "animate_table":
        if not ensure_circuit(session):
            return ("No circuit.", False)
        var delay_ms = 50
        var delay_val = get_arg(args, "delay_ms")
        if delay_val:
            try:
                delay_ms = require_int_flexible(delay_val.value())
            except:
                return ("Invalid delay_ms value.", True)
        var step_on_input = False
        var step_val = get_arg(args, "step")
        if step_val:
            step_on_input = parse_bool_from_string(step_val.value(), False)
        var state = QuantumState(session.circuit.num_qubits)
        animate_execution_table(
            session.circuit,
            state,
            short=True,
            use_color=True,
            show_step_label=True,
            delay_s=Float64(delay_ms) / 1000.0,
            step_on_input=step_on_input,
            redraw_in_place=True,
        )
        return ("Animation complete.", True)
    if name == "animate_grid":
        if not ensure_circuit(session):
            return ("No circuit.", False)
        var delay_ms = 50
        var delay_val = get_arg(args, "delay_ms")
        if delay_val:
            try:
                delay_ms = require_int_flexible(delay_val.value())
            except:
                return ("Invalid delay_ms value.", True)
        var step_on_input = False
        var step_val = get_arg(args, "step")
        if step_val:
            step_on_input = parse_bool_from_string(step_val.value(), False)
        var col_bits = default_col_bits(session.circuit.num_qubits)
        var rows_val = get_arg(args, "rows")
        if rows_val:
            var rows: Int
            try:
                rows = require_int_flexible(rows_val.value())
            except:
                return ("Invalid rows value.", True)
            var row_bits = row_bits_from_rows(rows)
            if row_bits < 0:
                return ("Rows must be a power of two.", True)
            col_bits = session.circuit.num_qubits - row_bits
            if col_bits < 0:
                return ("Too many rows for this circuit.", True)
        var origin_bottom = False
        var origin_val = get_arg(args, "origin_bottom")
        if origin_val:
            origin_bottom = parse_bool_from_string(origin_val.value(), False)
        var use_log = True
        var log_val = get_arg(args, "log")
        if log_val:
            use_log = parse_bool_from_string(log_val.value(), True)
        var state = QuantumState(session.circuit.num_qubits)
        animate_execution(
            session.circuit,
            state,
            col_bits,
            use_log=use_log,
            origin_bottom=origin_bottom,
            show_bin_labels=True,
            use_bg=True,
            show_chars=False,
            show_step_label=True,
            delay_s=Float64(delay_ms) / 1000.0,
            step_on_input=step_on_input,
            redraw_in_place=True,
        )
        return ("Animation complete.", True)
    if name == "shor_factor":
        var number_val = get_arg(args, "number")
        if not number_val:
            number_val = get_arg(args, "n")
        if not number_val:
            return ("Missing number.", True)
        var modulus: Int
        try:
            modulus = require_int_flexible(number_val.value())
        except:
            return ("Invalid number.", True)
        if modulus <= 1:
            return ("Number must be >= 2.", True)
        var value_bits = bit_length(modulus)
        var exp_bits = value_bits * 1
        var exp_val = get_arg(args, "exp_bits")
        if exp_val:
            try:
                exp_bits = require_int_flexible(exp_val.value())
            except:
                return ("Invalid exp_bits.", True)
        var value_val = get_arg(args, "value_bits")
        if value_val:
            try:
                value_bits = require_int_flexible(value_val.value())
            except:
                return ("Invalid value_bits.", True)
        if exp_bits <= 0 or value_bits <= 0:
            return ("exp_bits and value_bits must be >= 1.", True)
        var encoding = "classical"
        var enc_val = get_arg(args, "encoding")
        if enc_val:
            encoding = enc_val.value().lower()
        var max_a = modulus - 1
        if max_a > 8:
            max_a = 8
        var chosen_a = 0
        for a in range(2, max_a + 1):
            if gcd(a, modulus) == 1:
                chosen_a = a
                break
        if encoding == "polynomial":
            if chosen_a == 0:
                chosen_a = 2
            var qc = build_shor_polynomial_circuit(
                exp_bits,
                value_bits,
                modulus,
                base=chosen_a,
            )
            session.circuit = qc.copy()
            session.has_circuit = True
            var final_state = QuantumState(exp_bits + value_bits)
            execute(final_state, qc, ExecContext())
            var r = estimate_order_from_state(
                final_state,
                exp_bits,
                value_bits,
                chosen_a,
                modulus,
            )
            if r:
                var factors = factors_from_order(chosen_a, modulus, r.value())
                if factors:
                    var (p, q) = factors.value()
                    return (
                        "Factors: "
                        + String(p)
                        + " x "
                        + String(q)
                        + " (a="
                        + String(chosen_a)
                        + ", order="
                        + String(r.value())
                        + ", exp_bits="
                        + String(exp_bits)
                        + ", value_bits="
                        + String(value_bits)
                        + ", encoding=polynomial)",
                        True,
                    )
            return (
                "No factors found (polynomial encoding; try different exp_bits/value_bits).",
                True,
            )
        if chosen_a == 0:
            session.circuit = QuantumCircuit(exp_bits + value_bits)
            session.has_circuit = True
        else:
            try:
                session.circuit = order_finding_circuit(
                    exp_bits,
                    value_bits,
                    chosen_a,
                    modulus,
                )
                session.has_circuit = True
            except:
                session.circuit = QuantumCircuit(exp_bits + value_bits)
                session.has_circuit = True
        for a in range(2, max_a + 1):
            try:
                var factors = shor_factor_simulated(
                    modulus,
                    a,
                    exp_bits,
                    value_bits,
                )
                if factors:
                    var (p, q) = factors.value()
                    return (
                        "Factors: "
                        + String(p)
                        + " x "
                        + String(q)
                        + " (a="
                        + String(a)
                        + ", exp_bits="
                        + String(exp_bits)
                        + ", value_bits="
                        + String(value_bits)
                        + ")",
                        True,
                    )
            except:
                continue
        return (
            "No factors found (try different exp_bits/value_bits or a different number).",
            True,
        )
    if name == "list_tools":
        return (tools_summary(), True)
    raise Error("Unknown tool: " + name)
