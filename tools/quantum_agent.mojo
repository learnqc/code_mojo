from collections import List
from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.utils.circuit_print import print_circuit_ascii
from butterfly.utils.visualization import (
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
            qubits = require_int_flexible(qubits_val.value())
        if qubits <= 0:
            raise Error("Invalid qubit count.")
        session.circuit = QuantumCircuit(qubits)
        session.has_circuit = True
        return ("Created circuit with " + String(qubits) + " qubits.", False)
    if name == "add_gate":
        var gate_val = get_arg(args, "gate")
        if not gate_val:
            raise Error("Missing gate name.")
        var gate = gate_val.value().lower()
        var arg_list = get_arg_list(args, "args")
        for i in range(len(arg_list)):
            arg_list[i] = sanitize_token(arg_list[i])
        if (gate == "p" or gate == "rx" or gate == "ry" or gate == "rz") and len(arg_list) == 2:
            if looks_like_angle(arg_list[0]) and looks_like_index(arg_list[1]):
                var tmp = arg_list[0]
                arg_list[0] = arg_list[1]
                arg_list[1] = tmp
        apply_gate(session, gate, arg_list)
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
        print_state(state, short=False, use_color=True)
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
        var state = compute_state(session)
        print_state_grid_colored_cells(
            state,
            col_bits,
            use_log=use_log,
            show_bin_labels=show_bin,
        )
        return ("Displayed grid.", True)
    if name == "list_tools":
        return (tools_summary(), True)
    raise Error("Unknown tool: " + name)
