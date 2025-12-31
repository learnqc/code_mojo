from math import log2, log10, sqrt, atan2, floor
from butterfly.core.types import *
from butterfly.core.state import QuantumState, ArrayState, GridState
from butterfly.core.circuit import QuantumCircuit, Transformation
from butterfly.core.circuit import (
    GateTransformation,
    SingleControlGateTransformation,
    MultiControlGateTransformation,
    BitReversalTransformation,
    UnitaryTransformation,
    ControlledUnitaryTransformation,
)
from collections import InlineArray

# https://github.com/holoviz/colorcet/blob/main/assets/colorcet.m
alias c6_data = InlineArray[Int, 768](
    246,
    54,
    26,
    246,
    56,
    23,
    246,
    58,
    20,
    246,
    60,
    18,
    246,
    64,
    16,
    246,
    67,
    13,
    247,
    71,
    11,
    247,
    76,
    10,
    247,
    80,
    8,
    248,
    84,
    6,
    248,
    89,
    5,
    249,
    93,
    4,
    250,
    98,
    3,
    250,
    102,
    3,
    251,
    107,
    2,
    251,
    111,
    1,
    252,
    116,
    1,
    252,
    120,
    0,
    253,
    124,
    0,
    253,
    128,
    0,
    253,
    132,
    0,
    254,
    137,
    0,
    254,
    141,
    0,
    255,
    145,
    0,
    255,
    149,
    0,
    255,
    153,
    0,
    255,
    156,
    0,
    255,
    160,
    0,
    255,
    164,
    0,
    255,
    168,
    0,
    255,
    172,
    0,
    255,
    175,
    0,
    255,
    179,
    0,
    255,
    182,
    0,
    255,
    186,
    0,
    255,
    189,
    0,
    254,
    192,
    0,
    253,
    195,
    0,
    252,
    198,
    0,
    250,
    200,
    0,
    248,
    202,
    0,
    246,
    204,
    0,
    244,
    206,
    0,
    241,
    207,
    0,
    239,
    208,
    0,
    235,
    209,
    0,
    232,
    209,
    0,
    229,
    209,
    0,
    225,
    209,
    0,
    221,
    208,
    0,
    217,
    207,
    0,
    212,
    206,
    0,
    208,
    205,
    0,
    204,
    204,
    0,
    199,
    202,
    0,
    195,
    200,
    0,
    190,
    199,
    0,
    185,
    197,
    0,
    181,
    195,
    0,
    176,
    193,
    0,
    171,
    191,
    0,
    167,
    189,
    0,
    162,
    188,
    0,
    157,
    186,
    0,
    152,
    184,
    0,
    148,
    182,
    0,
    143,
    180,
    0,
    138,
    178,
    0,
    133,
    176,
    1,
    128,
    174,
    1,
    123,
    172,
    2,
    119,
    170,
    2,
    114,
    168,
    3,
    109,
    166,
    4,
    104,
    164,
    5,
    99,
    163,
    6,
    94,
    161,
    8,
    89,
    159,
    10,
    85,
    158,
    13,
    80,
    157,
    15,
    76,
    155,
    18,
    71,
    154,
    21,
    67,
    154,
    24,
    63,
    153,
    27,
    59,
    153,
    30,
    56,
    153,
    33,
    53,
    153,
    37,
    50,
    153,
    41,
    48,
    154,
    45,
    46,
    155,
    49,
    45,
    156,
    53,
    44,
    157,
    57,
    43,
    159,
    62,
    43,
    161,
    66,
    42,
    163,
    71,
    43,
    165,
    76,
    43,
    167,
    81,
    44,
    169,
    85,
    44,
    171,
    90,
    45,
    174,
    95,
    46,
    176,
    100,
    47,
    178,
    105,
    47,
    181,
    110,
    48,
    183,
    115,
    48,
    186,
    120,
    49,
    188,
    125,
    49,
    191,
    130,
    50,
    193,
    135,
    50,
    196,
    140,
    50,
    198,
    145,
    50,
    201,
    151,
    50,
    204,
    156,
    49,
    206,
    161,
    49,
    209,
    166,
    49,
    211,
    171,
    48,
    213,
    176,
    47,
    216,
    181,
    46,
    218,
    186,
    45,
    220,
    191,
    44,
    222,
    196,
    43,
    224,
    201,
    42,
    226,
    206,
    41,
    228,
    210,
    40,
    229,
    215,
    39,
    230,
    219,
    38,
    231,
    223,
    37,
    232,
    227,
    37,
    232,
    231,
    37,
    232,
    234,
    37,
    231,
    237,
    37,
    230,
    240,
    38,
    229,
    242,
    38,
    228,
    245,
    39,
    226,
    246,
    40,
    224,
    248,
    41,
    222,
    249,
    42,
    220,
    251,
    43,
    217,
    252,
    44,
    214,
    252,
    45,
    212,
    253,
    46,
    209,
    254,
    47,
    206,
    254,
    48,
    203,
    255,
    48,
    200,
    255,
    48,
    197,
    255,
    49,
    193,
    255,
    49,
    190,
    255,
    49,
    187,
    255,
    48,
    184,
    255,
    48,
    181,
    255,
    48,
    178,
    255,
    47,
    175,
    255,
    46,
    171,
    255,
    46,
    168,
    255,
    45,
    165,
    255,
    44,
    162,
    255,
    43,
    159,
    255,
    42,
    156,
    255,
    41,
    153,
    255,
    40,
    150,
    255,
    39,
    147,
    255,
    39,
    145,
    255,
    39,
    142,
    255,
    40,
    140,
    255,
    41,
    137,
    255,
    43,
    135,
    255,
    45,
    133,
    255,
    49,
    132,
    255,
    52,
    130,
    255,
    57,
    129,
    255,
    61,
    128,
    255,
    66,
    128,
    255,
    72,
    128,
    255,
    77,
    128,
    255,
    83,
    128,
    255,
    88,
    129,
    255,
    94,
    130,
    255,
    99,
    131,
    255,
    105,
    132,
    255,
    110,
    133,
    255,
    116,
    135,
    255,
    121,
    137,
    255,
    127,
    139,
    255,
    132,
    141,
    255,
    137,
    143,
    255,
    142,
    145,
    255,
    147,
    147,
    255,
    151,
    149,
    255,
    156,
    151,
    255,
    161,
    153,
    255,
    165,
    156,
    255,
    170,
    158,
    255,
    174,
    160,
    255,
    178,
    163,
    255,
    183,
    165,
    255,
    187,
    167,
    255,
    191,
    169,
    255,
    195,
    172,
    255,
    199,
    174,
    255,
    203,
    176,
    255,
    207,
    178,
    255,
    211,
    180,
    255,
    215,
    182,
    255,
    219,
    184,
    255,
    223,
    186,
    255,
    227,
    187,
    254,
    230,
    188,
    253,
    234,
    189,
    251,
    237,
    190,
    249,
    240,
    191,
    247,
    244,
    191,
    244,
    247,
    190,
    241,
    249,
    190,
    238,
    252,
    189,
    234,
    254,
    188,
    230,
    255,
    186,
    226,
    255,
    184,
    221,
    255,
    182,
    217,
    255,
    180,
    212,
    255,
    177,
    207,
    255,
    174,
    201,
    255,
    171,
    196,
    255,
    168,
    191,
    255,
    165,
    185,
    255,
    161,
    180,
    255,
    158,
    174,
    255,
    154,
    169,
    255,
    151,
    163,
    255,
    147,
    158,
    255,
    143,
    152,
    255,
    140,
    147,
    255,
    136,
    142,
    255,
    132,
    136,
    255,
    128,
    131,
    255,
    124,
    125,
    255,
    120,
    120,
    255,
    116,
    115,
    255,
    112,
    109,
    255,
    108,
    104,
    255,
    104,
    99,
    255,
    100,
    94,
    255,
    96,
    89,
    255,
    92,
    84,
    255,
    88,
    79,
    255,
    83,
    74,
    254,
    79,
    69,
    253,
    75,
    64,
    252,
    71,
    60,
    251,
    68,
    55,
    250,
    64,
    51,
    249,
    61,
    47,
    249,
    58,
    43,
    248,
    56,
    39,
    247,
    54,
    35,
    247,
    53,
    32,
    246,
    53,
    29,
)


def get_color_code(re: FloatType, im: FloatType) -> String:
    var angle = atan2(im, re)
    var deg = angle * 180.0 / pi
    if deg < 0:
        deg += 360.0

    # Map 0-360 to 0-255
    var idx = Int(deg / 360.0 * 256.0) % 256

    var r = c6_data[3 * idx]
    var g = c6_data[3 * idx + 1]
    var b = c6_data[3 * idx + 2]

    # TrueColor ANSI: \033[38;2;R;G;Bm
    return "\033[38;2;" + String(r) + ";" + String(g) + ";" + String(b) + "m"


def get_bar(value: FloatType) -> String:
    if value >= 1:
        return "█" * Int(floor(value)) + get_bar(value - Int(floor(value)))
    elif value < 1 and value >= 7.0 / 8.0:
        return "▉"
    elif value < 7.0 / 8.0 and value >= 0.75:
        return "▊"
    elif value < 0.75 and value >= 5.0 / 8.0:
        return "▋"
    elif value < 5.0 / 8.0 and value >= 0.5:
        return "▌"
    elif value < 0.5 and value >= 3.0 / 8.0:
        return "▍"
    elif value < 3.0 / 8.0 and value >= 0.25:
        return "▎"
    elif value < 0.25 and value >= 1.0 / 8.0:
        return "▏"
    else:
        return " "


def to_table(
    s: QuantumState,
    prefix: Tuple[Int, Int] = (0, 0),
    decimals: Int = 3,
) -> List[List[String]]:
    n = Int(log2(Float32(s.size())))
    m = Int(log10(Float32(s.size())))

    # Access s[k] using __getitem__
    round_state = [
        Amplitude(round(s[k].re, decimals), round(s[k].im, decimals))
        for k in range(s.size())
    ]

    table: List[List[String]] = List[List[String]]()

    for k in range(s.size()):
        var row = List[String]()
        row.append(
            "￤" + String(k + prefix[0] * s.size()).rjust(max(5, m + 1), " ")
        )
        row.append(
            bin(k + prefix[0] * s.size(), prefix="").rjust(n + prefix[1], "0")
        )

        var re_str = (" " if round_state[k].re >= 0 else "-") + String(
            abs(round_state[k].re)
        )[: decimals + 2].rjust(decimals + 2, " ")
        var im_str = (
            (" + " if round_state[k].im >= 0 else " - ")
            + "i"
            + String(abs(round_state[k].im))[: decimals + 2].ljust(
                decimals + 2, " "
            )
        )
        row.append(re_str + im_str)

        var mag = sqrt(s[k].re * s[k].re + s[k].im * s[k].im)
        row.append(
            String(round(mag, decimals))[: decimals + 2].rjust(
                decimals + 2, " "
            )
        )

        var angle = atan2(s[k].im, s[k].re)
        var angle_deg: Float64 = 0.0
        if s[k].re != 0 or s[k].im != 0:
            angle_deg = abs(
                round(
                    angle.cast[DType.float64]()
                    / (2.0 * pi.cast[DType.float64]())
                    * 360.0,
                    2,
                )
            )
        var dir_str = (" " if angle >= 0 else "-") + String(angle_deg) + "°"
        row.append(dir_str.rjust(decimals + 6, " "))

        var mag_len = Int(floor(16 * mag))
        var mag_bar = get_bar(16 * mag)
        row.append(mag_bar + (" " * (15 - mag_len)))

        var prob = s[k].re * s[k].re + s[k].im * s[k].im
        row.append(String(round(prob, decimals)).rjust(decimals + 2, " "))

        var prob_len = Int(floor(16 * prob))
        prob_bar = get_bar(16 * prob)
        row.append(prob_bar + (" " * (15 - prob_len)))

        table.append(row^)

    return table^


def print_state(
    state: QuantumState,
    prefix: Tuple[Int, Int] = (0, 0),
    short: Bool = True,
    use_color: Bool = True,
):
    rows = 16 if short else max(16, state.size())

    var sub_re = List[FloatType]()
    var sub_im = List[FloatType]()
    limit = min(state.size(), rows)
    for i in range(limit):
        sub_re.append(state[i].re)
        sub_im.append(state[i].im)
    var sub_state = QuantumState(sub_re^, sub_im^)
    table = to_table(sub_state, prefix, 3)

    headers = [
        "￤" + "Out".center(len(table[0][0]) - 2, " "),
        "Bin".center(len(table[0][1]) + 1, " "),
        "Ampl".center(len(table[0][2]) + 1, " "),
        "Mag".center(len(table[0][3]) + 1, " "),
        "Dir".center(len(table[0][4]), " "),
        "Ampl Bar".center(17, " "),
        "Prob".center(len(table[0][6]) + 1, " "),
        "Prob Bar".center(17, " "),
    ]

    print(" ", end="")
    print("-" * 97, end="\n")
    for i in range(len(headers)):
        print(headers[i], end="￤")
    print()
    print(" ", end="")
    print("-" * 97, end="")

    real_color_code = get_color_code(1, 0)
    reset_color_code = "\033[0m"

    for i in range(len(table)):
        print("\n")

        for j in range(len(table[i])):
            var cell = table[i][j]
            if use_color:
                if j == 5:  # MagBar
                    cell = (
                        get_color_code(sub_state[i].re, sub_state[i].im)
                        + cell
                        + reset_color_code
                    )
                if j == 7:  # ProbBar
                    cell = real_color_code + cell + reset_color_code
            print(cell, end=" ￤")
    print("\n")


def print_state_a(
    state: ArrayState,
    prefix: Tuple[Int, Int] = (0, 0),
    short: Bool = True,
    use_color: Bool = True,
):
    var re = List[FloatType]()
    var im = List[FloatType]()
    for i in range(len(state)):
        re.append(state[i].re)
        im.append(state[i].im)
    var qs = QuantumState(re^, im^)
    print_state(qs, prefix, short, use_color)


def print_grid_state(
    state: GridState, short: Bool = True, use_color: Bool = True
):
    col_bits = Int(log10(Float32(len(state[0]))))
    for r in range(len(state)):
        var row_re = List[FloatType]()
        var row_im = List[FloatType]()
        for c in range(len(state[r])):
            row_re.append(state[r][c].re)
            row_im.append(state[r][c].im)
        var qs = QuantumState(row_re^, row_im^)
        print_state(qs, (Int(r), Int(col_bits)), short, use_color)


def get_bg_color_code(
    re: FloatType, im: FloatType, max_amp: FloatType, use_log: Bool = False
) -> String:
    var angle = atan2(im, re)
    var deg = angle * 180.0 / pi
    if deg < 0:
        deg += 360.0

    # Map 0-360 to 0-255
    var idx = Int(deg / 360.0 * 256.0) % 256

    var r = FloatType(c6_data[3 * idx])
    var g = FloatType(c6_data[3 * idx + 1])
    var b = FloatType(c6_data[3 * idx + 2])

    # Scale by intensity
    var mag = sqrt(re * re + im * im)
    var intensity: FloatType = 0.0

    if use_log:
        if max_amp > 1e-9:
            # log10(1+x) scaling
            intensity = log10(1.0 + mag) / log10(1.0 + max_amp)
    else:
        intensity = mag / max_amp

    # Apply intensity scaling
    r = r * intensity
    g = g * intensity
    b = b * intensity

    # TrueColor ANSI Background: \033[48;2;R;G;Bm
    return (
        "\033[48;2;"
        + String(Int(r))
        + ";"
        + String(Int(g))
        + ";"
        + String(Int(b))
        + "m"
    )


def get_alpha_char(intensity: FloatType) -> String:
    if intensity > 0.6:
        return "█"
    elif intensity > 0.3:
        return "▓"
    elif intensity > 0.1:
        return "▒"
    elif intensity > 0.001:
        return "░"
    else:
        return " "


def print_state_colored_cells(
    state: QuantumState, max_width: Int = 80, use_log: Bool = False
):
    """
    Prints the quantum state as a row of colored cells.
    Hue = Phase (Foreground), Alpha = Amplitude Magnitude (Character Density).
    """
    var N = len(state)

    # Absolute scaling ref
    var ref_amp: FloatType = 1.0

    var row = String("")
    for i in range(N):
        var amp = state[i]

        # 1. Unscaled Color (Preserve Hue)
        var fg_code = get_color_code(amp.re, amp.im)

        # 2. Intensity (Alpha) calculation
        var mag = sqrt(amp.re * amp.re + amp.im * amp.im)
        if use_log:
            var intensity = log10(1.0 + mag) / log10(1.0 + ref_amp)
            var char = get_alpha_char(intensity)
            row += fg_code + char + char + "\033[0m" + "  "
        else:
            var intensity = mag / ref_amp
            var char = get_alpha_char(intensity)
            row += fg_code + char + char + "\033[0m" + "  "

    print(row)


def print_grid_state_colored_cells(
    state: GridState,
    use_log: Bool = False,
    origin_bottom: Bool = False,
    signed_y: Bool = False,
):
    """
    Prints the GridState as a 2D grid of colored cells with indices and grid lines.
    """
    var ref_amp: FloatType = 1.0
    var rows = len(state)
    if rows == 0:
        return
    var cols = len(state[0])

    # Calculate padding for row indices
    var row_idx_width = len(String(rows - 1))
    if signed_y:
        # Two's complement largest negative can be "-rows/2", which is same length or +1
        row_idx_width = len(String(-(rows // 2)))

    var h_line_seg = "----+"
    var h_line = " " * (row_idx_width + 1) + "+"
    for _ in range(cols):
        h_line += h_line_seg

    print(h_line)

    # 1. Print Rows
    # ... (row_indices calculation remains the same)
    var row_indices = List[Int]()
    if not signed_y:
        if origin_bottom:
            for i in range(rows - 1, -1, -1):
                row_indices.append(i)
        else:
            for i in range(rows):
                row_indices.append(i)
    else:
        for i in range(rows // 2 - 1, -1, -1):
            row_indices.append(i)
        for i in range(rows - 1, rows - rows // 2 - 1, -1):
            row_indices.append(i)

    for r in row_indices:
        var r_val = r
        if signed_y and r >= (rows // 2):
            r_val = r - rows

        # Row Index Label
        var r_str = String(r_val)
        var pad = " " * (row_idx_width - len(r_str))
        var line = pad + r_str + " |"

        for c in range(cols):
            var amp = state[r][c]
            var fg_code = get_color_code(amp.re, amp.im)
            var mag = sqrt(amp.re * amp.re + amp.im * amp.im)

            # Symmetrical centering in 4-char cell: " XX |"
            if use_log:
                var intensity = log10(1.0 + mag) / log10(1.0 + ref_amp)
                var char = get_alpha_char(intensity)
                line += fg_code + " " + char + char + " " + "\033[0m" + "|"
            else:
                var intensity = mag / ref_amp
                var char = get_alpha_char(intensity)
                line += fg_code + " " + char + char + " " + "\033[0m" + "|"

        print(line)
        print(h_line)

    # 2. Print Bottom Axis
    # Prefix length: row_idx_width + 1 (space) + 1 (pipe) = width + 2.
    var axis_pad = " " * (row_idx_width + 2)
    var axis_line = axis_pad

    # Sparse Labeling
    var col_active = List[Bool](capacity=cols)
    for c in range(cols):
        var active = False
        for r in range(rows):
            var amp = state[r][c]
            var mag = sqrt(amp.re * amp.re + amp.im * amp.im)
            if mag > 0.01:
                active = True
                break
        col_active.append(active)

    for c in range(cols):
        var s = String(c)
        var slot_width = 5  # Matches "----+"
        var should_print = col_active[c]

        if should_print:
            # Center label in the 4-char content area (index 0..3)
            # " XX |" -> we want label under the XX
            # If s is "0", we want "  0  "
            # If s is "10", we want " 10  " or "  10 "
            var label_pad = " " * ((4 - len(s)) // 2 + 1)
            var centered_s = label_pad + s
            axis_line += centered_s + " " * (slot_width - len(centered_s))
        else:
            axis_line += " " * slot_width

    print(axis_line)


def print_grid_state_colored_cells(
    state: QuantumState,
    num_rows: Int = 1,
    use_log: Bool = False,
    origin_bottom: Bool = False,
    signed_y: Bool = False,
):
    """
    Prints the QuantumState as a 2D grid of colored cells by reshaping it.
    """
    var size = state.size()
    var num_cols = size // num_rows
    if num_cols * num_rows != size:
        # Fallback to single row if invalid num_rows
        # print("Warning: num_rows does not divide state size. Using 1 row.")
        var grid = GridState()
        var row = List[Amplitude](capacity=size)
        for i in range(size):
            row.append(state[i])
        grid.append(row^)
        print_grid_state_colored_cells(grid, use_log, origin_bottom, signed_y)
        return

    var grid = GridState()
    for r in range(num_rows):
        var row = List[Amplitude](capacity=num_cols)
        for c in range(num_cols):
            # Reshape logic: (row << col_bits) + col?
            # In our function encoding demo, it was (col << row_bits) + row.
            # But the 'num_rows' parameter implies we want 'num_rows' vertical cells.
            # Usually row-major is (r * num_cols) + c.
            # Let's stick to standard row-major for this utility, UNLESS the user
            # specifically wants the function encoding layout (Value vs Key).
            # If the user provides num_rows, they likely expect row-major.
            var idx = r * num_cols + c
            row.append(state[idx])
        grid.append(row^)

    print_grid_state_colored_cells(grid, use_log, origin_bottom, signed_y)


def print_circuit(qc: QuantumCircuit):
    """Prints a simple ASCII representation of the quantum circuit."""

    var n = qc.num_qubits
    var num_transforms = qc.num_transformations()

    if num_transforms == 0:
        print("Empty Circuit")
        return

    var rows = List[String]()
    var name_width = len(String(n - 1)) + 2
    for q in range(n):
        rows.append("q" + String(q).rjust(name_width - 2, "0") + ": ──")

    # Initial vertical line with qubit names

    for i in range(num_transforms):
        var t = qc.transformations[i].copy()
        var targets = List[Int]()
        var controls = List[Int]()
        var label: String

        if t.isa[GateTransformation]():
            var gt = t[GateTransformation].copy()
            targets.append(gt.target)
            if gt.name in List[String]("rx", "ry", "rz", "p"):
                label = gt.name.upper() + "(" + String(gt.arg)[:4] + ")"
            else:
                label = gt.name.upper()
        elif t.isa[SingleControlGateTransformation]():
            var sct = t[SingleControlGateTransformation].copy()
            targets.append(sct.target)
            controls.append(sct.control)
            if sct.name in List[String]("rx", "ry", "rz", "p"):
                label = sct.name.upper() + "(" + String(sct.arg)[:4] + ")"
            else:
                label = sct.name.upper()
        elif t.isa[MultiControlGateTransformation]():
            var mct = t[MultiControlGateTransformation].copy()
            targets.append(mct.target)
            for j in range(len(mct.controls)):
                controls.append(mct.controls[j])
            if mct.name in List[String]("rx", "ry", "rz", "p"):
                label = mct.name.upper() + "(" + String(mct.arg)[:4] + ")"
            else:
                label = mct.name.upper()
        elif t.isa[BitReversalTransformation]():
            label = "REV"
            for j in range(n):
                targets.append(j)
        elif t.isa[UnitaryTransformation]():
            var ut = t[UnitaryTransformation].copy()
            label = ut.name.upper()
            for j in range(ut.m):
                targets.append(ut.target + j)
        elif t.isa[ControlledUnitaryTransformation]():
            var cut = t[ControlledUnitaryTransformation].copy()
            label = cut.name.upper()
            controls.append(cut.control)
            for j in range(cut.m):
                targets.append(cut.target + j)
        else:
            label = "???"

        var field_width = len(label) + 2

        for q in range(n):
            var is_target = False
            for j in range(len(targets)):
                if targets[j] == q:
                    is_target = True
                    break
            var is_control = False
            for j in range(len(controls)):
                if controls[j] == q:
                    is_control = True
                    break

            if is_target:
                rows[q] += "[" + label + "]──"
            elif is_control:
                var pad_l = field_width // 2
                var pad_r = field_width - pad_l - 1
                rows[q] += ("─" * pad_l) + "●" + ("─" * pad_r) + "──"
            else:
                rows[q] += "─" * (field_width + 2)

    for q in range(n):
        print(rows[q])
