from butterfly.core.types import Complex, FloatType
from butterfly.core.state import State
from butterfly.core.executors import execute
from butterfly.core.quantum_circuit import QuantumCircuit, QuantumTransformation
from butterfly.core.circuit import (
    GateTransformation,
    FusedPairTransformation,
    SwapTransformation,
    QubitReversalTransformation,
    UnitaryTransformation,
    ControlledUnitaryTransformation,
    ClassicalTransformation,
    MeasurementTransformation,
)
from butterfly.utils.context import ExecContext
from time import sleep
from math import log2, log10, sqrt, atan2, floor
from butterfly.core.types import pi

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

def get_bg_color_code(re: FloatType, im: FloatType) -> String:
    var angle = atan2(im, re)
    var deg = angle * 180.0 / pi
    if deg < 0:
        deg += 360.0

    var idx = Int(deg / 360.0 * 256.0) % 256
    var r = c6_data[3 * idx]
    var g = c6_data[3 * idx + 1]
    var b = c6_data[3 * idx + 2]

    # TrueColor ANSI background: \033[48;2;R;G;Bm
    return "\033[48;2;" + String(r) + ";" + String(g) + ";" + String(b) + "m"

def get_bg_color_code_intensity(
    re: FloatType, im: FloatType, intensity: FloatType
) -> String:
    var angle = atan2(im, re)
    var deg = angle * 180.0 / pi
    if deg < 0:
        deg += 360.0

    var idx = Int(deg / 360.0 * 256.0) % 256
    var r = FloatType(c6_data[3 * idx])
    var g = FloatType(c6_data[3 * idx + 1])
    var b = FloatType(c6_data[3 * idx + 2])

    var scale = intensity
    if scale < 0.0:
        scale = 0.0
    if scale > 1.0:
        scale = 1.0
    # Blend toward white for low intensity (washed out), full color at high.
    r = r * scale + 255.0 * (1.0 - scale)
    g = g * scale + 255.0 * (1.0 - scale)
    b = b * scale + 255.0 * (1.0 - scale)

    return (
        "\033[48;2;"
        + String(Int(r))
        + ";"
        + String(Int(g))
        + ";"
        + String(Int(b))
        + "m"
    )


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
    s: State,
    prefix: Tuple[Int, Int] = (0, 0),
    decimals: Int = 3,
) -> List[List[String]]:
    n = Int(log2(Float64(s.size())))
    m = Int(log10(Float64(s.size())))

    # Access s[k] using __getitem__
    round_state = [
        Complex(round(s[k].re, decimals), round(s[k].im, decimals))
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

        var amp = s[k]
        var mag = sqrt(amp.re * amp.re + amp.im * amp.im)
        row.append(
            String(round(mag, decimals))[: decimals + 2].rjust(
                decimals + 2, " "
            )
        )

        var angle = atan2(amp.im, amp.re)
        var angle_deg: FloatType = 0.0
        if amp.re != 0 or amp.im != 0:
            angle_deg = abs(
                round(
                    angle
                    / (2.0 * pi)
                    * 360.0,
                    2,
                )
            )
        var dir_str = (" " if angle >= 0 else "-") + String(angle_deg) + "°"
        row.append(dir_str.rjust(decimals + 6, " "))

        var mag_len = Int(floor(16 * mag))
        var mag_bar = get_bar(16 * mag)
        row.append(mag_bar + (" " * (15 - mag_len)))

        var prob = amp.re * amp.re + amp.im * amp.im
        row.append(String(round(prob, decimals)).rjust(decimals + 2, " "))

        var prob_len = Int(floor(16 * prob))
        var prob_bar = get_bar(16 * prob)
        row.append(prob_bar + (" " * (15 - prob_len)))

        table.append(row^)

    return table^


def print_state(
    state: State,
    prefix: Tuple[Int, Int] = (0, 0),
    short: Bool = True,
    use_color: Bool = True,
    left_pad: Int = 0,
    max_rows: Int = 0,
):
    var rows = 16 if short else max(16, state.size())
    if max_rows > 0:
        rows = max_rows

    var sub_re = List[FloatType]()
    var sub_im = List[FloatType]()
    limit = min(state.size(), rows)
    for i in range(limit):
        sub_re.append(state[i].re)
        sub_im.append(state[i].im)
    var sub_state = State(sub_re^, sub_im^)
    table = to_table(sub_state, prefix, 3)

    headers = [
        "|" + "Out".center(len(table[0][0]) - 2, " "),
        "Bin".center(len(table[0][1]) + 1, " "),
        "Ampl".center(len(table[0][2]) + 1, " "),
        "Mag".center(len(table[0][3]) + 1, " "),
        "Dir".center(len(table[0][4]), " "),
        "Ampl Bar".center(17, " "),
        "Prob".center(len(table[0][6]) + 1, " "),
        "Prob Bar".center(17, " "),
    ]

    var pad = ""
    if left_pad > 0:
        pad = " " * left_pad
    print(pad + " " + "-" * 97)
    var header_line = ""
    for i in range(len(headers)):
        header_line += headers[i] + "|"
    print(pad + header_line)
    print(pad + " " + "-" * 97)

    real_color_code = get_color_code(1, 0)
    reset_color_code = "\033[0m"

    for i in range(len(table)):
        print(pad)
        var row_line = ""
        for j in range(len(table[i])):
            var cell = table[i][j].replace("￤", "|")
            if use_color:
                if j == 5:  # MagBar
                    cell = (
                        get_color_code(sub_state[i].re, sub_state[i].im)
                        + cell
                        + reset_color_code
                    )
                if j == 7:  # ProbBar
                    cell = real_color_code + cell + reset_color_code
            row_line += cell + " |"
        print(pad + row_line)
    print(pad)


fn _visible_len(text: String) raises -> Int:
    var length = 0
    var i = 0
    var n = len(text)
    var esc = "\033"[0]
    var bracket = "["[0]
    var m_char = "m"[0]
    while i < n:
        if text[i] == esc:
            i += 1
            if i < n and text[i] == bracket:
                i += 1
                while i < n and text[i] != m_char:
                    i += 1
                if i < n:
                    i += 1
                continue
        length += 1
        i += 1
    return length


def _merge_frame_lines(
    left_lines: List[String],
    right_lines: List[String],
    gap: Int,
) -> List[String]:
    var out = List[String]()
    var max_lines = max(len(left_lines), len(right_lines))
    var left_width = 0
    for i in range(len(left_lines)):
        var width = _visible_len(left_lines[i])
        if width > left_width:
            left_width = width
    var gap_spaces = ""
    for _ in range(max(0, gap)):
        gap_spaces += " "
    for i in range(max_lines):
        var left = "" if i >= len(left_lines) else left_lines[i]
        var right = "" if i >= len(right_lines) else right_lines[i]
        var pad = left_width - _visible_len(left)
        if pad < 0:
            pad = 0
        var padding = ""
        for _ in range(pad):
            padding += " "
        out.append(left + padding + gap_spaces + right)
    return out^


def _merge_frame_lines_fixed(
    left_lines: List[String],
    right_lines: List[String],
    gap: Int,
    left_width: Int,
) -> List[String]:
    var out = List[String]()
    var max_lines = max(len(left_lines), len(right_lines))
    var fixed_width = left_width
    if fixed_width <= 0:
        for i in range(len(left_lines)):
            var width = _visible_len(left_lines[i])
            if width > fixed_width:
                fixed_width = width
    var gap_spaces = ""
    for _ in range(max(0, gap)):
        gap_spaces += " "
    for i in range(max_lines):
        var left = "" if i >= len(left_lines) else left_lines[i]
        var right = "" if i >= len(right_lines) else right_lines[i]
        var pad = fixed_width - _visible_len(left)
        if pad < 0:
            pad = 0
        var padding = ""
        for _ in range(pad):
            padding += " "
        out.append(left + padding + gap_spaces + right)
    return out^


struct FrameIterator(Movable):
    var circuit: QuantumCircuit
    var state: State
    var step: Int
    var total: Int
    var kind: Int
    var col_bits: Int
    var use_log: Bool
    var origin_bottom: Bool
    var show_bin_labels: Bool
    var use_bg: Bool
    var show_chars: Bool
    var short: Bool
    var use_color: Bool
    var show_step_label: Bool
    var left_pad: Int
    var row_separators: Bool
    var max_rows: Int
    var exp_bits: Int
    var value_bits: Int

    fn frame_lines(self) raises -> List[String]:
        if self.step > self.total:
            return List[String]()
        var label = "init" if self.step == 0 else ""
        if self.kind == 0:
            return render_table_frame_lines(
                self.state,
                self.step,
                self.total,
                label,
                self.short,
                self.use_color,
                self.show_step_label,
                self.left_pad,
                row_separators=self.row_separators,
                max_rows=self.max_rows,
            )
        if self.kind == 1:
            return render_grid_frame_lines(
                self.state,
                self.step,
                self.total,
                label,
                self.col_bits,
                self.use_log,
                self.origin_bottom,
                self.use_bg,
                self.show_bin_labels,
                self.show_chars,
                self.show_step_label,
                self.left_pad,
            )
        return render_exp_marginal_table_lines(
            self.state,
            self.step,
            self.total,
            label,
            self.exp_bits,
            self.value_bits,
            self.short,
            self.use_color,
            self.show_step_label,
            self.left_pad,
            row_separators=self.row_separators,
            max_rows=self.max_rows,
        )

    fn frame_width(self) raises -> Int:
        var lines = self.frame_lines()
        var width = 0
        for i in range(len(lines)):
            var w = _visible_len(lines[i])
            if w > width:
                width = w
        return width

    fn frame_height(self) raises -> Int:
        var lines = self.frame_lines()
        return len(lines)

    fn __init__(
        out self,
        var circuit: QuantumCircuit,
        var state: State,
        step: Int,
        total: Int,
        kind: Int,
        col_bits: Int,
        use_log: Bool,
        origin_bottom: Bool,
        show_bin_labels: Bool,
        use_bg: Bool,
        show_chars: Bool,
        short: Bool,
        use_color: Bool,
        show_step_label: Bool,
        left_pad: Int,
        row_separators: Bool,
        max_rows: Int,
        exp_bits: Int,
        value_bits: Int,
    ):
        self.circuit = circuit^
        self.state = state^
        self.step = step
        self.total = total
        self.kind = kind
        self.col_bits = col_bits
        self.use_log = use_log
        self.origin_bottom = origin_bottom
        self.show_bin_labels = show_bin_labels
        self.use_bg = use_bg
        self.show_chars = show_chars
        self.short = short
        self.use_color = use_color
        self.show_step_label = show_step_label
        self.left_pad = left_pad
        self.row_separators = row_separators
        self.max_rows = max_rows
        self.exp_bits = exp_bits
        self.value_bits = value_bits

    @staticmethod
    fn grid(
        var circuit: QuantumCircuit,
        var state: State,
        col_bits: Int,
        use_log: Bool = False,
        origin_bottom: Bool = False,
        show_bin_labels: Bool = False,
        use_bg: Bool = True,
        show_chars: Bool = False,
        show_step_label: Bool = True,
        left_pad: Int = 0,
    ) -> FrameIterator:
        var it = FrameIterator(
            circuit^,
            state^,
            0,
            len(circuit.transformations),
            1,
            col_bits,
            use_log,
            origin_bottom,
            show_bin_labels,
            use_bg,
            show_chars,
            True,
            True,
            show_step_label,
            left_pad,
            False,
            0,
            0,
            0,
        )
        return it^

    @staticmethod
    fn table(
        var circuit: QuantumCircuit,
        var state: State,
        short: Bool = True,
        use_color: Bool = True,
        show_step_label: Bool = True,
        left_pad: Int = 0,
        row_separators: Bool = False,
        max_rows: Int = 0,
    ) -> FrameIterator:
        var it = FrameIterator(
            circuit^,
            state^,
            0,
            len(circuit.transformations),
            0,
            0,
            False,
            False,
            False,
            True,
            False,
            short,
            use_color,
            show_step_label,
            left_pad,
            row_separators,
            max_rows,
            0,
            0,
        )
        return it^

    @staticmethod
    fn exp_marginal_table(
        var circuit: QuantumCircuit,
        var state: State,
        exp_bits: Int,
        value_bits: Int,
        short: Bool = True,
        use_color: Bool = True,
        show_step_label: Bool = True,
        left_pad: Int = 0,
        row_separators: Bool = False,
        max_rows: Int = 0,
    ) -> FrameIterator:
        var it = FrameIterator(
            circuit^,
            state^,
            0,
            len(circuit.transformations),
            2,
            0,
            False,
            False,
            False,
            True,
            False,
            short,
            use_color,
            show_step_label,
            left_pad,
            row_separators,
            max_rows,
            exp_bits,
            value_bits,
        )
        return it^

    fn next(mut self) raises -> Optional[List[String]]:
        if self.step > self.total:
            return None
        var lines = self.frame_lines()
        self.advance()
        return lines^

    fn advance(mut self) raises:
        if self.step < self.total:
            var tr = self.circuit.transformations[self.step]
            var step_circuit = QuantumCircuit(self.circuit.num_qubits)
            step_circuit.transformations.append(tr.copy())
            execute(self.state, step_circuit, ExecContext())
        self.step += 1


struct FrameSource(Movable):
    var iter: FrameIterator

    @staticmethod
    fn grid(
        var circuit: QuantumCircuit,
        var state: State,
        col_bits: Int,
        use_log: Bool = False,
        origin_bottom: Bool = False,
        show_bin_labels: Bool = False,
        use_bg: Bool = True,
        show_chars: Bool = False,
        show_step_label: Bool = True,
        left_pad: Int = 0,
    ) -> FrameSource:
        var it = FrameIterator.grid(
            circuit^,
            state^,
            col_bits,
            use_log=use_log,
            origin_bottom=origin_bottom,
            show_bin_labels=show_bin_labels,
            use_bg=use_bg,
            show_chars=show_chars,
            show_step_label=show_step_label,
            left_pad=left_pad,
        )
        return FrameSource(it^)

    @staticmethod
    fn table(
        var circuit: QuantumCircuit,
        var state: State,
        short: Bool = True,
        use_color: Bool = True,
        show_step_label: Bool = True,
        left_pad: Int = 0,
        row_separators: Bool = False,
        max_rows: Int = 0,
    ) -> FrameSource:
        var it = FrameIterator.table(
            circuit^,
            state^,
            short=short,
            use_color=use_color,
            show_step_label=show_step_label,
            left_pad=left_pad,
            row_separators=row_separators,
            max_rows=max_rows,
        )
        return FrameSource(it^)

    @staticmethod
    fn exp_marginal_table(
        var circuit: QuantumCircuit,
        var state: State,
        exp_bits: Int,
        value_bits: Int,
        short: Bool = True,
        use_color: Bool = True,
        show_step_label: Bool = True,
        left_pad: Int = 0,
        row_separators: Bool = False,
        max_rows: Int = 0,
    ) -> FrameSource:
        var it = FrameIterator.exp_marginal_table(
            circuit^,
            state^,
            exp_bits,
            value_bits,
            short=short,
            use_color=use_color,
            show_step_label=show_step_label,
            left_pad=left_pad,
            row_separators=row_separators,
            max_rows=max_rows,
        )
        return FrameSource(it^)

    fn __init__(out self, var iter: FrameIterator):
        self.iter = iter^

    fn next_frame(mut self) raises -> Optional[List[String]]:
        return self.iter.next()


fn animate_frame_source_pair(
    mut left: FrameSource,
    mut right: FrameSource,
    left_width: Int = 0,
    gap: Int = 4,
    delay_s: Float64 = 0.0,
    redraw_in_place: Bool = True,
    step_on_input: Bool = False,
) raises:
    """Animate two full-frame sources side-by-side."""
    @always_inline
    fn wait_for_step_delta() raises -> Int:
        from python import Python

        var sys = Python.import_module("sys")
        var termios = Python.import_module("termios")
        var tty = Python.import_module("tty")
        var stdin = sys.stdin
        var fd = stdin.fileno()
        var old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            var ch = String(stdin.read(1))
            if ch == "\r" or ch == "\n" or ch == " ":
                return 1
            if ch == "\x7f":
                return -1
            if ch == "q" or ch == "Q" or ch == "\x03":
                return 2
            if ch == "\x1b":
                var ch2 = String(stdin.read(1))
                if ch2 == "[":
                    var ch3 = String(stdin.read(1))
                    if ch3 == "C":
                        return 1
                    if ch3 == "D":
                        return -1
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return 0

    var fixed_width = left_width
    if fixed_width <= 0:
        fixed_width = left.iter.frame_width()
    var total_left = left.iter.total
    var total_right = right.iter.total
    var total = max(total_left, total_right)
    var step = 0
    var left_init = left.iter.state.copy()
    var right_init = right.iter.state.copy()
    var last_left_lines = List[String]()
    var last_right_lines = List[String]()

    @always_inline
    fn render_pair_with_cache(
        step: Int,
        mut left_ref: FrameSource,
        mut right_ref: FrameSource,
        mut last_left: List[String],
        mut last_right: List[String],
        gap: Int,
        fixed_width: Int,
        redraw_in_place: Bool,
        total_left: Int,
        total_right: Int,
    ) raises:
        var left_step = step
        var right_step = step
        if left_step > total_left:
            left_step = total_left
        if right_step > total_right:
            right_step = total_right
        left_ref.iter.step = left_step
        right_ref.iter.step = right_step
        var left_lines = left_ref.iter.frame_lines()
        var right_lines = right_ref.iter.frame_lines()
        if len(left_lines) > 0:
            last_left = left_lines.copy()
        else:
            left_lines = last_left.copy()
        if len(right_lines) > 0:
            last_right = right_lines.copy()
        else:
            right_lines = last_right.copy()
        var merged = _merge_frame_lines_fixed(
            left_lines,
            right_lines,
            gap,
            fixed_width,
        )
        if redraw_in_place:
            print("\033c", end="")
            print("\033[H", end="")
        for i in range(len(merged)):
            print(merged[i])

    render_pair_with_cache(
        step,
        left,
        right,
        last_left_lines,
        last_right_lines,
        gap,
        fixed_width,
        redraw_in_place,
        total_left,
        total_right,
    )
    while True:
        if step_on_input:
            var delta = wait_for_step_delta()
            if delta == 0:
                continue
            if delta == 2:
                return
            if delta < 0:
                if step == 0:
                    continue
                var target = step - 1
                var ok = True
                if target < total_left:
                    try:
                        var last_tr = left.iter.circuit.transformations[target]
                        if last_tr.isa[ClassicalTransformation[State]]():
                            ok = False
                        else:
                            var single = QuantumCircuit(left.iter.circuit.num_qubits)
                            single.transformations.append(last_tr.copy())
                            var inv_single = single.inverse()
                            execute(left.iter.state, inv_single, ExecContext())
                    except:
                        ok = False
                if target < total_right:
                    try:
                        var last_tr = right.iter.circuit.transformations[target]
                        if last_tr.isa[ClassicalTransformation[State]]():
                            ok = False
                        else:
                            var single = QuantumCircuit(right.iter.circuit.num_qubits)
                            single.transformations.append(last_tr.copy())
                            var inv_single = single.inverse()
                            execute(right.iter.state, inv_single, ExecContext())
                    except:
                        ok = False
                if not ok:
                    left.iter.state = left_init.copy()
                    right.iter.state = right_init.copy()
                    if target > 0:
                        if target <= total_left:
                            var sub_left = QuantumCircuit(left.iter.circuit.num_qubits)
                            for i in range(target):
                                sub_left.transformations.append(
                                    left.iter.circuit.transformations[i].copy()
                                )
                            execute(left.iter.state, sub_left, ExecContext())
                        if target <= total_right:
                            var sub_right = QuantumCircuit(
                                right.iter.circuit.num_qubits
                            )
                            for i in range(target):
                                sub_right.transformations.append(
                                    right.iter.circuit.transformations[i].copy()
                                )
                            execute(right.iter.state, sub_right, ExecContext())
                step = target
                render_pair_with_cache(
                    step,
                    left,
                    right,
                    last_left_lines,
                    last_right_lines,
                    gap,
                    fixed_width,
                    redraw_in_place,
                    total_left,
                    total_right,
                )
                continue
            if delta > 0:
                if step >= total:
                    continue
                if step < total_left:
                    left.iter.advance()
                if step < total_right:
                    right.iter.advance()
                step += 1
                render_pair_with_cache(
                    step,
                    left,
                    right,
                    last_left_lines,
                    last_right_lines,
                    gap,
                    fixed_width,
                    redraw_in_place,
                    total_left,
                    total_right,
                )
                continue
        else:
            if step >= total:
                break
            if step < total_left:
                left.iter.advance()
            if step < total_right:
                right.iter.advance()
            step += 1
            render_pair_with_cache(
                step,
                left,
                right,
                last_left_lines,
                last_right_lines,
                gap,
                fixed_width,
                redraw_in_place,
                total_left,
                total_right,
            )
            if delay_s > 0.0:
                sleep(delay_s)



fn render_table_frame_lines(
    state: State,
    step: Int,
    total: Int,
    label: String,
    short: Bool,
    use_color: Bool,
    show_step_label: Bool,
    left_pad: Int = 0,
    row_separators: Bool = False,
    max_rows: Int = 0,
) raises -> List[String]:
    var lines = List[String]()
    var pad = ""
    if left_pad > 0:
        pad = " " * left_pad
    if show_step_label:
        var header = "Step " + String(step) + "/" + String(total)
        if label != "":
            header += ": " + label
        lines.append(pad + header)

    var rows = 16 if short else max(16, state.size())
    if max_rows > 0:
        rows = max_rows
    var sub_re = List[FloatType]()
    var sub_im = List[FloatType]()
    limit = min(state.size(), rows)
    for i in range(limit):
        sub_re.append(state[i].re)
        sub_im.append(state[i].im)
    var sub_state = State(sub_re^, sub_im^)
    table = to_table(sub_state, (0, 0), 3)

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

    lines.append(pad + " " + "-" * 97)
    var header_line = ""
    for i in range(len(headers)):
        header_line += headers[i] + "￤"
    lines.append(pad + header_line)
    lines.append(pad + " " + "-" * 97)

    real_color_code = get_color_code(1, 0)
    reset_color_code = "\033[0m"

    for i in range(len(table)):
        var row_line = ""
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
            row_line += cell + " ￤"
        lines.append(pad + row_line)
        if row_separators:
            lines.append(pad + " " + "-" * 97)
    return lines^


fn _exp_marginal_state(
    state: State,
    exp_bits: Int,
    value_bits: Int,
) -> State:
    var total_bits = exp_bits + value_bits
    var size = state.size()
    if total_bits <= 0 or size != (1 << total_bits):
        return State(List[FloatType](), List[FloatType]())
    var exp_size = 1 << exp_bits
    var re = List[FloatType](length=exp_size, fill=0.0)
    var im = List[FloatType](length=exp_size, fill=0.0)
    var mask = exp_size - 1
    for idx in range(size):
        var x = idx & mask
        var amp = state[idx]
        var prob = amp.re * amp.re + amp.im * amp.im
        re[x] += prob
    for i in range(exp_size):
        if re[i] > 0.0:
            re[i] = sqrt(re[i])
    return State(re^, im^)


fn render_exp_marginal_table_lines(
    state: State,
    step: Int,
    total: Int,
    label: String,
    exp_bits: Int,
    value_bits: Int,
    short: Bool,
    use_color: Bool,
    show_step_label: Bool,
    left_pad: Int,
    row_separators: Bool = False,
    max_rows: Int = 0,
) raises -> List[String]:
    var projected = _exp_marginal_state(state, exp_bits, value_bits)
    return render_table_frame_lines(
        projected,
        step,
        total,
        label,
        short,
        use_color,
        show_step_label,
        left_pad,
        row_separators=row_separators,
        max_rows=max_rows,
    )


fn render_grid_frame_lines(
    state: State,
    step: Int,
    total: Int,
    label: String,
    var col_bits: Int,
    use_log: Bool,
    origin_bottom: Bool,
    use_bg: Bool,
    show_bin_labels: Bool,
    show_chars: Bool,
    show_step_label: Bool,
    left_pad: Int = 0,
) raises -> List[String]:
    var lines = List[String]()
    var pad = ""
    if left_pad > 0:
        pad = " " * left_pad
    if show_step_label:
        var header = "Step " + String(step) + "/" + String(total)
        if label != "":
            header += ": " + label
        lines.append(pad + header)

    @always_inline
    fn make_cell(char: String, cell_width: Int) -> String:
        var mid = cell_width // 2
        var out = ""
        for i in range(cell_width):
            if i == mid:
                out += char
            else:
                out += " "
        return out

    @always_inline
    fn center_label(label: String, cell_width: Int) -> String:
        if len(label) >= cell_width:
            return label[:cell_width]
        var left = (cell_width - len(label)) // 2
        var right = cell_width - len(label) - left
        return (" " * left) + label + (" " * right)

    @always_inline
    fn make_bg_cell(char: String, cell_width: Int, color_width: Int) -> String:
        var pad_local = ""
        var cw = color_width
        if cw > cell_width:
            cw = cell_width
        for _ in range(cell_width - cw):
            pad_local += " "
        return center_label(char, cw) + pad_local

    var size = state.size()
    if size <= 0:
        return lines^
    var row_size = 1 << col_bits
    if row_size <= 0 or row_size > size or (size % row_size) != 0:
        row_size = size
        col_bits = 0
    var rows = size // row_size
    var cols = row_size
    var ref_amp: FloatType = 1.0
    var col_label_bits = col_bits
    if show_bin_labels and col_label_bits == 0:
        var tmp_cols = cols
        while tmp_cols > 1:
            tmp_cols //= 2
            col_label_bits += 1
        if col_label_bits == 0:
            col_label_bits = 0

    var row_idx_width = len(String(rows - 1))
    var show_row_labels = rows > 1
    var row_label_bits = col_bits
    if show_bin_labels and row_label_bits == 0:
        var tmp_rows = rows
        while tmp_rows > 1:
            tmp_rows //= 2
            row_label_bits += 1
    if show_bin_labels and row_label_bits > 0 and show_row_labels:
        row_idx_width = max(
            row_idx_width,
            row_label_bits + len(" -> ") + len(String(rows - 1)),
        )
    if not show_row_labels:
        row_idx_width = 0

    var cell_width = 3
    var h_line = " " * (row_idx_width + 1) + "+"
    for _ in range(cols):
        h_line += "-" * cell_width + "+"
    lines.append(pad + h_line)

    var row_indices = List[Int]()
    if origin_bottom:
        for i in range(rows - 1, -1, -1):
            row_indices.append(i)
    else:
        for i in range(rows):
            row_indices.append(i)

    for r in row_indices:
        var r_str = String(r) if show_row_labels else ""
        if show_bin_labels and row_label_bits > 0 and show_row_labels:
            var bin = ""
            for bit in reversed(range(row_label_bits)):
                bin += "1" if ((r >> bit) & 1) == 1 else "0"
            r_str = bin + " - " + String(r)
        var row_pad = " " * (row_idx_width - len(r_str))
        var line = row_pad + r_str + " |"
        for c in range(cols):
            var idx = r * cols + c
            var amp = state[idx]
            var fg_code = get_color_code(amp.re, amp.im)
            if use_bg:
                bg_code = get_bg_color_code(amp.re, amp.im)
            var mag = sqrt(amp.re * amp.re + amp.im * amp.im)
            if mag < 1e-6:
                mag = 0.0
            if mag < 1e-6:
                mag = 0.0
            var intensity = mag / ref_amp
            if use_log:
                intensity = log10(1.0 + mag) / log10(1.0 + ref_amp)
            if use_bg:
                intensity = quantize_intensity(intensity)
                var bg_code_i = get_bg_color_code_intensity(
                    amp.re, amp.im, intensity
                )
                if show_chars:
                    var char = get_alpha_char(intensity)
                    line += (
                        bg_code_i
                        + fg_code
                        + make_bg_cell(char, cell_width, 2)
                        + "\033[0m"
                        + "|"
                    )
                else:
                    line += (
                        bg_code_i
                        + make_bg_cell(" ", cell_width, 2)
                        + "\033[0m"
                        + "|"
                    )
            else:
                intensity = quantize_intensity(intensity)
                var char = get_alpha_char(intensity)
                line += (
                    fg_code
                    + make_cell(char, cell_width)
                    + "\033[0m"
                    + "|"
                )
        lines.append(pad + line)
        lines.append(pad + h_line)

    var show_col_labels = cols > 1
    var axis_pad = " " * (row_idx_width + 2)
    var axis_line = axis_pad
    var col_active = List[Bool](capacity=cols)
    for c in range(cols):
        var active = True
        for r in range(rows):
            var idx = r * cols + c
            var amp = state[idx]
            var mag = sqrt(amp.re * amp.re + amp.im * amp.im)
            if mag < 1e-6:
                mag = 0.0
            if mag < 1e-6:
                mag = 0.0
            if mag > 0.01:
                active = True
                break
        col_active.append(active)

    if show_col_labels:
        for c in range(cols):
            var s = String(c)
            var slot_width = cell_width + 1
            if col_active[c]:
                axis_line += center_label(s, cell_width) + " "
            else:
                axis_line += " " * slot_width
        lines.append(pad + axis_line)

    if show_bin_labels and col_label_bits > 0 and show_col_labels:
        var col_bits_width = col_label_bits
        var vert_line = " " * (row_idx_width + 2)
        for _ in range(cols):
            vert_line += center_label("|", cell_width) + " "
        lines.append(pad + vert_line)

        for bit in range(col_bits_width - 1, -1, -1):
            var digits_line = " " * (row_idx_width + 2)
            for c in range(cols):
                var digit = "1" if ((c >> bit) & 1) == 1 else "0"
                digits_line += center_label(digit, cell_width) + " "
            lines.append(pad + digits_line)
    return lines^


def get_alpha_char(intensity: FloatType) -> String:
    if intensity > 0.6:
        return "#"
    elif intensity > 0.3:
        return "+"
    elif intensity > 0.1:
        return "."
    elif intensity > 0.001:
        return ":"
    else:
        return " "

def quantize_intensity(intensity: FloatType) -> FloatType:
    if intensity <= 0.0:
        return 0.0
    elif intensity <= 0.15:
        return 0.1
    elif intensity <= 0.35:
        return 0.3
    elif intensity <= 0.6:
        return 0.6
    return 1.0


def print_state_grid_colored_cells(
    state: State,
    var col_bits: Int,
    use_log: Bool = False,
    origin_bottom: Bool = False,
    signed_y: Bool = False,
    use_bg: Bool = True,
    show_bin_labels: Bool = False,
    show_chars: Bool = False,
    show_all_labels: Bool = True,
    left_pad: Int = 0,
):
    """
    Prints the quantum state as a 2D grid of colored cells.
    Layout uses col_bits for the row_size (low bits are columns).
    """
    @always_inline
    fn make_cell(char: String, cell_width: Int) -> String:
        var mid = cell_width // 2
        var out = ""
        for i in range(cell_width):
            if i == mid:
                out += char
            else:
                out += " "
        return out

    @always_inline
    fn center_label(label: String, cell_width: Int) -> String:
        if len(label) >= cell_width:
            return label[:cell_width]
        var left = (cell_width - len(label)) // 2
        var right = cell_width - len(label) - left
        return (" " * left) + label + (" " * right)

    @always_inline
    fn make_bg_cell(char: String, cell_width: Int, color_width: Int) -> String:
        var pad = ""
        var cw = color_width
        if cw > cell_width:
            cw = cell_width
        for _ in range(cell_width - cw):
            pad += " "
        return center_label(char, cw) + pad

    var size = state.size()
    if size <= 0:
        return
    var row_size = 1 << col_bits
    if row_size <= 0 or row_size > size or (size % row_size) != 0:
        row_size = size
        col_bits = 0
    var rows = size // row_size
    var cols = row_size
    var ref_amp: FloatType = 1.0
    var col_label_bits = col_bits
    if show_bin_labels and col_label_bits == 0:
        var tmp_cols = cols
        while tmp_cols > 1:
            tmp_cols //= 2
            col_label_bits += 1
        if col_label_bits == 0:
            col_label_bits = 0

    var row_idx_width = len(String(rows - 1))
    var show_row_labels = rows > 1
    var row_label_bits = col_bits
    if show_bin_labels and row_label_bits == 0:
        var tmp_rows = rows
        while tmp_rows > 1:
            tmp_rows //= 2
            row_label_bits += 1
    if signed_y:
        row_idx_width = len(String(-(rows // 2)))
    if show_bin_labels and row_label_bits > 0 and show_row_labels:
        # var bin_width = len(String((1 << row_label_bits) - 1))
        row_idx_width = max(
            row_idx_width,
            row_label_bits + len(" -> ") + len(String(rows - 1)),
        )
    if not show_row_labels:
        row_idx_width = 0

    var cell_width = 3
    var h_line = " " * (row_idx_width + 1) + "+"
    for _ in range(cols):
        h_line += "-" * cell_width + "+"

    var left_pad_str = ""
    if left_pad > 0:
        left_pad_str = " " * left_pad
    print(left_pad_str + h_line)

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
        var r_str = String(r_val) if show_row_labels else ""
        if show_bin_labels and row_label_bits > 0 and show_row_labels:
            var bin = ""
            for bit in reversed(range(row_label_bits)):
                bin += "1" if ((r_val >> bit) & 1) == 1 else "0"
            r_str = bin + " - " + String(r_val)
        var row_pad = " " * (row_idx_width - len(r_str))
        var line = row_pad + r_str + " |"
        for c in range(cols):
            var idx = r * cols + c
            var amp = state[idx]
            var fg_code = get_color_code(amp.re, amp.im)
            # var bg_code = ""
            if use_bg:
                bg_code = get_bg_color_code(amp.re, amp.im)
            var mag = sqrt(amp.re * amp.re + amp.im * amp.im)
            var intensity = mag / ref_amp
            if use_log:
                intensity = log10(1.0 + mag) / log10(1.0 + ref_amp)
            if use_bg:
                intensity = quantize_intensity(intensity)
                var bg_code_i = get_bg_color_code_intensity(
                    amp.re, amp.im, intensity
                )
                if show_chars:
                    var char = get_alpha_char(intensity)
                    line += bg_code_i + fg_code + make_bg_cell(char, cell_width, 2) + "\033[0m" + "|"
                else:
                    line += bg_code_i + make_bg_cell(" ", cell_width, 2) + "\033[0m" + "|"
            else:
                intensity = quantize_intensity(intensity)
                var char = get_alpha_char(intensity)
                line += fg_code + make_cell(char, cell_width) + "\033[0m" + "|"
        print(left_pad_str + line)
        print(left_pad_str + h_line)

    var show_col_labels = cols > 1
    var axis_pad = " " * (row_idx_width + 2)
    var axis_line = axis_pad
    var col_active = List[Bool](capacity=cols)
    for c in range(cols):
        var active = show_all_labels
        for r in range(rows):
            var idx = r * cols + c
            var amp = state[idx]
            var mag = sqrt(amp.re * amp.re + amp.im * amp.im)
            if mag > 0.01:
                active = True
                break
        col_active.append(active)

    if show_col_labels:
        for c in range(cols):
            var s = String(c)
            var slot_width = cell_width + 1
            if col_active[c]:
                axis_line += center_label(s, cell_width) + " "
            else:
                axis_line += " " * slot_width
        print(left_pad_str + axis_line)

    if show_bin_labels and col_label_bits > 0 and show_col_labels:
        var col_bits_width = col_label_bits
        var vert_line = " " * (row_idx_width + 2)
        for _ in range(cols):
            vert_line += center_label("|", cell_width) + " "
        print(left_pad_str + vert_line)

        for bit in range(col_bits_width - 1, -1, -1):
            var digits_line = " " * (row_idx_width + 2)
            for c in range(cols):
                var digit = "1" if ((c >> bit) & 1) == 1 else "0"
                digits_line += center_label(digit, cell_width) + " "
            print(left_pad_str + digits_line)


def render_grid_frame(
    state: State,
    step: Int,
    total: Int,
    label: String,
    col_bits: Int,
    use_log: Bool,
    origin_bottom: Bool,
    use_bg: Bool,
    show_bin_labels: Bool,
    show_chars: Bool,
    show_step_label: Bool,
    redraw_in_place: Bool,
    left_pad: Int = 0,
):
    if redraw_in_place:
        print("\033c", end="")
        print("\033[H", end="")
    if show_step_label:
        var header = "Step " + String(step) + "/" + String(total)
        if label != "":
            header += ": " + label
        var left_pad_str = ""
        if left_pad > 0:
            left_pad_str = " " * left_pad
        print(left_pad_str + header)
    print_state_grid_colored_cells(
        state,
        col_bits,
        use_log=use_log,
        origin_bottom=origin_bottom,
        use_bg=use_bg,
        show_bin_labels=show_bin_labels,
        show_chars=show_chars,
        show_all_labels=True,
        left_pad=left_pad,
    )


def render_table_frame(
    state: State,
    step: Int,
    total: Int,
    label: String,
    short: Bool,
    use_color: Bool,
    show_step_label: Bool,
    redraw_in_place: Bool,
    left_pad: Int = 0,
    max_rows: Int = 0,
):
    if redraw_in_place:
        print("\033c", end="")
        print("\033[H", end="")
    if show_step_label:
        var header = "Step " + String(step) + "/" + String(total)
        if label != "":
            header += ": " + label
        print(header)
    print_state(
        state,
        short=short,
        use_color=use_color,
        left_pad=left_pad,
        max_rows=max_rows,
    )


fn animate_execution(
    circuit: QuantumCircuit,
    mut state: State,
    col_bits: Int,
    use_log: Bool = False,
    origin_bottom: Bool = False,
    show_bin_labels: Bool = False,
    use_bg: Bool = True,
    show_chars: Bool = False,
    show_step_label: Bool = True,
    delay_s: Float64 = 0.0,
    step_on_input: Bool = False,
    redraw_in_place: Bool = True,
    left_pad: Int = 0,
    ctx: ExecContext = ExecContext(),
) raises:
    """Execute and render the state after each transformation."""
    @always_inline
    fn wait_for_step_delta() raises -> Int:
        from python import Python

        var sys = Python.import_module("sys")
        var termios = Python.import_module("termios")
        var tty = Python.import_module("tty")
        var stdin = sys.stdin
        var fd = stdin.fileno()
        var old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            var ch = String(stdin.read(1))
            if ch == "\r" or ch == "\n" or ch == " ":
                return 1
            if ch == "\x7f":
                return -1
            if ch == "q" or ch == "Q" or ch == "\x03":
                return 2
            if ch == "\x1b":
                var ch2 = String(stdin.read(1))
                if ch2 == "[":
                    var ch3 = String(stdin.read(1))
                    if ch3 == "C":
                        return 1
                    if ch3 == "D":
                        return -1
                else:
                    return 2
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return 0
    @always_inline
    fn join_indices(indices: List[Int]) -> String:
        if len(indices) == 0:
            return ""
        var out = String(indices[0])
        for i in range(1, len(indices)):
            out += "," + String(indices[i])
        return out

    @always_inline
    fn format_transformation(tr: QuantumTransformation) -> String:
        if tr.isa[GateTransformation]():
            var gate_tr = tr[GateTransformation].copy()
            var label = String(gate_tr.gate_info.name)
            var ctrls = join_indices(gate_tr.controls)
            if ctrls != "":
                return label + " c=" + ctrls + " t=" + String(gate_tr.target)
            return label + " t=" + String(gate_tr.target)
        if tr.isa[FusedPairTransformation]():
            return "FUSED_PAIR"
        if tr.isa[SwapTransformation]():
            var swap_tr = tr[SwapTransformation].copy()
            return "SWAP " + String(swap_tr.a) + "," + String(swap_tr.b)
        if tr.isa[QubitReversalTransformation]():
            return "QREV"
        if tr.isa[UnitaryTransformation]():
            var unitary_tr = tr[UnitaryTransformation].copy()
            return unitary_tr.name + " t=" + String(unitary_tr.target)
        if tr.isa[ControlledUnitaryTransformation]():
            var cu_tr = tr[ControlledUnitaryTransformation].copy()
            return cu_tr.name + " c=" + String(cu_tr.control) + " t=" + String(cu_tr.target)
        if tr.isa[MeasurementTransformation[State]]():
            return "MEASURE"
        if tr.isa[ClassicalTransformation[State]]():
            var cl_tr = tr[ClassicalTransformation[State]].copy()
            return cl_tr.name
        return "TRANSFORM"

    var total = len(circuit.transformations)
    if step_on_input:
        var init_state = state.copy()
        var step = 0
        render_grid_frame(
            state,
            step,
            total,
            "init",
            col_bits,
            use_log,
            origin_bottom,
            use_bg,
            show_bin_labels,
            show_chars,
            show_step_label,
            redraw_in_place,
        )
        while True:
            var delta = wait_for_step_delta()
            if delta == 0:
                continue
            if delta == 2:
                break
            var next = step + delta
            if next < 0 or next > total:
                continue
            if delta > 0:
                var tr = circuit.transformations[step]
                var step_circuit = QuantumCircuit(circuit.num_qubits)
                step_circuit.transformations.append(tr.copy())
                execute(state, step_circuit, ctx)
            else:
                # Step back by applying inverses of the last transformations.
                var times = step - next
                if times < 0:
                    times = 0
                for _ in range(times):
                    if step == 0:
                        break
                    # Try to invert the single last transformation and execute it.
                    var last_tr = circuit.transformations[step - 1]
                    if last_tr.isa[ClassicalTransformation[State]]():
                        # Force replay for non-invertible classical transforms.
                        state = init_state.copy()
                        if next > 0:
                            var sub_circuit = QuantumCircuit(circuit.num_qubits)
                            for i in range(next):
                                sub_circuit.transformations.append(
                                    circuit.transformations[i].copy()
                                )
                            execute(state, sub_circuit, ctx)
                        break
                    var single = QuantumCircuit(circuit.num_qubits)
                    single.transformations.append(last_tr.copy())
                    try:
                        var inv_single = single.inverse()
                        execute(state, inv_single, ctx)
                        step -= 1
                        continue
                    except:
                        # If inversion fails (e.g. measurement), fall back to full replay.
                        state = init_state.copy()
                        if next > 0:
                            var sub_circuit = QuantumCircuit(circuit.num_qubits)
                            for i in range(next):
                                sub_circuit.transformations.append(
                                    circuit.transformations[i].copy()
                                )
                            execute(state, sub_circuit, ctx)
                        break
            step = next
            var label = "init"
            if step > 0:
                label = format_transformation(
                    circuit.transformations[step - 1]
                )
            render_grid_frame(
                state,
                step,
                total,
                label,
                col_bits,
                use_log,
                origin_bottom,
                use_bg,
                show_bin_labels,
                show_chars,
                show_step_label,
                redraw_in_place,
                left_pad,
            )
    else:
        var step = 0
        render_grid_frame(
            state,
            step,
            total,
            "init",
            col_bits,
            use_log,
            origin_bottom,
            use_bg,
            show_bin_labels,
            show_chars,
            show_step_label,
            redraw_in_place,
            left_pad,
        )
        if delay_s > 0.0:
            sleep(delay_s)
        for i in range(total):
            var tr = circuit.transformations[i]
            var step_circuit = QuantumCircuit(circuit.num_qubits)
            step_circuit.transformations.append(tr.copy())
            execute(state, step_circuit, ctx)
            step = i + 1
            render_grid_frame(
                state,
                step,
                total,
                format_transformation(tr),
                col_bits,
                use_log,
                origin_bottom,
                use_bg,
                show_bin_labels,
                show_chars,
                show_step_label,
                redraw_in_place,
                left_pad,
            )
            if delay_s > 0.0:
                sleep(delay_s)


fn animate_execution_table(
    circuit: QuantumCircuit,
    mut state: State,
    short: Bool = True,
    use_color: Bool = True,
    show_step_label: Bool = True,
    delay_s: Float64 = 0.0,
    step_on_input: Bool = False,
    redraw_in_place: Bool = True,
    left_pad: Int = 0,
    max_rows: Int = 0,
    ctx: ExecContext = ExecContext(),
) raises:
    """Execute and render the state table after each transformation."""
    @always_inline
    fn wait_for_step_delta() raises -> Int:
        from python import Python

        var sys = Python.import_module("sys")
        var termios = Python.import_module("termios")
        var tty = Python.import_module("tty")
        var stdin = sys.stdin
        var fd = stdin.fileno()
        var old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            var ch = String(stdin.read(1))
            if ch == "\r" or ch == "\n" or ch == " ":
                return 1
            if ch == "\x7f":
                return -1
            if ch == "q" or ch == "Q" or ch == "\x03":
                return 2
            if ch == "\x1b":
                var ch2 = String(stdin.read(1))
                if ch2 == "[":
                    var ch3 = String(stdin.read(1))
                    if ch3 == "C":
                        return 1
                    if ch3 == "D":
                        return -1
                else:
                    return 2
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return 0

    var total = len(circuit.transformations)

    if step_on_input:
        var init_state = state.copy()
        var step = 0
        render_table_frame(
            state,
            step,
            total,
            "init",
            short,
            use_color,
            show_step_label,
            redraw_in_place,
            left_pad,
            max_rows,
        )
        while True:
            var delta = wait_for_step_delta()
            if delta == 0:
                continue
            if delta == 2:
                break
            var next = step + delta
            if next < 0 or next > total:
                continue
            if delta > 0:
                var tr = circuit.transformations[step]
                var step_circuit = QuantumCircuit(circuit.num_qubits)
                step_circuit.transformations.append(tr.copy())
                execute(state, step_circuit, ctx)
            else:
                # Step back by applying inverses of the last transformations.
                var times = step - next
                if times < 0:
                    times = 0
                for _ in range(times):
                    if step == 0:
                        break
                    var last_tr = circuit.transformations[step - 1]
                    if last_tr.isa[ClassicalTransformation[State]]():
                        state = init_state.copy()
                        if next > 0:
                            var sub_circuit = QuantumCircuit(circuit.num_qubits)
                            for i in range(next):
                                sub_circuit.transformations.append(
                                    circuit.transformations[i].copy()
                                )
                            execute(state, sub_circuit, ctx)
                        break
                    var single = QuantumCircuit(circuit.num_qubits)
                    single.transformations.append(last_tr.copy())
                    try:
                        var inv_single = single.inverse()
                        execute(state, inv_single, ctx)
                        step -= 1
                        continue
                    except:
                        # If inversion fails (e.g. measurement), fall back to full replay.
                        state = init_state.copy()
                        if next > 0:
                            var sub_circuit = QuantumCircuit(circuit.num_qubits)
                            for i in range(next):
                                sub_circuit.transformations.append(
                                    circuit.transformations[i].copy()
                                )
                            execute(state, sub_circuit, ctx)
                        break
            step = next
            render_table_frame(
                state,
                step,
                total,
                "",
                short,
                use_color,
                show_step_label,
                redraw_in_place,
                left_pad,
                max_rows,
            )
    else:
        var step = 0
        render_table_frame(
            state,
            step,
            total,
            "init",
            short,
            use_color,
            show_step_label,
            redraw_in_place,
            left_pad,
            max_rows,
        )
        if delay_s > 0.0:
            sleep(delay_s)
        for i in range(total):
            var tr = circuit.transformations[i]
            var step_circuit = QuantumCircuit(circuit.num_qubits)
            step_circuit.transformations.append(tr.copy())
            execute(state, step_circuit, ctx)
            step = i + 1
            render_table_frame(
                state,
                step,
                total,
                "",
                short,
                use_color,
                show_step_label,
                redraw_in_place,
                left_pad,
                max_rows,
            )
            if delay_s > 0.0:
                sleep(delay_s)


fn animate_execution_table_grid(
    circuit: QuantumCircuit,
    mut state: State,
    col_bits: Int,
    use_log: Bool = False,
    origin_bottom: Bool = False,
    show_bin_labels: Bool = False,
    use_bg: Bool = True,
    show_chars: Bool = False,
    short: Bool = True,
    use_color: Bool = True,
    show_step_label: Bool = True,
    delay_s: Float64 = 0.0,
    step_on_input: Bool = False,
    redraw_in_place: Bool = True,
    table_left_pad: Int = 0,
    grid_left_pad: Int = 20,
    gap: Int = 4,
    ctx: ExecContext = ExecContext(),
) raises:
    """Execute and render table + grid side-by-side."""
    @always_inline
    fn wait_for_step_delta() raises -> Int:
        from python import Python

        var sys = Python.import_module("sys")
        var termios = Python.import_module("termios")
        var tty = Python.import_module("tty")
        var stdin = sys.stdin
        var fd = stdin.fileno()
        var old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            var ch = String(stdin.read(1))
            if ch == "\r" or ch == "\n" or ch == " ":
                return 1
            if ch == "\x7f":
                return -1
            if ch == "q" or ch == "Q" or ch == "\x03":
                return 2
            if ch == "\x1b":
                var ch2 = String(stdin.read(1))
                if ch2 == "[":
                    var ch3 = String(stdin.read(1))
                    if ch3 == "C":
                        return 1
                    if ch3 == "D":
                        return -1
                else:
                    return 2
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return 0

    @always_inline
    fn render_pair(step: Int, total: Int, label: String) raises:
        if redraw_in_place:
            print("\033c", end="")
            print("\033[H", end="")
        var table_lines = render_table_frame_lines(
            state,
            step,
            total,
            label,
            short,
            use_color,
            show_step_label,
            table_left_pad,
            row_separators=False,
        )
        var grid_lines = render_grid_frame_lines(
            state,
            step,
            total,
            label,
            col_bits,
            use_log,
            origin_bottom,
            use_bg,
            show_bin_labels,
            show_chars,
            False,
            grid_left_pad,
        )
        if show_step_label:
            var pad = ""
            if grid_left_pad > 0:
                pad = " " * grid_left_pad
            var aligned = List[String](capacity=len(grid_lines) + 1)
            aligned.append(pad)
            for i in range(len(grid_lines)):
                aligned.append(grid_lines[i])
            grid_lines = aligned^
        var merged = _merge_frame_lines(table_lines, grid_lines, gap)
        for i in range(len(merged)):
            print(merged[i])

    var total = len(circuit.transformations)

    if step_on_input:
        var init_state = state.copy()
        var step = 0
        render_pair(step, total, "init")
        while True:
            var delta = wait_for_step_delta()
            if delta == 0:
                continue
            if delta == 2:
                break
            var next = step + delta
            if next < 0 or next > total:
                continue
            if delta > 0:
                var tr = circuit.transformations[step]
                var step_circuit = QuantumCircuit(circuit.num_qubits)
                step_circuit.transformations.append(tr.copy())
                execute(state, step_circuit, ctx)
            else:
                var times = step - next
                if times < 0:
                    times = 0
                for _ in range(times):
                    if step == 0:
                        break
                    var last_tr = circuit.transformations[step - 1]
                    if last_tr.isa[ClassicalTransformation[State]]():
                        state = init_state.copy()
                        if next > 0:
                            var sub_circuit = QuantumCircuit(circuit.num_qubits)
                            for i in range(next):
                                sub_circuit.transformations.append(
                                    circuit.transformations[i].copy()
                                )
                            execute(state, sub_circuit, ctx)
                        break
                    var single = QuantumCircuit(circuit.num_qubits)
                    single.transformations.append(last_tr.copy())
                    try:
                        var inv_single = single.inverse()
                        execute(state, inv_single, ctx)
                        step -= 1
                        continue
                    except:
                        state = init_state.copy()
                        if next > 0:
                            var sub_circuit = QuantumCircuit(circuit.num_qubits)
                            for i in range(next):
                                sub_circuit.transformations.append(
                                    circuit.transformations[i].copy()
                                )
                            execute(state, sub_circuit, ctx)
                        break
            step = next
            render_pair(step, total, "")
    else:
        var step = 0
        render_pair(step, total, "init")
        if delay_s > 0.0:
            sleep(delay_s)
        for i in range(total):
            var tr = circuit.transformations[i]
            var step_circuit = QuantumCircuit(circuit.num_qubits)
            step_circuit.transformations.append(tr.copy())
            execute(state, step_circuit, ctx)
            step = i + 1
            render_pair(step, total, "")
            if delay_s > 0.0:
                sleep(delay_s)


fn animate_execution_grid_pair(
    circuit_left: QuantumCircuit,
    mut state_left: State,
    circuit_right: QuantumCircuit,
    mut state_right: State,
    col_bits: Int,
    use_log: Bool = False,
    origin_bottom: Bool = False,
    show_bin_labels: Bool = False,
    use_bg: Bool = True,
    show_chars: Bool = False,
    show_step_label: Bool = True,
    delay_s: Float64 = 0.0,
    step_on_input: Bool = False,
    redraw_in_place: Bool = True,
    left_pad_left: Int = 0,
    left_pad_right: Int = 0,
    gap: Int = 4,
    ctx_left: ExecContext = ExecContext(),
    ctx_right: ExecContext = ExecContext(),
) raises:
    """Execute and render two grids side-by-side after each transformation."""
    @always_inline
    fn wait_for_step_delta() raises -> Int:
        from python import Python

        var sys = Python.import_module("sys")
        var termios = Python.import_module("termios")
        var tty = Python.import_module("tty")
        var stdin = sys.stdin
        var fd = stdin.fileno()
        var old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            var ch = String(stdin.read(1))
            if ch == "\r" or ch == "\n" or ch == " ":
                return 1
            if ch == "\x7f":
                return -1
            if ch == "q" or ch == "Q" or ch == "\x03":
                return 2
            if ch == "\x1b":
                var ch2 = String(stdin.read(1))
                if ch2 == "[":
                    var ch3 = String(stdin.read(1))
                    if ch3 == "C":
                        return 1
                    if ch3 == "D":
                        return -1
                else:
                    return 2
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return 0

    @always_inline
    fn render_pair(
        state_left_for_render: State,
        state_right_for_render: State,
        step: Int,
        total: Int,
        label: String,
    ) raises:
        if redraw_in_place:
            print("\033c", end="")
            print("\033[H", end="")
        var left_lines = render_grid_frame_lines(
            state_left_for_render,
            step,
            total,
            label,
            col_bits,
            use_log,
            origin_bottom,
            use_bg,
            show_bin_labels,
            show_chars,
            show_step_label,
            left_pad_left,
        )
        var right_lines = render_grid_frame_lines(
            state_right_for_render,
            step,
            total,
            label,
            col_bits,
            use_log,
            origin_bottom,
            use_bg,
            show_bin_labels,
            show_chars,
            show_step_label,
            left_pad_right,
        )
        var merged = _merge_frame_lines(left_lines, right_lines, gap)
        for i in range(len(merged)):
            print(merged[i])

    var total_left = len(circuit_left.transformations)
    var total_right = len(circuit_right.transformations)
    var total = max(total_left, total_right)

    if step_on_input:
        var init_left = state_left.copy()
        var init_right = state_right.copy()
        var step = 0
        render_pair(state_left, state_right, step, total, "init")
        while True:
            var delta = wait_for_step_delta()
            if delta == 0:
                continue
            if delta == 2:
                break
            var next = step + delta
            if next < 0 or next > total:
                continue
            if delta > 0:
                if step < total_left:
                    var tr_left = circuit_left.transformations[step]
                    var step_circuit_left = QuantumCircuit(circuit_left.num_qubits)
                    step_circuit_left.transformations.append(tr_left.copy())
                    execute(state_left, step_circuit_left, ctx_left)
                if step < total_right:
                    var tr_right = circuit_right.transformations[step]
                    var step_circuit_right = QuantumCircuit(
                        circuit_right.num_qubits
                    )
                    step_circuit_right.transformations.append(tr_right.copy())
                    execute(state_right, step_circuit_right, ctx_right)
            else:
                state_left = init_left.copy()
                state_right = init_right.copy()
                if next > 0:
                    for i in range(next):
                        if i < total_left:
                            var tr_left = circuit_left.transformations[i]
                            var step_circuit_left = QuantumCircuit(
                                circuit_left.num_qubits
                            )
                            step_circuit_left.transformations.append(
                                tr_left.copy()
                            )
                            execute(state_left, step_circuit_left, ctx_left)
                        if i < total_right:
                            var tr_right = circuit_right.transformations[i]
                            var step_circuit_right = QuantumCircuit(
                                circuit_right.num_qubits
                            )
                            step_circuit_right.transformations.append(
                                tr_right.copy()
                            )
                            execute(state_right, step_circuit_right, ctx_right)
            step = next
            render_pair(state_left, state_right, step, total, "")
    else:
        var step = 0
        render_pair(state_left, state_right, step, total, "init")
        if delay_s > 0.0:
            sleep(delay_s)
        for i in range(total):
            if i < total_left:
                var tr_left = circuit_left.transformations[i]
                var step_circuit_left = QuantumCircuit(circuit_left.num_qubits)
                step_circuit_left.transformations.append(tr_left.copy())
                execute(state_left, step_circuit_left, ctx_left)
            if i < total_right:
                var tr_right = circuit_right.transformations[i]
                var step_circuit_right = QuantumCircuit(
                    circuit_right.num_qubits
                )
                step_circuit_right.transformations.append(tr_right.copy())
                execute(state_right, step_circuit_right, ctx_right)
            step = i + 1
            render_pair(state_left, state_right, step, total, "")
            if delay_s > 0.0:
                sleep(delay_s)
