from math import log2, log10, sqrt, atan2, floor
from butterfly.core.types import *
from butterfly.core.state import QuantumState, ArrayState, GridState
from collections import InlineArray

alias c6_data = InlineArray[Int, 180](
    237,
    94,
    147,
    239,
    93,
    136,
    240,
    94,
    125,
    240,
    96,
    115,
    239,
    98,
    105,
    237,
    101,
    95,
    234,
    105,
    85,
    229,
    109,
    76,
    225,
    113,
    67,
    219,
    117,
    59,
    212,
    122,
    51,
    205,
    126,
    44,
    198,
    130,
    37,
    189,
    134,
    31,
    181,
    138,
    27,
    171,
    142,
    24,
    162,
    145,
    24,
    152,
    149,
    26,
    141,
    152,
    31,
    130,
    154,
    37,
    119,
    157,
    44,
    106,
    159,
    52,
    93,
    161,
    60,
    78,
    163,
    69,
    59,
    164,
    79,
    31,
    166,
    89,
    0,
    167,
    99,
    0,
    168,
    110,
    0,
    168,
    121,
    0,
    169,
    132,
    0,
    169,
    143,
    0,
    170,
    154,
    0,
    170,
    165,
    0,
    170,
    176,
    0,
    169,
    186,
    0,
    169,
    196,
    0,
    168,
    205,
    0,
    167,
    214,
    0,
    166,
    222,
    0,
    165,
    229,
    0,
    163,
    236,
    0,
    161,
    241,
    0,
    159,
    245,
    0,
    157,
    248,
    0,
    154,
    250,
    0,
    151,
    251,
    31,
    147,
    250,
    77,
    143,
    248,
    104,
    139,
    246,
    125,
    135,
    242,
    144,
    131,
    237,
    160,
    126,
    231,
    174,
    121,
    224,
    187,
    116,
    216,
    198,
    112,
    207,
    208,
    107,
    198,
    217,
    103,
    189,
    224,
    100,
    179,
    229,
    97,
    168,
    234,
    95,
    158,
)


def get_color_code(re: FloatType, im: FloatType) -> String:
    var angle = atan2(im, re)
    var deg = angle * 180.0 / pi
    if deg < 0:
        deg += 360.0

    # Map 0-360 to 0-59
    var idx = Int(deg / 6.0) % 60

    var r = c6_data[3 * idx]
    var g = c6_data[3 * idx + 1]
    var b = c6_data[3 * idx + 2]

    # TrueColor ANSI: \033[38;2;R;G;Bm
    return "\033[38;2;" + String(r) + ";" + String(g) + ";" + String(b) + "m"


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
        ).rjust(decimals + 2, " ")
        var im_str = (
            (" + " if round_state[k].im >= 0 else " - ")
            + "i"
            + String(abs(round_state[k].im)).ljust(decimals + 2, " ")
        )
        row.append(re_str + im_str)

        var mag = sqrt(s[k].re * s[k].re + s[k].im * s[k].im)
        row.append(String(round(mag, decimals)).rjust(decimals + 2, " "))

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

        var bar_char = "█"
        var mag_len = Int(floor(16 * mag))
        var mag_bar = bar_char * mag_len
        row.append(mag_bar + (" " * (16 - mag_len)))

        var prob = s[k].re * s[k].re + s[k].im * s[k].im
        row.append(String(round(prob, decimals)).rjust(decimals + 2, " "))

        var prob_len = Int(floor(16 * prob))
        var prob_bar = bar_char * prob_len
        row.append(prob_bar + (" " * (16 - prob_len)))

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
    real_color_code = get_color_code(1, 0)
    for i in range(len(table)):
        print("\n")

        var color_code = "\033[0m"
        if use_color:
            color_code = get_color_code(sub_state[i].re, sub_state[i].im)

        for j in range(len(table[i])):
            var cell = table[i][j]
            if use_color:
                if j == 5:  # MagBar
                    cell = color_code + cell + "\033[0m"
                if j == 7:  # ProbBar
                    cell = real_color_code + cell + "\033[0m"
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
