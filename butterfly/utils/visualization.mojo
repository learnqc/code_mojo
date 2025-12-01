from math import log2, log10, sqrt, atan2, floor
from butterfly.core.types import *
from butterfly.core.state import QuantumState, ArrayState, GridState


def to_table(
    s: QuantumState, prefix: Tuple[Int, Int] = (0, 0), decimals: Int = 3
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

        row.append(("#" * Int(floor(16 * mag))).ljust(16, " "))

        var prob = s[k].re * s[k].re + s[k].im * s[k].im
        row.append(String(round(prob, decimals)).rjust(decimals + 2, " "))
        row.append(("#" * Int(floor(16 * prob))).ljust(16, " "))

        table.append(row^)

    return table^


def print_state(
    state: QuantumState, prefix: Tuple[Int, Int] = (0, 0), short: Bool = True
):
    rows = 16 if short else max(16, state.size())

    var sub_re = List[FloatType]()
    var sub_im = List[FloatType]()
    limit = min(state.size(), rows)
    for i in range(limit):
        sub_re.append(state[i].re)
        sub_im.append(state[i].im)
    var sub_state = QuantumState(sub_re^, sub_im^)

    table = to_table(sub_state, prefix)
    for i in range(len(table)):
        print("\n")
        for j in range(len(table[i])):
            print(table[i][j], end=" ￤")
    print("\n")


def print_state_a(
    state: ArrayState, prefix: Tuple[Int, Int] = (0, 0), short: Bool = True
):
    var re = List[FloatType]()
    var im = List[FloatType]()
    for i in range(len(state)):
        re.append(state[i].re)
        im.append(state[i].im)
    var qs = QuantumState(re^, im^)
    print_state(qs, prefix, short)


def print_grid_state(state: GridState, short: Bool = True):
    col_bits = Int(log10(Float32(len(state[0]))))
    for r in range(len(state)):
        var row_re = List[FloatType]()
        var row_im = List[FloatType]()
        for c in range(len(state[r])):
            row_re.append(state[r][c].re)
            row_im.append(state[r][c].im)
        var qs = QuantumState(row_re^, row_im^)
        print_state(qs, (Int(r), Int(col_bits)), short)
