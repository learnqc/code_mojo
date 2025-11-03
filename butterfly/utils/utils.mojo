from math import log2, log10

from butterfly import *
from butterfly.core import *

def is_close(a: Amplitude, b: Amplitude) -> Bool:
    try:
        assert_almost_equal[Type, 1](a.re, b.re)
        assert_almost_equal[Type, 1](a.im, b.im)
        return True
    except e:
        return False

def to_table(s: State, decimals: UInt=3) -> List[List[String]]:
    n = Int(log2(Float32(len(s))))

    m =  Int(log10(Float32(len(s))))

    round_state = [Amplitude(round(s[k].re, decimals), round(s[k].im, decimals)) for k in range(len(s))]
    table: List[List[String]] =
        [['￤' + String(k).rjust(max(5, m+1), ' '), 
        bin(k, prefix='').rjust(n, '0'),
        (' ' if round_state[k].re >= 0 else '-') + String(abs(round_state[k].re)).rjust(decimals + 2, ' ') +
        (' + ' if round_state[k].im >= 0 else ' - ') + 'i' + String(abs(round_state[k].im)).ljust(decimals + 2, ' '),
        String(round(sqrt(s[k].re*s[k].re + s[k].im*s[k].im), decimals)).rjust(decimals + 2, ' ') ,
        ((' ' if atan2(s[k].im, s[k].re) >= 0 else '-') + String(0 if s[k].re == 0 and s[k].im == 0 else abs(round(atan2(s[k].im, s[k].re) / (2 * pi) * 360, 2))) + '°').rjust(decimals + 6, ' '),
        ('#' * Int(floor(16*sqrt(s[k].re*s[k].re + s[k].im*s[k].im)))).ljust(16, ' '),
        String(round(s[k].re*s[k].re + s[k].im*s[k].im, decimals)).rjust(decimals + 2, ' '),
        ('#' * Int(floor(16*(s[k].re*s[k].re + s[k].im*s[k].im)))).ljust(16, ' ')] for k in range(len(s))]

    
    headers: List[List[String]] = [
        [' ' + '-' * (len(table[0][0])-1),
        '-' * len(table[0][1]),
        '-' * len(table[0][2]),
        '-' * len(table[0][3]),
        '-' * (len(table[0][4])-1),
        '-' * len(table[0][5]),
        '-' * len(table[0][6]),
        '-' * (len(table[0][7])-2)],
        ['￤' + 'Out'.rjust(len(table[0][0])-3, ' '), 
        'Bin'.rjust(len(table[0][1]), ' '), 
        'Ampl'.rjust(len(table[0][2]), ' '), 
        'Mag'.rjust(len(table[0][3]), ' '), 
        'Dir'.rjust(len(table[0][4])-1, ' '), 
        'Ampl Bar'.rjust(len(table[0][5]), ' '), 
        'Prob'.rjust(len(table[0][6]), ' '),
        'Prob Bar'.rjust(len(table[0][7]), ' ')],
        [' ' + '-' * (len(table[0][0])-1),
        '-' * len(table[0][1]),
        '-' * len(table[0][2]),
        '-' * len(table[0][3]),
        '-' * (len(table[0][4])-1),
        '-' * len(table[0][5]),
        '-' * len(table[0][6]),
        '-' * (len(table[0][7])-1)]
        ]

    bottom: List[List[String]] = [
        [' ' + '-' * (len(table[0][0])-1),
        '-' * len(table[0][1]),
        '-' * len(table[0][2]),
        '-' * len(table[0][3]),
        '-' * (len(table[0][4])-1),
        '-' * len(table[0][5]),
        '-' * len(table[0][6]),
        '-' * (len(table[0][7])-2)]
        ]

    ret = headers + (table + bottom^)
    return ret^

def print_state(state: State, short: Bool=True):
    rows = 16 if short else max(16, len(state))
    table = to_table(state[:rows])
    for i in range(len(table)):
        print('\n')
        for j in range(len(table[i])):
            print(table[i][j], end='---' if i == 0 or i == 2 or i == len(table)-1 else ' ￤')

    print('\n')