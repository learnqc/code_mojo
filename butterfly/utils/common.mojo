from butterfly.core.state import *

def is_close(a: Amplitude, b: Amplitude) -> Bool:
    try:
        assert_almost_equal[Type, 1](a.re, b.re)
        assert_almost_equal[Type, 1](a.im, b.im)
        return True
    except e:
        return False