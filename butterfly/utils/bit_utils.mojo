"""Helper functions for bit manipulation used in fused kernels."""


@always_inline
fn insert_zero_bit(x: Int, pos: Int) -> Int:
    """Insert a zero bit at position `pos` in the binary representation of `x`."""
    var mask = (1 << pos) - 1
    var low = x & mask
    var high = (x & ~mask) << 1
    return high | low
