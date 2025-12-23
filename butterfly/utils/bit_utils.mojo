"""Helper function for bit manipulation in fused gates."""


@always_inline
fn insert_zero_bit(x: Int, pos: Int) -> Int:
    """
    Insert a zero bit at position `pos` in the binary representation of `x`.

    Example: insert_zero_bit(0b101, 1) = 0b1001
    The bit at position 1 becomes 0, and all higher bits shift left.
    """
    var mask = (1 << pos) - 1
    var low = x & mask
    var high = (x & ~mask) << 1
    return high | low
