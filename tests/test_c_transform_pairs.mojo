from testing import assert_true, assert_equal


fn is_bit_set(m: Int, k: Int) -> Bool:
    return (m & (1 << k)) != 0


fn get_pairs_original(n: Int, control: Int, target: Int) -> List[Int]:
    var pairs = List[Int]()
    var stride = 1 << target
    var size = 1 << n
    var r = 0
    for j in range(size // 2):
        var idx = 2 * j - r
        if is_bit_set(Int(idx), control):
            pairs.append(idx)
            pairs.append(idx + stride)
        r += 1
        if r == stride:
            r = 0
    return pairs^


fn get_pairs_interval(n: Int, control: Int, target: Int) -> List[Int]:
    var pairs = List[Int]()
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = 1 << n

    if target < control:
        for k in range(size // (2 * c_stride)):
            var block_start = k * 2 * c_stride + c_stride
            for j in range(c_stride // (2 * t_stride)):
                var sub_start = block_start + j * 2 * t_stride
                for idx in range(sub_start, sub_start + t_stride):
                    pairs.append(idx)
                    pairs.append(idx + t_stride)
    else:
        for k in range(size // (2 * t_stride)):
            var base = k * 2 * t_stride
            var num_periods = t_stride // (2 * c_stride)
            for p in range(num_periods):
                var p_start = base + p * 2 * c_stride + c_stride
                for idx in range(p_start, p_start + c_stride):
                    pairs.append(idx)
                    pairs.append(idx + t_stride)
    return pairs^


fn check_pairs(n: Int, control: Int, target: Int) raises:
    var pairs1 = get_pairs_original(n, control, target)
    var pairs2 = get_pairs_interval(n, control, target)

    if len(pairs1) != len(pairs2):
        print("Length mismatch for n=", n, " c=", control, " t=", target)
        print("Original:", len(pairs1) // 2)
        print("Interval:", len(pairs2) // 2)
        assert_true(False)

    # Check if all pairs in pairs1 are in pairs2
    for i in range(0, len(pairs1), 2):
        var p1_k0 = pairs1[i]
        var p1_k1 = pairs1[i + 1]
        var found = False
        for j in range(0, len(pairs2), 2):
            if p1_k0 == pairs2[j] and p1_k1 == pairs2[j + 1]:
                found = True
                break
        if not found:
            print(
                "Pair (",
                p1_k0,
                ", ",
                p1_k1,
                ") missing in interval implementation",
            )
            assert_true(False)

    # Check if all pairs in pairs2 are in pairs1
    for i in range(0, len(pairs2), 2):
        var p2_k0 = pairs2[i]
        var p2_k1 = pairs2[i + 1]
        var found = False
        for j in range(0, len(pairs1), 2):
            if p2_k0 == pairs1[j] and p2_k1 == pairs1[j + 1]:
                found = True
                break
        if not found:
            print(
                "Pair (",
                p2_k0,
                ", ",
                p2_k1,
                ") extra in interval implementation",
            )
            assert_true(False)


fn main() raises:
    print("Testing pair generation logic...")
    var n = 5

    # Test all pairs of (control, target)
    for c in range(n):
        for t in range(n):
            if c == t:
                continue
            print("Checking n=", n, " control=", c, " target=", t)
            check_pairs(n, c, t)

    print("All pair tests passed!")
