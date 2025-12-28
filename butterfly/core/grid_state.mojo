"""
GridQuantumState: Unified quantum state representation with configurable rows.

A grid state represents an n-qubit system as num_rows × row_size grid:
- num_rows = 2^row_bits
- row_size = 2^col_bits  
- n = row_bits + col_bits

Operations on qubits 0 to col_bits-1 can be parallelized across rows.
"""

from butterfly.core.types import FloatType, Type
from butterfly.core.gates import Gate
from butterfly.core.circuit import QuantumCircuit, BitReversalTransformation
from buffer import NDBuffer
from bit.bit import bit_reverse


struct GridQuantumState:
    """Quantum state organized as a grid for efficient parallel execution.

    The state is stored as a single flat List, but logically organized as rows.
    NDBuffer views provide zero-copy access to individual rows.
    """

    var n: Int
    var row_bits: Int
    var col_bits: Int
    var num_rows: Int
    var row_size: Int
    var re: List[FloatType]
    var im: List[FloatType]

    fn __init__(out self, n: Int, row_bits: Int):
        """Initialize grid state.

        Args:
            n: Total number of qubits.
            row_bits: Number of bits for row indexing (num_rows = 2^row_bits).
        """
        self.n = n
        self.row_bits = row_bits
        self.col_bits = n - row_bits
        self.num_rows = 1 << row_bits
        self.row_size = 1 << self.col_bits

        var total_size = 1 << n
        self.re = List[FloatType](capacity=total_size)
        self.im = List[FloatType](capacity=total_size)
        self.re.resize(total_size, 0.0)
        self.im.resize(total_size, 0.0)

        # Initialize to |0...0⟩
        self.re[0] = 1.0

    fn size(self) -> Int:
        """Total number of amplitudes."""
        return 1 << self.n

    fn bit_reverse(mut self):
        """Bit-reverse the quantum state."""
        if self.row_bits == self.col_bits:
            # Optimized Grid-Based Bit Reversal (for square grids)
            # 1. Bit-reverse inside each row
            from algorithm import parallelize

            @parameter
            fn reverse_rows(row: Int):
                var offset = row * self.row_size
                var r_ptr = self.re.unsafe_ptr() + offset
                var i_ptr = self.im.unsafe_ptr() + offset

                for i in range(self.row_size):
                    var rev_i = Int(
                        bit_reverse(SIMD[DType.uint64, 1](i))[0]
                        >> (64 - self.col_bits)
                    )
                    if i < rev_i:
                        var tmp_re = r_ptr[i]
                        var tmp_im = i_ptr[i]
                        r_ptr[i] = r_ptr[rev_i]
                        i_ptr[i] = i_ptr[rev_i]
                        r_ptr[rev_i] = tmp_re
                        i_ptr[rev_i] = tmp_im

            parallelize[reverse_rows](self.num_rows)

            # 2. Bit-reverse the row order
            var row_visited = List[Bool](capacity=self.num_rows)
            for _ in range(self.num_rows):
                row_visited.append(False)

            for r in range(self.num_rows):
                if row_visited[r]:
                    continue
                var rev_r = Int(
                    bit_reverse(SIMD[DType.uint64, 1](r))[0]
                    >> (64 - self.row_bits)
                )
                if r != rev_r:
                    # Swap row blocks
                    var r_ptr = self.re.unsafe_ptr()
                    var i_ptr = self.im.unsafe_ptr()
                    for i in range(self.row_size):
                        var idx0 = r * self.row_size + i
                        var idx1 = rev_r * self.row_size + i

                        var tmp_re = r_ptr[idx0]
                        var tmp_im = i_ptr[idx0]
                        r_ptr[idx0] = r_ptr[idx1]
                        i_ptr[idx0] = i_ptr[idx1]
                        r_ptr[idx1] = tmp_re
                        i_ptr[idx1] = tmp_im

                row_visited[r] = True
                row_visited[rev_r] = True

            # 3. Transpose
            # Full bit reversal transpose: (rev_r, rev_c) -> (rev_c, rev_r)
            var r_ptr = self.re.unsafe_ptr()
            var i_ptr = self.im.unsafe_ptr()
            for r in range(self.num_rows):
                for c in range(r + 1, self.num_rows):
                    var idx_rc = r * self.row_size + c
                    var idx_cr = c * self.row_size + r

                    var tmp_re = r_ptr[idx_rc]
                    var tmp_im = i_ptr[idx_rc]
                    r_ptr[idx_rc] = r_ptr[idx_cr]
                    i_ptr[idx_rc] = i_ptr[idx_cr]
                    r_ptr[idx_cr] = tmp_re
                    i_ptr[idx_cr] = tmp_im
        else:
            # Rectangular fallback: High-performance flat bit-reversal
            var n_total = 1 << self.n
            var log_n = self.n
            var s_re = List[FloatType](length=n_total, fill=0.0)
            var s_im = List[FloatType](length=n_total, fill=0.0)

            var ptr_in_re = self.re.unsafe_ptr()
            var ptr_in_im = self.im.unsafe_ptr()
            var ptr_out_re = s_re.unsafe_ptr()
            var ptr_out_im = s_im.unsafe_ptr()

            @parameter
            fn worker(idx: Int):
                alias width = 8
                var base = idx * width
                var offsets = SIMD[DType.uint64, width]()
                for i in range(width):
                    offsets[i] = i
                var vec_idx = SIMD[DType.uint64, width](base) + offsets
                var r_idx_u64 = bit_reverse(vec_idx) >> (64 - log_n)
                var r_idx = r_idx_u64.cast[DType.int64]()
                ptr_out_re.store(base, ptr_in_re.gather(r_idx))
                ptr_out_im.store(base, ptr_in_im.gather(r_idx))

            from algorithm import parallelize

            parallelize[worker](n_total // 8)
            self.re = s_re^
            self.im = s_im^

    fn get_row_offset(self, row: Int) -> Int:
        """Get the offset to a specific row."""
        return row * self.row_size

    fn transform_row(mut self, row: Int, target: Int, gate: Gate):
        """Transform a single row using scalar operations.

        Args:
            row: Row index.
            target: Target qubit within the row.
            gate: Gate to apply.
        """
        var offset = self.get_row_offset(row)
        var re_ptr = self.re.unsafe_ptr() + offset
        var im_ptr = self.im.unsafe_ptr() + offset

        # Extract gate components
        var g00_re = gate[0][0].re
        var g00_im = gate[0][0].im
        var g01_re = gate[0][1].re
        var g01_im = gate[0][1].im
        var g10_re = gate[1][0].re
        var g10_im = gate[1][0].im
        var g11_re = gate[1][1].re
        var g11_im = gate[1][1].im

        var stride = 1 << target

        # Process pairs
        for k in range(self.row_size // (2 * stride)):
            var base = k * 2 * stride
            for i in range(base, base + stride):
                var idx = i
                var idx1 = idx + stride

                # Load
                var re0 = re_ptr[idx]
                var im0 = im_ptr[idx]
                var re1 = re_ptr[idx1]
                var im1 = im_ptr[idx1]

                # Apply gate
                re_ptr[idx] = (
                    g00_re * re0 - g00_im * im0 + g01_re * re1 - g01_im * im1
                )
                im_ptr[idx] = (
                    g00_re * im0 + g00_im * re0 + g01_re * im1 + g01_im * re1
                )
                re_ptr[idx1] = (
                    g10_re * re0 - g10_im * im0 + g11_re * re1 - g11_im * im1
                )
                im_ptr[idx1] = (
                    g10_re * im0 + g10_im * re0 + g11_re * im1 + g11_im * re1
                )

    fn transform_row_simd[
        simd_width: Int = 8
    ](mut self, row: Int, target: Int, gate: Gate):
        """Transform a single row using SIMD operations.

        Args:
            row: Row index.
            target: Target qubit within the row.
            gate: Gate to apply.
        """
        var offset = self.get_row_offset(row)
        var re_ptr = self.re.unsafe_ptr() + offset
        var im_ptr = self.im.unsafe_ptr() + offset

        # Extract gate components
        var g00_re = gate[0][0].re
        var g00_im = gate[0][0].im
        var g01_re = gate[0][1].re
        var g01_im = gate[0][1].im
        var g10_re = gate[1][0].re
        var g10_im = gate[1][0].im
        var g11_re = gate[1][1].re
        var g11_im = gate[1][1].im

        var stride = 1 << target

        # --- OLD MODULO-BASED METHOD (Commented out) ---
        # @always_inline
        # @parameter
        # fn butterfly_simd(idx: Int):
        #     var zero_idx = 2 * idx - idx % stride
        #     var one_idx = zero_idx + stride
        #
        #     # Load SIMD vectors
        #     var elem0_re = re_ptr.load[width=simd_width](zero_idx)
        #     var elem0_im = im_ptr.load[width=simd_width](zero_idx)
        #     var elem1_re = re_ptr.load[width=simd_width](one_idx)
        #     var elem1_im = im_ptr.load[width=simd_width](one_idx)
        #
        #     # Save originals
        #     var elem0_orig_re = elem0_re
        #     var elem0_orig_im = elem0_im
        #     var elem1_orig_re = elem1_re
        #     var elem1_orig_im = elem1_im
        #
        #     # Apply gate
        #     re_ptr.store(zero_idx, elem0_orig_re * g00_re - elem0_orig_im * g00_im + elem1_orig_re * g01_re - elem1_orig_im * g01_im)
        #     im_ptr.store(zero_idx, elem0_orig_re * g00_im + elem0_orig_im * g00_re + elem1_orig_re * g01_im + elem1_orig_im * g01_re)
        #     re_ptr.store(one_idx, elem0_orig_re * g10_re - elem0_orig_im * g10_im + elem1_orig_re * g11_re - elem1_orig_im * g11_im)
        #     im_ptr.store(one_idx, elem0_orig_re * g10_im + elem0_orig_im * g10_re + elem1_orig_re * g11_im + elem1_orig_im * g11_re)
        #
        # for idx in range(0, self.row_size // 2, simd_width):
        #     butterfly_simd(idx)

        # --- NEW NESTED-LOOP METHOD (Optimized indexing) ---
        for k in range(0, self.row_size, 2 * stride):
            if stride >= simd_width:
                # Full block SIMD
                for i in range(0, stride, simd_width):
                    var idx0 = k + i
                    var idx1 = idx0 + stride
                    var r0 = re_ptr.load[width=simd_width](idx0)
                    var i0 = im_ptr.load[width=simd_width](idx0)
                    var r1 = re_ptr.load[width=simd_width](idx1)
                    var i1 = im_ptr.load[width=simd_width](idx1)

                    re_ptr.store(
                        idx0,
                        r0 * g00_re - i0 * g00_im + r1 * g01_re - i1 * g01_im,
                    )
                    im_ptr.store(
                        idx0,
                        r0 * g00_im + i0 * g00_re + r1 * g01_im + i1 * g01_re,
                    )
                    re_ptr.store(
                        idx1,
                        r0 * g10_re - i0 * g10_im + r1 * g11_re - i1 * g11_im,
                    )
                    im_ptr.store(
                        idx1,
                        r0 * g10_im + i0 * g10_re + r1 * g11_im + i1 * g11_re,
                    )
            else:
                # Small stride fallback: process linearly within the SIMD width if possible,
                # but for simplicity and correctness, we use scalar for stride < simd_width.
                for i in range(stride):
                    var idx0 = k + i
                    var idx1 = idx0 + stride
                    var r0 = re_ptr[idx0]
                    var i0 = im_ptr[idx0]
                    var r1 = re_ptr[idx1]
                    var i1 = im_ptr[idx1]

                    re_ptr[idx0] = (
                        r0 * g00_re - i0 * g00_im + r1 * g01_re - i1 * g01_im
                    )
                    im_ptr[idx0] = (
                        r0 * g00_im + i0 * g00_re + r1 * g01_im + i1 * g01_re
                    )
                    re_ptr[idx1] = (
                        r0 * g10_re - i0 * g10_im + r1 * g11_re - i1 * g11_im
                    )
                    im_ptr[idx1] = (
                        r0 * g10_im + i0 * g10_re + r1 * g11_im + i1 * g11_re
                    )

    fn c_transform_row_scalar(
        mut self, row: Int, control: Int, target: Int, gate: Gate
    ):
        var offset = self.get_row_offset(row)
        var c_stride = 1 << control
        var t_stride = 1 << target

        # Extract gate components
        var g00_re = gate[0][0].re
        var g00_im = gate[0][0].im
        var g01_re = gate[0][1].re
        var g01_im = gate[0][1].im
        var g10_re = gate[1][0].re
        var g10_im = gate[1][0].im
        var g11_re = gate[1][1].re
        var g11_im = gate[1][1].im

        if target < control:
            for k in range(self.row_size // (2 * c_stride)):
                var block_start = k * 2 * c_stride + c_stride
                for j in range(c_stride // (2 * t_stride)):
                    var sub_start = block_start + j * 2 * t_stride
                    for idx in range(sub_start, sub_start + t_stride):
                        var idx0 = offset + idx
                        var idx1 = idx0 + t_stride
                        var r0 = self.re[idx0]
                        var i0 = self.im[idx0]
                        var r1 = self.re[idx1]
                        var i1 = self.im[idx1]
                        self.re[idx0] = (
                            g00_re * r0
                            - g00_im * i0
                            + g01_re * r1
                            - g01_im * i1
                        )
                        self.im[idx0] = (
                            g00_re * i0
                            + g00_im * r0
                            + g01_re * i1
                            + g01_im * r1
                        )
                        self.re[idx1] = (
                            g10_re * r0
                            - g10_im * i0
                            + g11_re * r1
                            - g11_im * i1
                        )
                        self.im[idx1] = (
                            g10_re * i0
                            + g10_im * r0
                            + g11_re * i1
                            + g11_im * r1
                        )
        else:  # target > control
            for k in range(self.row_size // (2 * t_stride)):
                var base = k * 2 * t_stride
                var num_periods = t_stride // (2 * c_stride)
                for p in range(num_periods):
                    var p_start = base + p * 2 * c_stride + c_stride
                    for idx in range(p_start, p_start + c_stride):
                        var idx0 = offset + idx
                        var idx1 = idx0 + t_stride
                        var r0 = self.re[idx0]
                        var i0 = self.im[idx0]
                        var r1 = self.re[idx1]
                        var i1 = self.im[idx1]
                        self.re[idx0] = (
                            g00_re * r0
                            - g00_im * i0
                            + g01_re * r1
                            - g01_im * i1
                        )
                        self.im[idx0] = (
                            g00_re * i0
                            + g00_im * r0
                            + g01_re * i1
                            + g01_im * r1
                        )
                        self.re[idx1] = (
                            g10_re * r0
                            - g10_im * i0
                            + g11_re * r1
                            - g11_im * i1
                        )
                        self.im[idx1] = (
                            g10_re * i0
                            + g10_im * r0
                            + g11_re * i1
                            + g11_im * r1
                        )

    fn c_transform_row_simd[
        simd_width: Int = 8
    ](mut self, row: Int, control: Int, target: Int, gate: Gate):
        var offset = self.get_row_offset(row)
        var re_ptr = self.re.unsafe_ptr() + offset
        var im_ptr = self.im.unsafe_ptr() + offset
        var c_stride = 1 << control
        var t_stride = 1 << target

        # Extract gate components
        var g00_re = gate[0][0].re
        var g00_im = gate[0][0].im
        var g01_re = gate[0][1].re
        var g01_im = gate[0][1].im
        var g10_re = gate[1][0].re
        var g10_im = gate[1][0].im
        var g11_re = gate[1][1].re
        var g11_im = gate[1][1].im

        @always_inline
        @parameter
        fn butterfly_simd(idx: Int):
            var elem0_re = re_ptr.load[width=simd_width](idx)
            var elem0_im = im_ptr.load[width=simd_width](idx)
            var elem1_re = re_ptr.load[width=simd_width](idx + t_stride)
            var elem1_im = im_ptr.load[width=simd_width](idx + t_stride)
            re_ptr.store[width=simd_width](
                idx,
                g00_re * elem0_re
                - g00_im * elem0_im
                + g01_re * elem1_re
                - g01_im * elem1_im,
            )
            im_ptr.store[width=simd_width](
                idx,
                g00_re * elem0_im
                + g00_im * elem0_re
                + g01_re * elem1_im
                + g01_im * elem1_re,
            )
            re_ptr.store[width=simd_width](
                idx + t_stride,
                g10_re * elem0_re
                - g10_im * elem0_im
                + g11_re * elem1_re
                - g11_im * elem1_im,
            )
            im_ptr.store[width=simd_width](
                idx + t_stride,
                g10_re * elem0_im
                + g10_im * elem0_re
                + g11_re * elem1_im
                + g11_im * elem1_re,
            )

        if target < control:
            for k in range(self.row_size // (2 * c_stride)):
                var block_start = k * 2 * c_stride + c_stride
                for j in range(c_stride // (2 * t_stride)):
                    var sub_start = block_start + j * 2 * t_stride
                    for idx in range(
                        sub_start, sub_start + t_stride, simd_width
                    ):
                        butterfly_simd(idx)
        else:  # target > control
            for k in range(self.row_size // (2 * t_stride)):
                var base = k * 2 * t_stride
                var num_periods = t_stride // (2 * c_stride)
                for p in range(num_periods):
                    var p_start = base + p * 2 * c_stride + c_stride
                    for idx in range(p_start, p_start + c_stride, simd_width):
                        butterfly_simd(idx)

    fn transform[
        with_simd: Bool = False
    ](mut self, target: Int, gate: Gate) raises:
        """Apply gate to the grid state.

        If target < col_bits, operates within rows (can parallelize by row).
        If target >= col_bits, operates across rows (parallelize by column).

        Args:
            target: Target qubit index.
            gate: Gate to apply.
        """
        if target < self.col_bits:
            # Operation within rows - parallelize by row!
            from algorithm import parallelize

            @parameter
            fn process_row(row: Int):
                var stride = 1 << target
                if with_simd and stride >= 8 and self.row_size >= 8:
                    self.transform_row_simd[8](row, target, gate)
                else:
                    self.transform_row(row, target, gate)

            parallelize[process_row](self.num_rows)
        else:
            # Operation across rows - parallelize by column!
            from algorithm import parallelize

            var t = target - self.col_bits  # Target within row dimension
            var stride = 1 << t
            alias chunk_size = 8

            if with_simd and self.row_size >= chunk_size:

                @parameter
                fn process_column_simd(chunk_idx: Int):
                    var col_base = chunk_idx * chunk_size
                    var g00_re = gate[0][0].re
                    var g00_im = gate[0][0].im
                    var g01_re = gate[0][1].re
                    var g01_im = gate[0][1].im
                    var g10_re = gate[1][0].re
                    var g10_im = gate[1][0].im
                    var g11_re = gate[1][1].re
                    var g11_im = gate[1][1].im
                    var re_ptr = self.re.unsafe_ptr()
                    var im_ptr = self.im.unsafe_ptr()
                    for k in range(self.num_rows // (2 * stride)):
                        for row_idx in range(
                            k * 2 * stride, k * 2 * stride + stride
                        ):
                            var idx0 = row_idx * self.row_size + col_base
                            var idx1 = (
                                row_idx + stride
                            ) * self.row_size + col_base
                            var r0 = re_ptr.load[width=chunk_size](idx0)
                            var i0 = im_ptr.load[width=chunk_size](idx0)
                            var r1 = re_ptr.load[width=chunk_size](idx1)
                            var i1 = im_ptr.load[width=chunk_size](idx1)
                            re_ptr.store[width=chunk_size](
                                idx0,
                                g00_re * r0
                                - g00_im * i0
                                + g01_re * r1
                                - g01_im * i1,
                            )
                            im_ptr.store[width=chunk_size](
                                idx0,
                                g00_re * i0
                                + g00_im * r0
                                + g01_re * i1
                                + g01_im * r1,
                            )
                            re_ptr.store[width=chunk_size](
                                idx1,
                                g10_re * r0
                                - g10_im * i0
                                + g11_re * r1
                                - g11_im * i1,
                            )
                            im_ptr.store[width=chunk_size](
                                idx1,
                                g10_re * i0
                                + g10_im * r0
                                + g11_re * i1
                                + g11_im * r1,
                            )

                parallelize[process_column_simd](self.row_size // chunk_size)
            else:

                @parameter
                fn process_column(col: Int):
                    var g00_re = gate[0][0].re
                    var g00_im = gate[0][0].im
                    var g01_re = gate[0][1].re
                    var g01_im = gate[0][1].im
                    var g10_re = gate[1][0].re
                    var g10_im = gate[1][0].im
                    var g11_re = gate[1][1].re
                    var g11_im = gate[1][1].im
                    for k in range(self.num_rows // (2 * stride)):
                        for row_idx in range(
                            k * 2 * stride, k * 2 * stride + stride
                        ):
                            var idx0 = row_idx * self.row_size + col
                            var idx1 = (row_idx + stride) * self.row_size + col

                            # Load
                            var r0 = self.re[idx0]
                            var i0 = self.im[idx0]
                            var r1 = self.re[idx1]
                            var i1 = self.im[idx1]

                            # Apply gate
                            self.re[idx0] = (
                                g00_re * r0
                                - g00_im * i0
                                + g01_re * r1
                                - g01_im * i1
                            )
                            self.im[idx0] = (
                                g00_re * i0
                                + g00_im * r0
                                + g01_re * i1
                                + g01_im * r1
                            )
                            self.re[idx1] = (
                                g10_re * r0
                                - g10_im * i0
                                + g11_re * r1
                                - g11_im * i1
                            )
                            self.im[idx1] = (
                                g10_re * i0
                                + g10_im * r0
                                + g11_re * i1
                                + g11_im * r1
                            )

                parallelize[process_column](self.row_size)

    fn c_transform[
        with_simd: Bool = False
    ](mut self, control: Int, target: Int, gate: Gate) raises:
        """Apply controlled gate to the grid state."""
        from algorithm import parallelize

        if control < self.col_bits and target < self.col_bits:
            # Case 1: Both within rows
            @parameter
            fn process_row(row: Int):
                var t_stride = 1 << target
                var c_stride = 1 << control
                if (
                    with_simd
                    and min(c_stride, t_stride) >= 8
                    and self.row_size >= 8
                ):
                    self.c_transform_row_simd[8](row, control, target, gate)
                else:
                    self.c_transform_row_scalar(row, control, target, gate)

            parallelize[process_row](self.num_rows)

        elif control >= self.col_bits and target < self.col_bits:
            # Case 2: Control is across rows, Target is within row
            var c_row_bit = control - self.col_bits
            var c_stride = 1 << c_row_bit

            @parameter
            fn process_active_rows(k: Int):
                # Rows with control bit set
                for row_idx in range(
                    k * 2 * c_stride + c_stride, k * 2 * c_stride + 2 * c_stride
                ):
                    if with_simd and (1 << target) >= 8 and self.row_size >= 8:
                        self.transform_row_simd[8](row_idx, target, gate)
                    else:
                        self.transform_row(row_idx, target, gate)

            parallelize[process_active_rows](self.num_rows // (2 * c_stride))

        elif control < self.col_bits and target >= self.col_bits:
            # Case 3: Control is within row, Target is across rows
            var t_row_bit = target - self.col_bits
            var t_stride = 1 << t_row_bit
            var c_stride = 1 << control

            @parameter
            fn process_column_pairs(col: Int):
                # Only if control bit is set in col
                if (col // c_stride) % 2 == 1:
                    var g00_re = gate[0][0].re
                    var g00_im = gate[0][0].im
                    var g01_re = gate[0][1].re
                    var g01_im = gate[0][1].im
                    var g10_re = gate[1][0].re
                    var g10_im = gate[1][0].im
                    var g11_re = gate[1][1].re
                    var g11_im = gate[1][1].im
                    for k in range(self.num_rows // (2 * t_stride)):
                        for row_idx in range(
                            k * 2 * t_stride, k * 2 * t_stride + t_stride
                        ):
                            var idx0 = row_idx * self.row_size + col
                            var idx1 = (
                                row_idx + t_stride
                            ) * self.row_size + col
                            var r0 = self.re[idx0]
                            var i0 = self.im[idx0]
                            var r1 = self.re[idx1]
                            var i1 = self.im[idx1]
                            self.re[idx0] = (
                                g00_re * r0
                                - g00_im * i0
                                + g01_re * r1
                                - g01_im * i1
                            )
                            self.im[idx0] = (
                                g00_re * i0
                                + g00_im * r0
                                + g01_re * i1
                                + g01_im * r1
                            )
                            self.re[idx1] = (
                                g10_re * r0
                                - g10_im * i0
                                + g11_re * r1
                                - g11_im * i1
                            )
                            self.im[idx1] = (
                                g10_re * i0
                                + g10_im * r0
                                + g11_re * i1
                                + g11_im * r1
                            )

            parallelize[process_column_pairs](self.row_size)

        else:
            # Case 4: Both across rows
            var c_row_bit = control - self.col_bits
            var t_row_bit = target - self.col_bits
            var c_stride = 1 << c_row_bit
            var t_stride = 1 << t_row_bit

            @parameter
            fn process_column_monolithic(col: Int):
                var g00_re = gate[0][0].re
                var g00_im = gate[0][0].im
                var g01_re = gate[0][1].re
                var g01_im = gate[0][1].im
                var g10_re = gate[1][0].re
                var g10_im = gate[1][0].im
                var g11_re = gate[1][1].re
                var g11_im = gate[1][1].im

                if t_row_bit < c_row_bit:
                    for k in range(self.num_rows // (2 * c_stride)):
                        var block_start = k * 2 * c_stride + c_stride
                        for j in range(c_stride // (2 * t_stride)):
                            var sub_start = block_start + j * 2 * t_stride
                            for row_idx in range(
                                sub_start, sub_start + t_stride
                            ):
                                var idx0 = row_idx * self.row_size + col
                                var idx1 = (
                                    row_idx + t_stride
                                ) * self.row_size + col
                                var r0 = self.re[idx0]
                                var i0 = self.im[idx0]
                                var r1 = self.re[idx1]
                                var i1 = self.im[idx1]
                                self.re[idx0] = (
                                    g00_re * r0
                                    - g00_im * i0
                                    + g01_re * r1
                                    - g01_im * i1
                                )
                                self.im[idx0] = (
                                    g00_re * i0
                                    + g00_im * r0
                                    + g01_re * i1
                                    + g01_im * r1
                                )
                                self.re[idx1] = (
                                    g10_re * r0
                                    - g10_im * i0
                                    + g11_re * r1
                                    - g11_im * i1
                                )
                                self.im[idx1] = (
                                    g10_re * i0
                                    + g10_im * r0
                                    + g11_re * i1
                                    + g11_im * r1
                                )
                else:  # t_row_bit > c_row_bit
                    for k in range(self.num_rows // (2 * t_stride)):
                        var base = k * 2 * t_stride
                        var num_periods = t_stride // (2 * c_stride)
                        for p in range(num_periods):
                            var p_start = base + p * 2 * c_stride + c_stride
                            for row_idx in range(p_start, p_start + c_stride):
                                var idx0 = row_idx * self.row_size + col
                                var idx1 = (
                                    row_idx + t_stride
                                ) * self.row_size + col
                                var r0 = self.re[idx0]
                                var i0 = self.im[idx0]
                                var r1 = self.re[idx1]
                                var i1 = self.im[idx1]
                                self.re[idx0] = (
                                    g00_re * r0
                                    - g00_im * i0
                                    + g01_re * r1
                                    - g01_im * i1
                                )
                                self.im[idx0] = (
                                    g00_re * i0
                                    + g00_im * r0
                                    + g01_re * i1
                                    + g01_im * r1
                                )
                                self.re[idx1] = (
                                    g10_re * r0
                                    - g10_im * i0
                                    + g11_re * r1
                                    - g11_im * i1
                                )
                                self.im[idx1] = (
                                    g10_re * i0
                                    + g10_im * r0
                                    + g11_re * i1
                                    + g11_im * r1
                                )

            parallelize[process_column_monolithic](self.row_size)

    fn execute[
        with_simd: Bool = True
    ](mut self, circuit: QuantumCircuit) raises:
        """Execute a circuit on this grid state.

        Args:
            circuit: The circuit to execute.
        """
        from butterfly.core.circuit import (
            GateTransformation,
            SingleControlGateTransformation,
        )

        for i in range(len(circuit.transformations)):
            var t = circuit.transformations[i]

            # Handle gate transformations
            if t.isa[GateTransformation]():
                var gt = t[GateTransformation].copy()
                self.transform[with_simd](gt.target, gt.gate)
            elif t.isa[SingleControlGateTransformation]():
                var g = t[SingleControlGateTransformation].copy()
                self.c_transform[with_simd](g.control, g.target, g.gate)
            elif t.isa[BitReversalTransformation]():
                self.bit_reverse()
