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
from butterfly.core.circuit import QuantumCircuit
from buffer import NDBuffer


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

        # Initialize to |0...0⟩
        self.re.append(1.0)
        self.im.append(0.0)
        for _ in range(1, total_size):
            self.re.append(0.0)
            self.im.append(0.0)

    fn size(self) -> Int:
        """Total number of amplitudes."""
        return 1 << self.n

    fn get_row_offset(self, row: Int) -> Int:
        """Get the offset to a specific row."""
        return row * self.row_size

    fn transform_row[
        row_size: Int
    ](mut self, row: Int, target: Int, gate: Gate):
        """Transform a single row using NDBuffer views.

        Args:
            row: Row index.
            target: Target qubit within the row.
            gate: Gate to apply.
        """
        var offset = self.get_row_offset(row)
        var base_re = self.re.unsafe_ptr() + offset
        var base_im = self.im.unsafe_ptr() + offset

        var buf_re = NDBuffer[Type, 1, _, row_size](base_re)
        var buf_im = NDBuffer[Type, 1, _, row_size](base_im)

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
        for k in range(row_size // (2 * stride)):
            var base = k * 2 * stride
            for idx in range(base, base + stride):
                var idx1 = idx + stride

                # Load
                var re0 = buf_re[idx]
                var im0 = buf_im[idx]
                var re1 = buf_re[idx1]
                var im1 = buf_im[idx1]

                # Apply gate
                var new_re0 = (
                    g00_re * re0 - g00_im * im0 + g01_re * re1 - g01_im * im1
                )
                var new_im0 = (
                    g00_re * im0 + g00_im * re0 + g01_re * im1 + g01_im * re1
                )
                var new_re1 = (
                    g10_re * re0 - g10_im * im0 + g11_re * re1 - g11_im * im1
                )
                var new_im1 = (
                    g10_re * im0 + g10_im * re0 + g11_re * im1 + g11_im * re1
                )

                # Store
                buf_re[idx] = new_re0
                buf_im[idx] = new_im0
                buf_re[idx1] = new_re1
                buf_im[idx1] = new_im1

    fn transform_row_simd[
        row_size: Int, simd_width: Int = 4
    ](mut self, row: Int, target: Int, gate: Gate):
        """Transform a single row using SIMD operations on NDBuffer views.

        Args:
            row: Row index.
            target: Target qubit within the row.
            gate: Gate to apply.
        """
        var offset = self.get_row_offset(row)
        var base_re = self.re.unsafe_ptr() + offset
        var base_im = self.im.unsafe_ptr() + offset

        var buf_re = NDBuffer[Type, 1, _, row_size](base_re)
        var buf_im = NDBuffer[Type, 1, _, row_size](base_im)

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

        # SIMD butterfly operation
        @always_inline
        @parameter
        fn butterfly_simd(idx: Int):
            var zero_idx = 2 * idx - idx % stride
            var one_idx = zero_idx + stride

            # Load SIMD vectors
            var elem0_re = buf_re.load[width=simd_width](zero_idx)
            var elem0_im = buf_im.load[width=simd_width](zero_idx)
            var elem1_re = buf_re.load[width=simd_width](one_idx)
            var elem1_im = buf_im.load[width=simd_width](one_idx)

            # Save originals
            var elem0_orig_re = elem0_re
            var elem0_orig_im = elem0_im
            var elem1_orig_re = elem1_re
            var elem1_orig_im = elem1_im

            # Apply gate with FMA
            elem0_re = elem0_orig_re.fma(
                g00_re,
                -g00_im * elem0_orig_im
                + elem1_orig_re.fma(g01_re, -g01_im * elem1_orig_im),
            )
            elem0_im = elem0_orig_re.fma(
                g00_im,
                g00_re * elem0_orig_im
                + elem1_orig_re.fma(g01_im, g01_re * elem1_orig_im),
            )
            elem1_re = elem0_orig_re.fma(
                g10_re,
                -g10_im * elem0_orig_im
                + elem1_orig_re.fma(g11_re, -g11_im * elem1_orig_im),
            )
            elem1_im = elem0_orig_re.fma(
                g10_im,
                g10_re * elem0_orig_im
                + elem1_orig_re.fma(g11_im, g11_re * elem1_orig_im),
            )

            # Store results
            buf_re.store[width=simd_width](zero_idx, elem0_re)
            buf_im.store[width=simd_width](zero_idx, elem0_im)
            buf_re.store[width=simd_width](one_idx, elem1_re)
            buf_im.store[width=simd_width](one_idx, elem1_im)

        # Process all pairs with SIMD
        for idx in range(0, row_size // 2, simd_width):
            butterfly_simd(idx)

    fn transform[row_size: Int](mut self, target: Int, gate: Gate) raises:
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
                self.transform_row[row_size](row, target, gate)

            parallelize[process_row](self.num_rows)
        else:
            # Operation across rows - parallelize by column!
            from algorithm import parallelize

            var t = target - self.col_bits  # Target within row dimension
            var stride = 1 << t

            @parameter
            fn process_column(col: Int):
                # Apply gate to pairs of rows within this column
                for k in range(self.num_rows // (2 * stride)):
                    for row_idx in range(
                        k * 2 * stride, k * 2 * stride + stride
                    ):
                        var idx0 = row_idx * self.row_size + col
                        var idx1 = (row_idx + stride) * self.row_size + col

                        # Load
                        var re0 = self.re[idx0]
                        var im0 = self.im[idx0]
                        var re1 = self.re[idx1]
                        var im1 = self.im[idx1]

                        # Apply gate
                        var g00_re = gate[0][0].re
                        var g00_im = gate[0][0].im
                        var g01_re = gate[0][1].re
                        var g01_im = gate[0][1].im
                        var g10_re = gate[1][0].re
                        var g10_im = gate[1][0].im
                        var g11_re = gate[1][1].re
                        var g11_im = gate[1][1].im

                        var new_re0 = (
                            g00_re * re0
                            - g00_im * im0
                            + g01_re * re1
                            - g01_im * im1
                        )
                        var new_im0 = (
                            g00_re * im0
                            + g00_im * re0
                            + g01_re * im1
                            + g01_im * re1
                        )
                        var new_re1 = (
                            g10_re * re0
                            - g10_im * im0
                            + g11_re * re1
                            - g11_im * im1
                        )
                        var new_im1 = (
                            g10_re * im0
                            + g10_im * re0
                            + g11_re * im1
                            + g11_im * re1
                        )

                        # Store
                        self.re[idx0] = new_re0
                        self.im[idx0] = new_im0
                        self.re[idx1] = new_re1
                        self.im[idx1] = new_im1

            parallelize[process_column](self.row_size)

    fn execute[row_size: Int](mut self, circuit: QuantumCircuit) raises:
        """Execute a circuit on this grid state.

        Args:
            circuit: The circuit to execute.
        """
        from butterfly.core.circuit import GateTransformation

        for i in range(len(circuit.transformations)):
            var t = circuit.transformations[i]

            # Handle gate transformations
            if t.isa[GateTransformation]():
                var gt = t[GateTransformation].copy()
                self.transform[row_size](gt.target, gt.gate)
            # Skip other transformation types for now
