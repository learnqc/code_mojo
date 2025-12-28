"""
Four-quarter zero-copy split-state execution (v2).
Follows the two-half pattern from execute_split_runtime with helper functions.
"""

from butterfly.core.state import QuantumState
from butterfly.core.circuit import (
    QuantumCircuit,
    Transformation,
    GateTransformation,
    SingleControlGateTransformation,
    Gate,
    get_target,
)
from butterfly.core.execute_split_runtime import get_controls
from butterfly.core.state import transform, c_transform
from collections import InlineArray


fn swap_middle_quarters(mut middle: InlineArray[Int, 2]):
    """Zero-copy swap of middle quarters by swapping indices."""
    var temp = middle[0]
    middle[0] = middle[1]
    middle[1] = temp


fn transform_two_quarters(
    quarter_size: Int,
    mut state: QuantumState,
    q0: Int,
    q1: Int,
    target: Int,
    gate: Gate,
):
    """Apply gate to two quarters (q0 and q1).

    Args:
        state: The quantum state.
        q0: First quarter index.
        q1: Second quarter index.
        target: Target qubit within each quarter.
        gate: Gate to apply.
    """
    var stride = 1 << target

    # When stride == quarter_size, target is qubit n-2
    # The stride-based algorithm doesn't work, so apply between quarters instead
    if stride == quarter_size:
        transform_single_quarter_pair(quarter_size, state, q0, q1, gate)
        return

    # Normal case: target < n-2, apply within each quarter
    var offset0 = q0 * quarter_size
    var offset1 = q1 * quarter_size

    # Extract gate components
    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im

    # Apply to both quarters
    for quarter_idx in range(2):
        var offset = offset0 if quarter_idx == 0 else offset1

        for k in range(quarter_size // (2 * stride)):
            var base = k * 2 * stride
            for idx in range(base, base + stride):
                var idx1 = idx + stride

                # Load
                var re0 = state.re[offset + idx]
                var im0 = state.im[offset + idx]
                var re1 = state.re[offset + idx1]
                var im1 = state.im[offset + idx1]

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
                state.re[offset + idx] = new_re0
                state.im[offset + idx] = new_im0
                state.re[offset + idx1] = new_re1
                state.im[offset + idx1] = new_im1


fn c_transform_two_quarters(
    quarter_size: Int,
    mut state: QuantumState,
    q0: Int,
    q1: Int,
    control: Int,
    target: Int,
    gate: Gate,
):
    """Apply controlled gate to two quarters (q0 and q1).

    Args:
        state: The quantum state.
        q0: First quarter index.
        q1: Second quarter index.
        control: Control qubit within each quarter.
        target: Target qubit within each quarter.
        gate: Gate to apply.
    """
    var stride = 1 << target

    # When stride == quarter_size, target is qubit n-2
    # Need to apply controlled gate between quarters instead of within
    if stride == quarter_size:
        # Apply controlled gate between the two quarters
        var g00_re = gate[0][0].re
        var g00_im = gate[0][0].im
        var g01_re = gate[0][1].re
        var g01_im = gate[0][1].im
        var g10_re = gate[1][0].re
        var g10_im = gate[1][0].im
        var g11_re = gate[1][1].re
        var g11_im = gate[1][1].im

        var control_mask = 1 << control
        var offset_a = q0 * quarter_size
        var offset_b = q1 * quarter_size

        for idx in range(quarter_size):
            if (idx & control_mask) != 0:
                var re0 = state.re[offset_a + idx]
                var im0 = state.im[offset_a + idx]
                var re1 = state.re[offset_b + idx]
                var im1 = state.im[offset_b + idx]

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

                state.re[offset_a + idx] = new_re0
                state.im[offset_a + idx] = new_im0
                state.re[offset_b + idx] = new_re1
                state.im[offset_b + idx] = new_im1
        return

    # Normal case: target < n-2, apply within each quarter
    var offset0 = q0 * quarter_size
    var offset1 = q1 * quarter_size

    # Extract gate components
    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im

    var control_mask = 1 << control

    # Apply to both quarters
    for quarter_idx in range(2):
        var offset = offset0 if quarter_idx == 0 else offset1

        for k in range(quarter_size // (2 * stride)):
            var base = k * 2 * stride
            for idx in range(base, base + stride):
                # Only apply if control bit is 1
                if (idx & control_mask) != 0:
                    var idx1 = idx + stride

                    # Load
                    var re0 = state.re[offset + idx]
                    var im0 = state.im[offset + idx]
                    var re1 = state.re[offset + idx1]
                    var im1 = state.im[offset + idx1]

                    # Apply gate
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
                    state.re[offset + idx] = new_re0
                    state.im[offset + idx] = new_im0
                    state.re[offset + idx1] = new_re1
                    state.im[offset + idx1] = new_im1


fn transform_between_quarter_pairs(
    quarter_size: Int, mut state: QuantumState, qa: Int, qb: Int, gate: Gate
):
    """Apply gate between two pairs of quarters: (q0↔q1) and (q2↔q3).

    Used for qubits n-2 and n-1 which are represented by quarter selection.

    Args:
        state: The quantum state.
        q0, q1: First pair of quarters.
        q2, q3: Second pair of quarters.
        gate: Gate to apply.
    """
    # Just call transform_single_quarter_pair
    transform_single_quarter_pair(quarter_size, state, qa, qb, gate)


fn transform_single_quarter_pair(
    quarter_size: Int, mut state: QuantumState, qa: Int, qb: Int, gate: Gate
):
    """Apply gate between a single pair of quarters.

    Args:
        state: The quantum state.
        qa: First quarter index.
        qb: Second quarter index.
        gate: Gate to apply.
    """
    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im

    var offset_a = qa * quarter_size
    var offset_b = qb * quarter_size

    for idx in range(quarter_size):
        var re0 = state.re[offset_a + idx]
        var im0 = state.im[offset_a + idx]
        var re1 = state.re[offset_b + idx]
        var im1 = state.im[offset_b + idx]

        var new_re0 = g00_re * re0 - g00_im * im0 + g01_re * re1 - g01_im * im1
        var new_im0 = g00_re * im0 + g00_im * re0 + g01_re * im1 + g01_im * re1
        var new_re1 = g10_re * re0 - g10_im * im0 + g11_re * re1 - g11_im * im1
        var new_im1 = g10_re * im0 + g10_im * re0 + g11_re * im1 + g11_im * re1

        state.re[offset_a + idx] = new_re0
        state.im[offset_a + idx] = new_im0
        state.re[offset_b + idx] = new_re1
        state.im[offset_b + idx] = new_im1


fn c_transform_between_quarter_pairs(
    quarter_size: Int,
    mut state: QuantumState,
    q0: Int,
    q1: Int,
    q2: Int,
    q3: Int,
    control: Int,
    gate: Gate,
):
    """Apply controlled gate between two pairs of quarters with control check.

    Args:
        state: The quantum state.
        q0, q1: First pair of quarters.
        q2, q3: Second pair of quarters.
        control: Control qubit.
        gate: Gate to apply.
    """
    # Extract gate components
    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im

    var control_mask = 1 << control

    # Process both pairs
    var quarter_pairs = InlineArray[InlineArray[Int, 2], 2](
        InlineArray[Int, 2](q0, q1),
        InlineArray[Int, 2](q2, q3),
    )

    for pair_idx in range(2):
        var qa = quarter_pairs[pair_idx][0]
        var qb = quarter_pairs[pair_idx][1]
        var offset_a = qa * quarter_size
        var offset_b = qb * quarter_size

        # Apply gate between these two quarters, only where control=1
        for idx in range(quarter_size):
            if (idx & control_mask) != 0:
                var re0 = state.re[offset_a + idx]
                var im0 = state.im[offset_a + idx]
                var re1 = state.re[offset_b + idx]
                var im1 = state.im[offset_b + idx]

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

                state.re[offset_a + idx] = new_re0
                state.im[offset_a + idx] = new_im0
                state.re[offset_b + idx] = new_re1
                state.im[offset_b + idx] = new_im1


fn execute_split_four_quarters_v2_runtime(
    n: Int, mut state: QuantumState, circuit: QuantumCircuit
) raises:
    """Execute circuit on four-quarter split state (runtime version).

    Follows the two-half pattern from execute_split_runtime.
    Quarter pairs are always: (0, middle[0]) and (middle[1], 3).

    Args:
        n: Number of qubits (runtime parameter).
        state: The quantum state to modify in-place.
        circuit: The circuit to execute.
    """
    var quarter_size = 1 << (n - 2)

    # Track which physical quarters are the middle two (01 and 10)
    # Initially: middle[0] = 1 (quarter 01), middle[1] = 2 (quarter 10)
    var middle = InlineArray[Int, 2](1, 2)

    # Process transformations - following two-half pattern exactly
    for i in range(len(circuit.transformations)):
        var transformation = circuit.transformations[i]
        var target = get_target(transformation)

        var last_qubit_target: Bool = target == n - 1

        # Get controls
        var controls = get_controls(transformation)
        var control = controls[0] if len(controls) > 0 else -1

        var swap_last: Bool = False
        if last_qubit_target:
            target = target - 1
            if control == target:
                control = control + 1
            else:
                swap_last = True

            if swap_last:
                swap_middle_quarters(middle)

        if transformation.isa[GateTransformation]():
            var g = transformation[GateTransformation].copy()
            var gate = g.gate

            # Apply to both quarter pairs - helper handles target==n-2 internally
            transform_two_quarters(
                quarter_size, state, 0, middle[0], target, gate
            )
            transform_two_quarters(
                quarter_size, state, middle[1], 3, target, gate
            )
        elif transformation.isa[SingleControlGateTransformation]():
            var g = transformation[SingleControlGateTransformation].copy()
            var gate = g.gate

            # Check if control is the last qubit (following two-half pattern)
            if control == n - 1:
                # Control is last qubit - only apply to quarters where qubit n-1 = 1
                transform_two_quarters(
                    quarter_size, state, middle[1], 3, target, gate
                )
            elif control == n - 2:
                # Control is second-to-last qubit - only apply to quarters where qubit n-2 = 1
                # That's middle[0] and 3 (quarters 1 and 3)
                transform_two_quarters(
                    quarter_size, state, middle[0], 3, target, gate
                )
            else:
                # Control is not the last two qubits - apply to both pairs
                c_transform_two_quarters(
                    quarter_size, state, 0, middle[0], control, target, gate
                )
                c_transform_two_quarters(
                    quarter_size, state, middle[1], 3, control, target, gate
                )

        if swap_last:
            swap_middle_quarters(middle)

    # If middle != [1, 2], quarters 1 and 2 are swapped - swap them back
    assert_true(middle[0] == 1 and middle[1] == 2)


fn execute_split_four_quarters_v2[
    n: Int
](mut state: QuantumState, circuit: QuantumCircuit) raises:
    """Execute circuit on four-quarter split state (parametric version).

    This is a thin wrapper that calls the runtime version.

    Args:
        state: The quantum state to modify in-place.
        circuit: The circuit to execute.
    """
    execute_split_four_quarters_v2_runtime(n, state, circuit)
