from butterfly.core.circuit import QuantumTransformation, Transformation
from butterfly.core.gates import H, X
from butterfly.core.types import *
from testing import assert_true


def main():
    # basic constructor without controls
    var t = QuantumTransformation(H, 1)
    assert_true(t.target == 1)
    assert_true(not t.is_controlled())
    assert_true(t.num_controls() == 0)

    # add a control and verify
    t.add_control(0)
    assert_true(t.is_controlled())
    assert_true(t.num_controls() == 1)
    assert_true(t.controls[0] == 0)

    # constructor with initial controls (move-init of list)
    var ctrls = List[Int]()
    ctrls.append(2)
    # move the controls list into the transformation (List is not ImplicitlyCopyable)
    var t2 = QuantumTransformation(X, 3, ctrls^)
    assert_true(t2.target == 3)
    assert_true(t2.is_controlled())
    assert_true(t2.num_controls() == 1)
    assert_true(t2.controls[0] == 2)

    # copyinit creates an independent controls list
    var t3 = t2.copy()  # explicit copy
    t2.add_control(4)
    assert_true(t2.num_controls() == 2)
    assert_true(t3.num_controls() == 1)

    # alias works
    var t4 = Transformation(H, 0)
    assert_true(t4.target == 0)
