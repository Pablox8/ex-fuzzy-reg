import pytest
import numpy as np
import sys
import os

# Add the library path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ex_fuzzy_reg import fuzzy_sets as fs


def test_trapezoidal_membership_function():
    """Test the trapezoidal membership function."""
    # Test with typical trapezoidal parameters
    fs_test = fs.TrapezoidalFS("trial", [0.2, 0.4, 0.6, 0.8], [0, 1])
    test_points = np.array([0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0])
    
    result = fs_test.membership(test_points)
    
    # Check key points
    assert result[0] == 0.0  # Before start
    assert result[1] == 0.0  # At start
    assert result[3] == 1.0  # In plateau
    assert result[5] == 0.0  # At end
    assert result[6] == 0.0  # After end

def test_triangular_membership_function():
    """Test triangular membership function."""
    # Triangular: [a, b, c] where b is the peak
    fs_test = fs.TriangularFS("trial", [0.2, 0.5, 0.8], [0, 1])
    test_points = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    
    result = fs_test.membership(test_points)
    
    assert result[0] == 0.0  # At left base
    assert result[2] == 1.0  # At peak
    assert result[4] == 0.0  # At right base

def test_cut_function():
    """Tests that cut compute correctly."""
    fs_test = fs.TrapezoidalFS("trial", [1, 2, 3, 4], [0, 5])

    fs2 = fs.cut(fs_test, h=0.5)
    assert fs2.membership_parameters == [1, 1.5, 3.5, 4], '[h=0.5] Cut membership parameters not correctly computed'
    assert fs2.height == 0.5, '[h=0.5] Cut max height not correctly computed'

    fs3 = fs.cut(fs_test, h=0)
    assert fs3.membership_parameters == [0], '[h=0] Cut membership parameters not correctly computed'
    assert fs3.height == 0, '[h=0] Cut max height not correctly computed'

    fs4 = fs.cut(fs_test, h=1)
    assert fs4.membership_parameters == [1, 2, 3, 4], '[h=1] Cut membership parameters not correctly computed'
    assert fs4.height == 1, '[h=1] Cut max height not correctly computed'

def test_trapezoidal_union_function():
    t1_x = [0, 1, 3, 5]
    t1_h = [0, 0.3, 0.3, 0]

    t2_x = [2, 4, 5, 6]
    t2_h = [0, 0.4, 0.4, 0]

    t3_x = [1, 3, 4, 7]
    t3_h = [0, 0.55, 0.55, 0]

    t4_x = [5, 6, 7, 9]
    t4_h = [0, 0.9, 0.9, 0]
    pass


if __name__ == '__main__':
    test_trapezoidal_membership_function()
    test_triangular_membership_function()
    test_cut_function()
    test_trapezoidal_union_function()