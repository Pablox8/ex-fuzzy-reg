import pytest
import numpy as np
import sys
import os

# Add the library path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ex_fuzzy_reg import fuzzy_sets as fs

# TODO: update trapezoidal_union tests to use trapezoidal_triangular_union instead
def test_trapezoidal_membership_evaluates_key_points_correctly():
    """Test the trapezoidal membership function."""
    # Trapezoidal: [a, b, c, d] where b, c is the plateau
    fs_test = fs.TrapezoidalFS("trial", [0.2, 0.4, 0.6, 0.8], [0, 1])
    test_points = np.array([0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0])
    
    result = fs_test.membership(test_points)
    
    # Check key points
    assert result[0] == 0.0  # Before start
    assert result[1] == 0.0  # At start
    assert result[3] == 1.0  # In plateau
    assert result[5] == 0.0  # At end
    assert result[6] == 0.0  # After end


def test_triangular_membership_evaluates_base_and_peak_correctly():
    """Test triangular membership function."""
    # Triangular: [a, b, c] where b is the peak
    fs_test = fs.TriangularFS("trial", [0.2, 0.5, 0.8], [0, 1])
    test_points = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    
    result = fs_test.membership(test_points)
    
    assert result[0] == 0.0  # At left base
    assert result[2] == 1.0  # At peak
    assert result[4] == 0.0  # At right base


def test_gaussian_membership_is_symmetric_and_peaks_at_mean():
    """Test gaussian membership function."""
    # Gaussian: [mean, std]
    fs_test = fs.GaussianFS("trial", [0.0, 1.0], 50)
    test_points = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    result = fs_test.membership(test_points)

    assert result[2] == 1.0        # At mean
    assert result[1] == result[3]  # Symmetry
    assert result[0] < result[1]   # Decreasing away from mean
    assert result[4] < result[3]


def test_cut_trapezoidal_fs_scales_support_and_sets_height():
    """Test general cut operation on trapezoidal membership function."""
    fs_base = fs.TrapezoidalFS("trial", [1, 2, 3, 4], [0, 5])

    fs_cut = fs.cut(fs_base, h=0.5)
    assert fs_cut.membership_parameters == [1, 1.5, 3.5, 4]
    assert fs_cut.height == 0.5


def test_cut_trapezoidal_fs_with_zero_height_returns_empty_set():
    """Test cut function on trapezoidal membership function with h=0."""
    fs_base = fs.TrapezoidalFS("trial", [1, 2, 3, 4], [0, 5])

    fs_cut = fs.cut(fs_base, h=0.0)
    assert fs_cut.membership_parameters == [0, 0, 0, 0]
    assert fs_cut.height == 0.0


def test_cut_trapezoidal_fs_with_full_height_returns_original_set():
    """Test cut function on trapezoidal membership function with h=1."""
    fs_base = fs.TrapezoidalFS("trial", [1, 2, 3, 4], [0, 5])

    fs_cut = fs.cut(fs_base, h=1.0)
    assert fs_cut.membership_parameters == [1, 2, 3, 4]
    assert fs_cut.height == 1.0


def test_segments_may_intersect_when_overlapping_in_x():
    """Segments overlap in x."""
    s1 = [(0, 0), (3, 1)]
    s2 = [(2, 1), (4, 0)]

    assert fs.segments_may_intersect(s1, s2) is True


def test_segments_may_intersect_when_touching_at_endpoint():
    """Segments touch at boundary."""
    s1 = [(0, 0), (2, 1)]
    s2 = [(2, 1), (4, 0)]

    assert fs.segments_may_intersect(s1, s2) is True


def test_segments_may_intersect_returns_false_when_disjoint():
    """Segments do not overlap."""
    s1 = [(0, 0), (1, 1)]
    s2 = [(2, 1), (3, 0)]

    assert fs.segments_may_intersect(s1, s2) is False


def test_compute_intersection_x_returns_correct_value_for_crossing_lines():
    """Intersect two segments."""
    s1 = [(0, 0), (2, 2)]      # y = x
    s2 = [(0, 2), (2, 0)]      # y = -x + 2

    x = fs.compute_intersection_x(s1, s2)

    assert x == pytest.approx(1.0)


def test_compute_intersection_x_returns_none_for_parallel_lines():
    """Parallel segments should not intersect."""
    s1 = [(0, 0), (2, 2)]
    s2 = [(0, 1), (2, 3)]

    x = fs.compute_intersection_x(s1, s2)

    assert x is None


def test_compute_intersection_x_handles_vertical_segments_correctly():
    """Vertical segments return None."""
    s1 = [(1, 0), (1, 2)]
    s2 = [(0, 1), (2, 1)]

    x = fs.compute_intersection_x(s1, s2)

    assert x == 1


def test_trapezoidal_triangular_union_combines_overlapping_sets_correctly():
    fs1 = fs.TrapezoidalFS("T1", [0, 2, 3, 4], [0, 6], 0.7)
    fs2 = fs.TrapezoidalFS("T2", [2, 3, 4, 6], [0, 6])

    u_x, u_y = fs.trapezoidal_triangular_union([fs1, fs2])

    assert u_x == pytest.approx([0, 2, 2.7, 3, 4, 6])
    assert u_y == pytest.approx([0, 0.7, 0.7, 1, 1, 0])


def test_trapezoidal_triangular_union_ignores_empty_set():
    fs_test = fs.TrapezoidalFS("trial 1", [1, 2, 3, 4], [0, 5], 0.7)
    empty = fs.TrapezoidalFS("empty", [1, 2, 3, 4], [0, 5], 0)

    u_x, u_y = fs.trapezoidal_triangular_union([fs_test, empty])

    assert u_x == pytest.approx([1, 2, 3, 4])
    assert u_y == pytest.approx([0, 0.7, 0.7, 0])


def test_trapezoidal_triangular_union_takes_maximum_height_when_overlapping():
    fs_test = fs.TrapezoidalFS("trial 1", [1, 2, 3, 4], [0, 5], 0.7)
    all_max_h = fs.TrapezoidalFS("all max h", [1, 2, 3, 4], [0, 5], 1)
    
    u_x, u_y = fs.trapezoidal_triangular_union([fs_test, all_max_h])

    assert u_x == pytest.approx([1, 2, 3, 4])
    assert u_y == pytest.approx([0, 1, 1, 0])


def test_trapezoidal_triangular_union_returns_empty_lists_if_no_trapezoids_passed():
    u_x, u_y = fs.trapezoidal_union([])

    assert u_x == []
    assert u_y == []


def test_trapezoidal_triangular_union_returns_empty_lists_if_all_trapezoids_are_empty():
    fs_empty1 = fs.TrapezoidalFS("empty 1", [0, 0, 0, 0], [0, 0])
    fs_empty2 = fs.TrapezoidalFS("empty 2", [0, 0, 0, 0], [0, 0])
    u_x, u_y = fs.trapezoidal_union([fs_empty1, fs_empty2])

    assert u_x == []
    assert u_y == []


def test_centroid_defuzzification_returns_correct_value_for_matched_example():
    p_x1 = np.array([0, 2, 2.7, 3, 4, 6])
    p_y1 = np.array([0, 0.7, 0.7, 1, 1, 0])

    x_crisp_cont1 = fs.centroid_defuzzification(p_x1, p_y1)

    assert round(x_crisp_cont1, 3) == 3.187


def test_centroid_defuzzification_of_symmetric_triangle_is_center():
    """Simple triangular fuzzy set."""
    p_x = [0, 1, 2]
    p_y = [0, 1, 0]

    x_crisp = fs.centroid_defuzzification(p_x, p_y)

    assert x_crisp == pytest.approx(1.0)


def test_centroid_defuzzification_raises_error_for_empty_inputs():
    """Raises ValueError if inputs are empty."""
    with pytest.raises(ValueError):
        fs.centroid_defuzzification([], [0, 1])

    with pytest.raises(ValueError):
        fs.centroid_defuzzification([0, 1], [])


def test_centroid_defuzzification_raises_error_for_length_mismatch():
    """Raises ValueError if inputs have different lengths."""
    p_x = [0, 1, 2]
    p_y = [0, 1]

    with pytest.raises(ValueError):
        fs.centroid_defuzzification(p_x, p_y)


def test_centroid_defuzzification_of_constant_segment_is_midpoint():
    """Flat segment."""
    p_x = [0, 1]
    p_y = [2, 2]

    x_crisp = fs.centroid_defuzzification(p_x, p_y)

    # centroid of a flat segment is the midpoint
    assert x_crisp == pytest.approx(0.5)


def test_centroid_defuzzification_returns_zero_for_zero_area_set():
    """Returns 0 if total area is zero."""
    p_x = [0, 1]
    p_y = [0, 0]

    x_crisp = fs.centroid_defuzzification(p_x, p_y)

    assert x_crisp == 0


def test_centroid_defuzzification_ignores_vertical_segments():
    """Vertical segments contribute no area and are ignored."""
    p_x = [1, 1, 5, 5]
    p_y = [0, 1, 1, 0]

    x_crisp = fs.centroid_defuzzification(p_x, p_y)

    assert x_crisp == pytest.approx(3.0)