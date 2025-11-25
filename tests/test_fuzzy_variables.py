import pytest
import numpy as np
import sys
import os

# Add the library path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ex_fuzzy_reg import fuzzy_sets as fs
from ex_fuzzy_reg import fuzzy_variable as fv

 
def test_fuzzy_variable_creation():
    """Test creation of fuzzy variable."""
    low = fs.TrapezoidalFS('low', [1, 2, 3, 4], [0, 9])
    medium = fs.TriangularFS('medium', [4, 5, 6], [0, 9])
    large = fs.TrapezoidalFS('large', [5, 6, 7, 8], [0, 9])

    sample_fuzzy_sets = [low, medium, large]    

    f_v = fv.FuzzyVariable('test_var', sample_fuzzy_sets, 'units')
    assert f_v.name == 'test_var'
    assert f_v.units == 'units'
    assert len(f_v.linguistic_variables) == len(sample_fuzzy_sets)


def test_fuzzy_variable_membership_computation():
    """Test fuzzy variable membership computation across all sets."""
    low = fs.TrapezoidalFS('low', [1, 2, 3, 4], [0, 9])
    medium = fs.TriangularFS('medium', [4, 5, 6], [0, 9])
    large = fs.TrapezoidalFS('large', [5, 6, 7, 8], [0, 9])

    sample_fuzzy_sets = [low, medium, large] 
    
    f_v = fv.FuzzyVariable('test_var', sample_fuzzy_sets, 'units')
    
    input_values = np.array([0.1, 0.5, 0.9])
    memberships = f_v.compute_memberships(input_values)
    
    # Should return matrix: samples x linguistic_variables
    expected_shape = (len(input_values), len(sample_fuzzy_sets))
    assert memberships.shape == expected_shape


def test_fuzzy_variable_with_t2_sets():
    """Test fuzzy variable with Type-2 fuzzy sets."""
    pass


def test_fuzzy_variable_linguistic_names():
    """Test retrieval of linguistic variable names."""
    low = fs.TrapezoidalFS('low', [1, 2, 3, 4], [0, 9])
    medium = fs.TriangularFS('medium', [4, 5, 6], [0, 9])
    large = fs.TrapezoidalFS('large', [5, 6, 7, 8], [0, 9])

    sample_fuzzy_sets = [low, medium, large] 
    
    f_v = fv.FuzzyVariable('test_var', sample_fuzzy_sets, 'units')
    names = f_v.linguistic_variable_names()
    
    expected_names = [fs.name for fs in sample_fuzzy_sets]
    assert names == expected_names


