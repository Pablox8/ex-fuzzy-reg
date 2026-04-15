import pytest
import numpy as np
import sys
import os

# Add the library path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ex_fuzzy_reg import rules_reg_utils as utils
from ex_fuzzy_reg import fuzzy_sets as fs
from ex_fuzzy_reg import fuzzy_variable as fv
from ex_fuzzy_reg import rules_reg as rules


def test_RuleBaseRegT1_initializes_correctly():
    rbt1 = rules.RuleBaseRegT1([], [])

    assert rbt1.antecedents == []
    assert rbt1.rules == []
    assert rbt1.consequent is None
    assert rbt1.tnorm == np.prod
    assert rbt1.fuzzy_type() == fs.FUZZY_SETS.t1


def test_compute_antecedent_memberships_returns_correct_shape():
    # (variables, linguistic_variables)
    low = fs.TriangularFS('low', [1, 2, 3], [1, 4])
    high = fs.TriangularFS('high', [2, 3.5, 4], [1, 4])
    antecedent = fv.FuzzyVariable('antecedent', [low, high])

    low = fs.TriangularFS('low', [0, 1, 2], [0, 4])
    high = fs.TriangularFS('high', [1, 2, 4], [0, 4])
    consequent = fv.FuzzyVariable('consequent', [low, high])

    low_rule = rules.RuleSimple([0], 0) # low -> low
    high_rule = rules.RuleSimple([1], 1) # high -> high

    rb = rules.RuleBaseRegT1([antecedent], [low_rule, high_rule], consequent)
    x = np.array([2.5])
    assert rb.compute_antecedents_memberships(x).shape == (1, 2)


def test_compute_antecedent_memberships_returns_correct_values():   
    low = fs.TriangularFS('low', [1, 2, 3], [1, 4])
    high = fs.TriangularFS('high', [2, 3.5, 4], [1, 4])
    antecedent = fv.FuzzyVariable('antecedent', [low, high])

    low = fs.TriangularFS('low', [0, 1, 2], [0, 4])
    high = fs.TriangularFS('high', [1, 2, 4], [0, 4])
    consequent = fv.FuzzyVariable('consequent', [low, high])

    low_rule = rules.RuleSimple([0], 0) # low -> low
    high_rule = rules.RuleSimple([1], 1) # high -> high

    rb = rules.RuleBaseRegT1([antecedent], [low_rule, high_rule], consequent)
    x = np.array([2.5])
    assert (rb.compute_antecedents_memberships(x) == [[0.5, 1/3]]).all()


def test_compute_cut_heights_returns_correct_shape():
    # (rules,)
    low = fs.TriangularFS('low', [1, 2, 3], [1, 4])
    high = fs.TriangularFS('high', [2, 3.5, 4], [1, 4])
    antecedent = fv.FuzzyVariable('antecedent', [low, high])

    low = fs.TriangularFS('low', [0, 1, 2], [0, 4])
    high = fs.TriangularFS('high', [1, 2, 4], [0, 4])
    consequent = fv.FuzzyVariable('consequent', [low, high])

    low_rule = rules.RuleSimple([0], 0) # low -> low
    high_rule = rules.RuleSimple([1], 1) # high -> high

    rb = rules.RuleBaseRegT1([antecedent], [low_rule, high_rule], consequent)
    x = np.array([2.5])

    antecedents_memberships = rb.compute_antecedents_memberships(x)

    assert rb.compute_cut_heights(antecedents_memberships).shape == (2,)


def test_compute_cut_heights_returns_correct_values():
    low = fs.TriangularFS('low', [1, 2, 3], [1, 4])
    high = fs.TriangularFS('high', [2, 3.5, 4], [1, 4])
    antecedent1 = fv.FuzzyVariable('antecedent1', [low, high])

    low = fs.TriangularFS('low', [0, 1, 2], [0, 4])
    high = fs.TriangularFS('high', [1, 2, 4], [0, 4])
    antecedent2 = fv.FuzzyVariable('antecedent2', [low, high])
    
    low = fs.TriangularFS('low', [0, 1, 2], [0, 4])
    high = fs.TriangularFS('high', [1, 2, 4], [0, 4])
    consequent = fv.FuzzyVariable('consequent', [low, high])

    low_rule = rules.RuleSimple([0, 1], 0) # low, high -> low
    high_rule = rules.RuleSimple([1, 1], 1) # high, high -> high

    rb = rules.RuleBaseRegT1([antecedent1, antecedent2], [low_rule, high_rule], consequent)
    x = np.array([2.5, 1.75])
    antecedents_memberships = rb.compute_antecedents_memberships(x)

    assert (rb.compute_cut_heights(antecedents_memberships) == [0.5, 1/3]).all()

# TODO: finish tests
def test_forward_and_inference_return_same_output_in_RuleBaseRegT1():
    pass