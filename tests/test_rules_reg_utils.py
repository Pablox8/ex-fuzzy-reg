import pytest
import numpy as np
import sys
import os

# Add the library path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ex_fuzzy_reg import rules_reg_utils as utils
from ex_fuzzy_reg import fuzzy_sets as fs
from ex_fuzzy_reg import fuzzy_variable as fv


def test_generate_triangular_partitions_returns_one_variable_per_column():
    data = np.random.rand(10, 4)
    partitions = utils.generate_triangular_partitions(data)

    assert len(partitions) == 4


def test_generate_triangular_partitions_uses_default_variable_names_when_not_provided():
    data = np.array([[0, 1]])

    partitions = utils.generate_triangular_partitions(data)

    assert partitions[0].name == "Label 0"
    assert partitions[1].name == "Label 1"


def test_generate_triangular_partitions_sets_domain_from_data_min_and_max():
    data = np.array([[1, 10], [3, 20]])

    partitions = utils.generate_triangular_partitions(data)

    lv = partitions[0].linguistic_variables[0]
    assert lv.domain == [1, 3]


def test_generate_triangular_partitions_creates_n_labels_per_variable():
    data = np.array([[1, 40],
                     [1.8, 75],
                     [1.6, 150],
                     [2.2, 85]])

    n_labels = 5
    partitions = utils.generate_triangular_partitions(data, n_labels)

    assert len(partitions[0].linguistic_variables) == n_labels
    assert len(partitions[1].linguistic_variables) == n_labels


def test_generate_triangular_partitions_assigns_linguistic_label_names_and_variable_names():
    fv_label_names = ['height', 'weight']
    fs_label_names = ['low', 'medium', 'high']

    data = np.array([[1, 40],
                     [1.8, 75],
                     [1.6, 150],
                     [2.2, 85]])

    partitions = utils.generate_triangular_partitions(data, n_labels=3, fs_label_names=fs_label_names, fv_label_names=fv_label_names)

    assert partitions[0].name == 'height'
    assert partitions[1].name == 'weight'

    for i in range(3):
        assert partitions[0].linguistic_variables[i].name == fs_label_names[i]
        assert partitions[1].linguistic_variables[i].name == fs_label_names[i]


def test_generate_triangular_partitions_handles_constant_column():
    data = np.array([[5, 10],
                     [5, 20],
                     [5, 30]])

    partitions = utils.generate_triangular_partitions(data)

    for fs in partitions[0].linguistic_variables:
        assert fs.membership_parameters == [5, 5, 5]


def test_generate_triangular_partitions_computes_correct_membership_parameters():
    data = np.array([[1, 40],
                     [1.8, 75],
                     [1.6, 150],
                     [2.2, 85]])

    partitions = utils.generate_triangular_partitions(data, 3)

    assert partitions[0].linguistic_variables[0].membership_parameters == [1, 1, 1.6]
    assert partitions[0].linguistic_variables[1].membership_parameters == [1, 1.6, 2.2]
    assert partitions[0].linguistic_variables[2].membership_parameters == [1.6, 2.2, 2.2]

    assert partitions[1].linguistic_variables[0].membership_parameters == [40, 40, 95]
    assert partitions[1].linguistic_variables[1].membership_parameters == [40, 95, 150]
    assert partitions[1].linguistic_variables[2].membership_parameters == [95, 150, 150]


def test_invalid_tolerance_is_ignored():
    data = np.array([
        [2.0, 3.0, 4.0],
        [8.0, 7.0, 9.0],
    ])

    low = fs.TrapezoidalFS('low', [0, 0, 2, 4], [0, 10])
    mid = fs.TriangularFS('mid', [3, 5, 7], [0, 10])
    high = fs.TrapezoidalFS('high', [6, 8, 10, 10], [0, 10])

    partitions = [
        fv.FuzzyVariable('x1', [low, mid, high], 'u'),
        fv.FuzzyVariable('x2', [low, mid, high], 'u'),
        fv.FuzzyVariable('y',  [low, mid, high], 'u'),
    ]

    rb_valid = utils.generate_rules(data, partitions=partitions)
    rb_invalid = utils.generate_rules(data, partitions=partitions, tolerance=-0.5)

    assert len(rb_valid.rules) == len(rb_invalid.rules)


def test_invalid_n_rules_is_ignored():
    data = np.array([
        [2.0, 3.0, 4.0],
        [8.0, 7.0, 9.0],
    ])

    low = fs.TrapezoidalFS('low', [0, 0, 2, 4], [0, 10])
    mid = fs.TriangularFS('mid', [3, 5, 7], [0, 10])
    high = fs.TrapezoidalFS('high', [6, 8, 10, 10], [0, 10])

    partitions = [
        fv.FuzzyVariable('x1', [low, mid, high], 'u'),
        fv.FuzzyVariable('x2', [low, mid, high], 'u'),
        fv.FuzzyVariable('y',  [low, mid, high], 'u'),
    ]

    rb_valid = utils.generate_rules(data, partitions=partitions)
    rb_invalid = utils.generate_rules(data, partitions=partitions, n_rules=-2)

    assert len(rb_valid.rules) == len(rb_invalid.rules)


def test_partitions_are_computed_when_not_passed():
    data = np.array([
        [2.0, 3.0, 4.0],
        [8.0, 7.0, 9.0],
    ])

    rule_base = utils.generate_rules(data)

    assert rule_base.antecedents is not None
    assert rule_base.consequent is not None

    assert len(rule_base.antecedents) == data.shape[1] - 1
    assert isinstance(rule_base.consequent, fv.FuzzyVariable)


def test_computed_and_passed_partitions_produce_same_result():
    data = np.array([
        [2.0, 3.0, 4.0],
        [8.0, 7.0, 9.0],
    ])

    rb_computed = utils.generate_rules(data)

    partitions = utils.generate_triangular_partitions(data)
    rb_passed = utils.generate_rules(data, partitions=partitions)

    assert len(rb_computed.rules) == len(rb_passed.rules)

    for r1, r2 in zip(rb_computed.rules, rb_passed.rules):
        assert r1.antecedents == r2.antecedents
        assert r1.consequent == r2.consequent


def test_rules_with_dof_below_tolerance_are_discarded():
    data = np.array([
        [2.0, 3.0, 4.0],
        [2.1, 3.1, 4.1],
        [8.0, 7.5, 9.0],
    ])

    low = fs.TrapezoidalFS('low', [0, 0, 2, 4], [0, 10])
    mid = fs.TriangularFS('mid', [3, 5, 7], [0, 10])
    high = fs.TrapezoidalFS('high', [6, 8, 10, 10], [0, 10])

    partitions = [
        fv.FuzzyVariable('x1', [low, mid, high], 'u'),
        fv.FuzzyVariable('x2', [low, mid, high], 'u'),
        fv.FuzzyVariable('y',  [low, mid, high], 'u'),
    ]

    rb_no_tol = utils.generate_rules(data, partitions=partitions)
    rb_high_tol = utils.generate_rules(data, partitions=partitions, tolerance=0.95)

    assert len(rb_high_tol.rules) <= len(rb_no_tol.rules)


def test_rule_with_higher_dof_is_kept_when_having_same_antecedents():
    data = np.array([
        [2.0, 2.0, 4.0],
        [2.1, 2.1, 8.0],
    ])

    low = fs.TrapezoidalFS('low', [0, 0, 2, 4], [0, 10])
    mid = fs.TriangularFS('mid', [3, 5, 7], [0, 10])
    high = fs.TrapezoidalFS('high', [6, 8, 10, 10], [0, 10])

    partitions = [
        fv.FuzzyVariable('x1', [low, mid, high], 'u'),
        fv.FuzzyVariable('x2', [low, mid, high], 'u'),
        fv.FuzzyVariable('y',  [low, mid, high], 'u'),
    ]

    rb = utils.generate_rules(data, partitions=partitions)

    assert len(rb.rules) == 1


def test_best_n_rules_are_kept_when_n_rules_is_passed():
    data = np.array([
        [2.0, 3.0, 4.0],
        [2.1, 3.1, 4.1],
        [8.0, 7.5, 9.0],
        [8.2, 7.7, 9.1],
    ])

    low = fs.TrapezoidalFS('low', [0, 0, 2, 4], [0, 10])
    mid = fs.TriangularFS('mid', [3, 5, 7], [0, 10])
    high = fs.TrapezoidalFS('high', [6, 8, 10, 10], [0, 10])

    partitions = [
        fv.FuzzyVariable('x1', [low, mid, high], 'u'),
        fv.FuzzyVariable('x2', [low, mid, high], 'u'),
        fv.FuzzyVariable('y',  [low, mid, high], 'u'),
    ]

    rb_all = utils.generate_rules(data, partitions=partitions)
    rb_top_1 = utils.generate_rules(data, partitions=partitions, n_rules=1)

    assert len(rb_top_1.rules) == 1
    assert len(rb_top_1.rules) <= len(rb_all.rules)
