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
