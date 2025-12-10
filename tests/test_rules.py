import pytest
import numpy as np
import sys
import os

# Add the library path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ex_fuzzy_reg import rules_reg_utils as utils
from ex_fuzzy.utils import construct_partitions, _triangular_construct_partitions

def test_partition_creation() -> None:
    label_names = ['height', 'weight']

    data = np.array([[1, 40],
                     [1.8, 75],
                     [1.6, 150],
                     [2.2, 85]])
    
    n_labels = 3
    partitions = utils.generate_partitions(data, n_labels, label_names)

    lv_height = partitions[0].linguistic_variables
    lv_weight = partitions[1].linguistic_variables

    assert lv_height[0].membership_parameters == [1, 1, 1.6]
    assert lv_height[1].membership_parameters == [1, 1.6, 2.2]
    assert lv_height[2].membership_parameters == [1.6, 2.2, 2.2]

    assert lv_weight[0].membership_parameters == [40, 40, 95]
    assert lv_weight[1].membership_parameters == [40, 95, 150]
    assert lv_weight[2].membership_parameters == [95, 150, 150]

# TODO: complete tests below
def test_partition_creation_with_exfuzzy() -> None:
    label_names = ['height', 'weight']

    data = np.array([[1, 40],
                     [1.8, 75],
                     [1.6, 150],
                     [2.2, 85]])
    
    n_labels = 3
    # partitions = construct_partitions(data, n_partitions=n_labels, shape='triangular')
    partitions = _triangular_construct_partitions(data, n_partitions=n_labels)
    # partitions = utils.generate_partitions(data, n_labels, label_names)

    lv_height = partitions[0].linguistic_variables
    lv_weight = partitions[1].linguistic_variables

    print(lv_height)
    print(lv_weight)

    assert lv_height[0].membership_parameters == [0, 0, 1.6]
    assert lv_height[1].membership_parameters == [0, 1.6, 2.2]
    assert lv_height[2].membership_parameters == [1.6, 2.2, 2.2]

    assert lv_weight[0].membership_parameters == [0, 0, 95]
    assert lv_weight[1].membership_parameters == [0, 95, 150]
    assert lv_weight[2].membership_parameters == [95, 150, 150]


def test_generate_rules():
    data = np.array([[1, 40, 2],
                     [1.8, 75, 6.5],
                     [1.6, 150, 4],
                     [2.2, 85, 9]])
    print('height | weight | skill')
    print(data, '\n')
    label_names = ['height', 'weight', 'skill']
    partitions = utils.generate_partitions(data, fv_label_names=label_names)

    for i in range(len(partitions)):
        print(partitions[i])
    print()

    RB = utils.generate_rules(data)

    RB.print_rules()
    print(RB.get_rulebase_matrix())

if __name__ == '__main__':
    #test_partition_creation_with_exfuzzy()
    test_generate_rules()