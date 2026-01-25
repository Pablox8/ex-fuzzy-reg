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


def test_rulebase_tsk():
    '''
    Implementation of Example 2 from 'Fuzzy Identification of Systems and Its Applications to Modeling and Control' article, 1985 
    '''
    x1_small = fs.TriangularFS('x1 small', [0, 0, 16], [0, 20])
    x1_big = fs.TriangularFS('x1 big', [10, 20, 20], [0, 20])
    
    x2_small = fs.TriangularFS('x2 small', [0, 0, 8], [0, 10])
    x2_big = fs.TriangularFS('x2 big', [2, 10, 10], [0, 10])

    X1 = fv.FuzzyVariable('x1', [x1_small, x1_big])
    X2 = fv.FuzzyVariable('x2', [x2_small, x2_big])

    R1_consq = rules.ConsequentTSK([1, 1])
    R2_consq = rules.ConsequentTSK([2, 0])
    R3_consq = rules.ConsequentTSK([0, 3])

    R1 = rules.RuleSimpleTSK([0, 0], R1_consq)
    R2 = rules.RuleSimpleTSK([1, -1], R2_consq)
    R3 = rules.RuleSimpleTSK([-1, 1], R3_consq)

    RB = rules.RuleBaseRegTSK([X1, X2], [R1, R2, R3])

    x = np.array([[12, 5]])

    print(round(RB.inference(x)[0], 1))


if __name__ == '__main__':
    #test_partition_creation_with_exfuzzy()
    #test_generate_rules()
    test_rulebase_tsk()