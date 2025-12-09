import numpy as np
import sys
import os

# Add the library path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ex_fuzzy_reg import fuzzy_sets as fs
from ex_fuzzy_reg import fuzzy_variable as fv
from ex_fuzzy_reg import rules_reg as r


def rule_base_t1_demo():
    bad  = fs.TriangularFS('Bad', [0, 0, 5], [0, 10])
    medium = fs.TrapezoidalFS('Medium', [2.5, 4, 6, 7], [0, 10])
    good = fs.TriangularFS('Good', [6, 10, 10], [0, 10])
    service = fv.FuzzyVariable('Service', [bad, medium, good])

    poor  = fs.TriangularFS('Poor', [0, 0, 4], [0, 10])
    normal = fs.TrapezoidalFS('Normal', [2.5, 4, 5, 7], [0, 10])
    good_food = fs.TriangularFS('Good', [6.5, 10, 10], [0, 10])
    food = fv.FuzzyVariable('Food', [poor, normal, good_food])

    low = fs.TriangularFS('Low', [0, 0, 11], [0, 20])
    medium_tip = fs.TriangularFS('Medium', [10, 12, 14], [0, 20])
    high = fs.TriangularFS('High', [13.5, 20, 20], [0, 20])
    tip = fv.FuzzyVariable('Tip', [low, medium_tip, high])

    R1 = r.RuleSimple([0, 0], 0)  # Bad, Poor -> Low
    R2 = r.RuleSimple([0, 1], 0)  # Bad, Normal -> Low
    R3 = r.RuleSimple([1, 0], 1)  # Medium, Poor -> Medium
    R4 = r.RuleSimple([2, 2], 2)  # Good, Good -> High
    R5 = r.RuleSimple([1, 1], 1)  # Medium, Medium -> Medium

    RB = r.RuleBaseRegT1(
        antecedents=[service, food],
        rules=[R1, R2, R3, R4, R5],
        consequent=tip
    )

    RB.print_rules()
    print(RB.get_rulebase_matrix(), '\n')

    x = np.array([
        [1, 3],
        [4, 2],
        [9, 10],
        [5, 6],
        [10, 10],
        [7, 7]
    ])

    y = RB.inference(x)
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"[{i}] Service: {xi[0]}, Food: {xi[1]} -> Tip: {yi}")


if __name__ == '__main__':
    rule_base_t1_demo()
