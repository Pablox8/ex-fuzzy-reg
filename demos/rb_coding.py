import os
import sys
import numpy as np

# Add the library path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ex_fuzzy_reg import fuzzy_sets as fs
from ex_fuzzy_reg import fuzzy_variable as fv
from ex_fuzzy_reg import rules_reg as rules
from ex_fuzzy_reg import evolutionary_fit_reg as ef


def main():
    x1_low = fs.TriangularFS('x1_low', [0, 0, 2], [0, 10])
    x1_mid = fs.TriangularFS('x1_mid', [2, 5, 7], [0, 10])
    x1_high = fs.TriangularFS('x1_high', [6, 10, 10], [0, 10])
    x1 = fv.FuzzyVariable('x1', [x1_low, x1_mid, x1_high])

    x2_low = fs.TriangularFS('x2_low', [0, 0, 2], [0, 10])
    x2_mid = fs.TriangularFS('x2_mid', [2, 5, 7], [0, 10])
    x2_high = fs.TriangularFS('x2_high', [6, 10, 10], [0, 10])
    x2 = fv.FuzzyVariable('x2', [x2_low, x2_mid, x2_high])

    x3_low = fs.TriangularFS('x3_low', [0, 0, 2], [0, 10])
    x3_mid = fs.TriangularFS('x3_mid', [2, 5, 7], [0, 10])
    x3_high = fs.TriangularFS('x3_high', [6, 10, 10], [0, 10])
    x3 = fv.FuzzyVariable('x3', [x3_low, x3_mid, x3_high])

    y1_low = fs.TriangularFS('y1_low', [0, 0, 2], [0, 10])
    y1_mid = fs.TriangularFS('y1_mid', [2, 5, 7], [0, 10])
    y1_high = fs.TriangularFS('y1_high', [6, 10, 10], [0, 10])
    y1 = fv.FuzzyVariable('y1', [y1_low, y1_mid, y1_high])

    r1 = rules.RuleSimple([1, -1, -1], 0)
    r2 = rules.RuleSimple([2, -1, 0], 2)

    rb = rules.RuleBaseRegT1([x1, x2, x3], [r1, r2], y1)

    frb = ef.FitRuleBaseReg(np.array([[2, 5, 7]]), np.array([2.5]), 2, 3, antecedents=[x1, x2, x3], consequent=y1)

    #x = frb.encode_rulebase(rule_base=rb, optimize_lv=True)
    #rb_d = frb._construct_ruleBase(x, optimize_lv=True)
    x = frb.encode_rulebase(rule_base=rb)
    rb_d = frb._construct_ruleBase(x)

    print(x)
    print(len(x))
    print()
    rb_d.print_rules()


if __name__ == '__main__':
    main()
