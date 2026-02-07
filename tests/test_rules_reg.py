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