# Public API of ex-fuzzy-reg
#
# Users should import from here, not from internal modules:
#   from ex_fuzzy_reg import MamdaniFIS, BaseFuzzyRulesRegressor
#   from ex_fuzzy_reg import FuzzyVariable, TriangularFS, FUZZY_SETS

# Regressors (main entry points)
from .regressors import MamdaniFIS, load_from_json
from .evolutionary_fit_reg import BaseFuzzyRulesRegressor

# Fuzzy sets
from .fuzzy_sets import (
    FUZZY_SETS,
    FS,
    TrapezoidalFS,
    TriangularFS,
    GaussianFS,
)

# Fuzzy variables
from .fuzzy_variable import FuzzyVariable

# Rule bases
from .rules_reg import RuleBaseRegT1

# Partition and rule generation utilities
from .rules_reg_utils import (
    generate_triangular_partitions,
    generate_trapezoidal_partitions,
    generate_rules,
)

__all__ = [
    # Regressors
    "MamdaniFIS",
    "BaseFuzzyRulesRegressor",
    "load_from_json",

    # Fuzzy sets
    "FUZZY_SETS",
    "FS",
    "TrapezoidalFS",
    "TriangularFS",
    "GaussianFS",

    # Fuzzy variables
    "FuzzyVariable",

    # Rule bases
    "RuleBaseRegT1",

    # Utilities
    "generate_triangular_partitions",
    "generate_trapezoidal_partitions",
    "generate_rules",
]