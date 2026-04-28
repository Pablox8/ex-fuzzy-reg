# ex-fuzzy-reg

Interpretable regression using Type-1 Mamdani fuzzy inference systems, built on top of [ex-fuzzy](https://github.com/Fuminides/ex-fuzzy). Supports both rule extraction from data and evolutionary rule optimization, and integrates with scikit-learn as a drop-in estimator.

Built as part of a research collaboration at UPNA.

> Warning: this project is a work in progress. Documentation, demos and API are not stable. ⚠️


## Project Structure

```
ex-fuzzy-reg/
├── demos/
│   ├── datasets/
│   │   ├── advertising.csv
│   │   └── salary_data.csv
│   ├── genetic.ipynb                   # Evolutionary optimization
│   ├── inference.ipynb                 # Manual rule base construction
│   ├── mamdani.ipynb                   # MamdaniFIS quickstart
│   └── rb_coding.py
├── ex_fuzzy_reg/
│   ├── __init__.py
│   ├── evolutionary_backends_reg.py    # Backend abstraction (pymoo, EvoX)
│   ├── evolutionary_fit_reg.py         # BaseFuzzyRulesRegressor, FitRuleBaseReg
│   ├── fuzzy_sets.py                   # FS base class, Trapezoidal, Triangular, Gaussian
│   ├── fuzzy_variable.py               # FuzzyVariable container
│   ├── regressors.py                   # MamdaniFIS (sklearn estimator), load_from_json
│   ├── rules_reg.py                    # RuleBaseRegT1, TSK rule base
│   └── rules_reg_utils.py              # Partition generators, rule extraction
├── tests/
│   ├── test_fuzzy_sets.py
│   ├── test_fuzzy_variables.py
│   ├── test_rules_reg.py
│   └── test_rules_reg_utils.py
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```


## Installation

```bash
git clone https://github.com/Pablox8/ex-fuzzy-reg
cd ex-fuzzy-reg
pip install -e .
```

Optional extras:

```bash
pip install -e .[validate]   # FuzzyVariable.validate() — requires scipy
pip install -e .[gpu]        # GPU-accelerated optimization via EvoX
pip install -e .[dev]        # Development dependencies and test runner
```


## Quickstart

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from ex_fuzzy_reg import MamdaniFIS, FUZZY_SETS

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MamdaniFIS(fuzzy_type=FUZZY_SETS.t1, n_labels=3, n_rules=20)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
```

Save and reload a trained model:

```python
model.export_to_json("model.json")

from ex_fuzzy_reg import load_from_json
model = load_from_json("model.json")
```

## Core Concepts

**Fuzzy sets** define linguistic terms (e.g. *low*, *medium*, *high*) for each variable through a membership function. Three shapes are supported:

```python
from ex_fuzzy_reg import TriangularFS, TrapezoidalFS, GaussianFS

low  = TriangularFS("low",  [0.0, 0.0, 0.5], domain=[0, 1])
mid  = TriangularFS("mid",  [0.0, 0.5, 1.0], domain=[0, 1])
high = TriangularFS("high", [0.5, 1.0, 1.0], domain=[0, 1])
```

**Fuzzy variables** group a set of fuzzy sets for one input or output:

```python
from ex_fuzzy_reg import FuzzyVariable

temperature = FuzzyVariable("temperature", [low, mid, high], units="°C")
```

**Rule bases** hold the IF-THEN rules and perform inference. `RuleBaseRegT1` is the main T1 Mamdani rule base; it can be built automatically from data or constructed by hand:

```python
import numpy as np
from ex_fuzzy_reg import generate_triangular_partitions, generate_rules

# data is (n_samples, n_features + 1) — last column is the target
data = np.hstack((X_train, y_train.reshape(-1, 1)))
partitions = generate_triangular_partitions(data, n_labels=3)
rule_base  = generate_rules(data, partitions, n_rules=10)
```

**Regressors** are the top-level estimators. `MamdaniFIS` extracts rules directly from training data. `BaseFuzzyRulesRegressor` uses an evolutionary algorithm to optimise the rule base:

```python
from ex_fuzzy_reg import BaseFuzzyRulesRegressor, FUZZY_SETS

model = BaseFuzzyRulesRegressor(
    n_rules=20,
    n_ants=4,
    fuzzy_type=FUZZY_SETS.t1,
    fuzzy_set_type='trapezoidal',
    n_linguistic_variables=3,
)
model.fit(X_train, y_train, n_gen=50, pop_size=30)
```

## Demos

| Notebook | Description |
|---|---|
| `demos/mamdani.ipynb` | Fit and evaluate a `MamdaniFIS` on a real dataset |
| `demos/genetic.ipynb` | Optimise a rule base with `BaseFuzzyRulesRegressor` |
| `demos/inference.ipynb` | Build a rule base manually and run inference step by step |


## Requirements

- Python >= 3.10
- See `pyproject.toml` for the full dependency list, or `requirements.txt` for pinned floor versions


## License

This project is licensed under the **AGPL v3 License**, see the [LICENSE](LICENSE) for details.