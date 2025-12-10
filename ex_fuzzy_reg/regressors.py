import numpy as np
from sklearn.base import RegressorMixin

from ex_fuzzy_reg import fuzzy_sets as fs
from ex_fuzzy_reg import fuzzy_variable as fv
from ex_fuzzy_reg import rules_reg_utils as utils


class MamdamiFIS(RegressorMixin):
    def __init__(self, fuzzy_type: fs.FUZZY_SETS, linguistic_variables: list[fv.FuzzyVariable]=None, n_rules: int=30, tolerance: float=0.0) -> None:
        self.fuzzy_type = fuzzy_type
        self.linguistic_variables = linguistic_variables

    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        data = np.hstack((X, y))

        if not self.linguistic_variables:
            self.linguistic_variables = utils.generate_partitions(data) # TODO: generalize with more parameters (n_labels, label_names)
      
        self.rule_base = utils.generate_rules(data, self.linguistic_variables)


    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.rule_base.inference(X)
        return y_pred
