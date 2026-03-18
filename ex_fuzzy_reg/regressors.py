import json
import numpy as np
from sklearn.base import RegressorMixin

from ex_fuzzy_reg import fuzzy_sets as fs
from ex_fuzzy_reg import fuzzy_variable as fv
from ex_fuzzy_reg import rules_reg_utils as utils


class MamdamiFIS(RegressorMixin):
    def __init__(self, fuzzy_type: fs.FUZZY_SETS, linguistic_variables: list[fv.FuzzyVariable]=None, n_rules: int=30, n_labels: int=3, tolerance: float=0.0) -> None:
        self.fuzzy_type = fuzzy_type
        self.linguistic_variables = linguistic_variables
        self.n_labels = n_labels

    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        data = np.hstack((X, y))

        if not self.linguistic_variables:
            # TODO: extract label_names and fuzzy variable names from X and y
            self.linguistic_variables = utils.generate_triangular_partitions(data, self.n_labels) 
      
        self.rule_base = utils.generate_rules(data, self.linguistic_variables)


    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.rule_base.inference(X)
        return y_pred

    # TODO: load model from json
    def export_to_json(self, path: str = "./model.json") -> None:
        linguistic_vars_str = {}

        for var in self.linguistic_variables:
            linguistic_vars_str[var.name] = {}
            for fs in var.linguistic_variables:
                linguistic_vars_str[var.name][fs.name] = fs.membership_parameters

        model_data = {
            "model": __class__.__name__,
            "linguistic_variables": linguistic_vars_str,
            "rules": self.rule_base.get_rulebase_matrix().astype(int).tolist()
        }

        with open(path, mode="w", encoding="utf-8") as write_file:
            json.dump(model_data, write_file, indent=4)