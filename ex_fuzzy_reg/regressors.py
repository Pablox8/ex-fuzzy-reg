import json
import numpy as np
from sklearn.base import RegressorMixin

from ex_fuzzy.rules import RuleSimple

from ex_fuzzy_reg import fuzzy_sets as fs
from ex_fuzzy_reg import fuzzy_variable as fv
from ex_fuzzy_reg import rules_reg_utils as utils
from ex_fuzzy_reg.rules_reg import RuleBaseRegT1


class MamdaniFIS(RegressorMixin):
    def __init__(self, fuzzy_type: fs.FUZZY_SETS, linguistic_variables_type: str='triangular', linguistic_variables: list[fv.FuzzyVariable]=None, n_rules: int=30, n_labels: int=3, tolerance: float=0.0) -> None:
        self.fuzzy_type = fuzzy_type
        self.linguistic_variables = linguistic_variables
        self.linguistic_variables_type = linguistic_variables_type
        self.n_labels = n_labels
        self.n_rules = n_rules
        self.tolerance = tolerance

    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y = y.reshape(-1, 1)
        data = np.hstack((X, y))

        if not self.linguistic_variables:
            # TODO: extract label_names and fuzzy variable names from X and y
            self.linguistic_variables = utils.generate_triangular_partitions(data, self.n_labels) 
      
        self.rule_base = utils.generate_rules(data, self.linguistic_variables, self.n_rules, self.tolerance)


    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.rule_base.inference(X)
        return y_pred


    def export_to_json(self, path: str = "./model.json") -> None:
        linguistic_vars_str = {}

        for var in self.linguistic_variables:
            linguistic_vars_str[var.name] = {}
            for fs in var.linguistic_variables:
                linguistic_vars_str[var.name][fs.name] = fs.membership_parameters
        
        rules = self.rule_base.get_rules()
        rules_learned = []

        for rule in rules:
            rules_learned.append([rule.antecedents, rule.consequent])

        model_data = {
            "model": __class__.__name__,
            "fuzzy_type": self.fuzzy_type.name,
            "linguistic_variables_type": self.linguistic_variables_type,
            "linguistic_variables": linguistic_vars_str,
            "rules_learned": rules_learned
        }

        with open(path, mode="w", encoding="utf-8") as model_json:
            json.dump(model_data, model_json, indent=4)
    

regressors = {"MamdaniFIS": MamdaniFIS}

def load_from_json(path: str = "./model.json"):
    with open(path, mode="r", encoding="utf-8") as model_json:
        model_json = json.load(model_json)
    
    model_type = model_json['model']
    fuzzy_type = fs.FUZZY_SETS[model_json['fuzzy_type']]
    linguistic_variables_type = model_json['linguistic_variables_type']
    lvs = model_json['linguistic_variables']
    rules_learned = model_json['rules_learned']

    linguistic_variables = []
    for lv_name, lv_fuzzy_sets in lvs.items():
        fuzzy_sets = []
        for fs_name, membership_parameters in lv_fuzzy_sets.items():
            if linguistic_variables_type == 'triangular':
                fuzzy_sets.append(fs.TriangularFS(fs_name, membership_parameters, [-np.inf, np.inf])) # TODO: domain

        lv = fv.FuzzyVariable(name=lv_name, fuzzy_sets=fuzzy_sets)
        linguistic_variables.append(lv)

    loaded_rules = []
    for rule in rules_learned:
        antecedents = rule[0]
        consequent = rule[1]
        loaded_rules.append(RuleSimple(antecedents, consequent))
    
    antecedents = linguistic_variables[:-1]
    consequent = linguistic_variables[-1]
    loaded_rule_base = RuleBaseRegT1(antecedents, loaded_rules, consequent)

    model = regressors[model_type] 
    
    fitted_model = model(fuzzy_type=fuzzy_type, linguistic_variables_type=linguistic_variables_type, linguistic_variables=linguistic_variables, n_rules=len(loaded_rules))
    fitted_model.rule_base = loaded_rule_base

    return fitted_model