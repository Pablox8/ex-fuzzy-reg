import numpy as np
from ex_fuzzy.rules import RuleSimple

from ex_fuzzy_reg.fuzzy_sets import TriangularFS
from ex_fuzzy_reg import fuzzy_variable as fv
from ex_fuzzy_reg.rules_reg import RuleBaseRegT1


def generate_triangular_partitions(data: np.ndarray, n_labels: int=3, fs_label_names: list[str]=None, fv_label_names: list[str]=None) -> list[fv.FuzzyVariable]:
    if n_labels < 3:
        raise ValueError("n_labels must be greater or equal to 3.")
    
    if fs_label_names and len(fs_label_names) != n_labels:
        raise ValueError("fs_label_names length must match n_labels")

    if fv_label_names and len(fv_label_names) != data.shape[1]:
        raise ValueError("fv_label_names length must match number of features")
   
    partitions = []

    for i in range(data.shape[1]):
        label_min = np.min(data[:, i])
        label_max = np.max(data[:, i])
    
        centers = np.linspace(label_min, label_max, n_labels)

        fs_list = []

        for j in range(n_labels):
            if j == 0:
                params = [centers[j], centers[j], centers[j+1]]
            elif j == n_labels - 1:
                params = [centers[j-1], centers[j], centers[j]]
            else:
                params = [centers[j-1], centers[j], centers[j+1]]
            
            fs_name = fs_label_names[j] if fs_label_names else f"Value {j}"

            fs_list.append(TriangularFS(fs_name, params, [label_min, label_max])) 
        
        fv_label = fv_label_names[i] if fv_label_names else f"Label {i}"
        partitions.append(fv.FuzzyVariable(fv_label, fs_list))
    
    return partitions


def generate_rules(data: np.ndarray, partitions: list[fv.FuzzyVariable]=None, tolerance: float=None, n_rules: int=None) -> RuleBaseRegT1:
    if tolerance and (tolerance < 0 or tolerance > 1):
        tolerance = None
    
    if n_rules and n_rules < 0:
        n_rules = None

    m = data.shape[0]
    if not partitions:
        partitions = generate_triangular_partitions(data)

    rules = {} # antecendents (tuple): consequent (int), dof (float)

    for i in range(m):
        memberships = np.array([partitions[j].compute_memberships(data[i, j]) for j in range(data.shape[1])])
        labels = np.argmax(memberships, axis=1)

        dof = np.prod(memberships.max(axis=1)) # degree of fulfillment

        if tolerance and dof < tolerance:
            continue

        antecedents = tuple(labels[:-1])
        consequent = labels[-1]

        if antecedents not in rules:
            rules[antecedents] = (consequent, dof)
        else:
            last_DOF = rules[antecedents][1]

            if last_DOF < dof: 
                rules[antecedents] = (consequent, dof)
    
    rules_list = []
    dofs = []
    
    for antecedents, (consequent, dof) in rules.items():
        rules_list.append(RuleSimple(list(antecedents), consequent))
        dofs.append(dof)
    
    if not n_rules:
        return RuleBaseRegT1(partitions[:-1], rules_list, partitions[-1])
     
    dofs = np.array(dofs)
    rules_list = np.array(rules_list, dtype=RuleSimple) 
    valid_rules = np.argsort(dofs)[::-1][:n_rules]

    return RuleBaseRegT1(partitions[:-1], rules_list[valid_rules], partitions[-1])