import numpy as np
from ex_fuzzy.rules import RuleSimple

from ex_fuzzy_reg.fuzzy_sets import TriangularFS
from ex_fuzzy_reg import fuzzy_variable as fv
from ex_fuzzy_reg.rules_reg import RuleBaseRegT1

# TODO: generalize for any n_labels
def generate_partitions(data: np.ndarray, n_labels: int=3, fv_label_names: list[str]=None) -> list[fv.FuzzyVariable]:
    partitions = []
    
    for i in range(data.shape[1]):
        label_min = np.min(data[:, i])
        label_max = np.max(data[:, i])
        label_mid = (label_max + label_min) / (n_labels - 1)

        if fv_label_names:
            low = TriangularFS('low ' + fv_label_names[i], [label_min, label_min, label_mid], [label_min, label_max])
            medium = TriangularFS('medium ' + fv_label_names[i], [label_min, label_mid, label_max], [label_min, label_max])
            high = TriangularFS('high ' + fv_label_names[i], [label_mid, label_max, label_max], [label_min, label_max])

            fv_label = fv.FuzzyVariable(fv_label_names[i], [low, medium, high])

        else:
            low = TriangularFS('low', [label_min, label_min, label_mid], [label_min, label_max])
            medium = TriangularFS('medium', [label_min, label_mid, label_max], [label_min, label_max])
            high = TriangularFS('high', [label_mid, label_max, label_max], [label_min, label_max])

            fv_label = fv.FuzzyVariable(f'Label {i}', [low, medium, high])

        partitions.append(fv_label)

    return partitions


def generate_rules(data: np.ndarray, partitions: list[fv.FuzzyVariable]=None, tolerance: float=None, n_rules: int=None) -> RuleBaseRegT1:
    if tolerance and (tolerance < 0 or tolerance > 1):
        tolerance = None
    
    if n_rules and n_rules < 0:
        n_rules = None

    m = data.shape[0]
    if not partitions:
        partitions = generate_partitions(data)

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