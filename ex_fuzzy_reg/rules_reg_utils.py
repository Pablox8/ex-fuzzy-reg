import numpy as np
from ex_fuzzy.rules import RuleSimple

from ex_fuzzy_reg.fuzzy_sets import TriangularFS
from ex_fuzzy_reg import fuzzy_variable as fv
from ex_fuzzy_reg.rules_reg import RuleBaseRegT1


def generate_partitions(data: np.ndarray, n_labels: int=3, label_names: list[str]=None) -> list[fv.FuzzyVariable]:
    partitions = []
    
    for i in range(data.shape[1]):
        label_min = np.min(data[:, i])
        label_max = np.max(data[:, i])
        label_mid = (label_max + label_min) / (n_labels - 1)

        if label_names:
            low = TriangularFS('low ' + label_names[i], [label_min, label_min, label_mid], [label_min, label_max])
            medium = TriangularFS('medium ' + label_names[i], [label_min, label_mid, label_max], [label_min, label_max])
            high = TriangularFS('high ' + label_names[i], [label_mid, label_max, label_max], [label_min, label_max])

            fv_label = fv.FuzzyVariable(label_names[i], [low, medium, high])

        else:
            low = TriangularFS('low', [label_min, label_min, label_mid], [label_min, label_max])
            medium = TriangularFS('medium', [label_min, label_mid, label_max], [label_min, label_max])
            high = TriangularFS('high', [label_mid, label_max, label_max], [label_min, label_max])

            fv_label = fv.FuzzyVariable(f'Label {i}', [low, medium, high])

        partitions.append(fv_label)

    return partitions


def generate_rules(data: np.ndarray) -> RuleBaseRegT1:
    m = data.shape[0]
    partitions = generate_partitions(data)

    rules = {} # antecendents (tuple): consequent (int), dof (float)

    for i in range(m):
        memberships = np.array([partitions[j].compute_memberships(data[i, j]) for j in range(data.shape[1])])
        labels = np.argmax(memberships, axis=1)

        dof = np.prod(memberships.max(axis=1)) # degree of fulfillment

        antecedents = tuple(labels[:-1])
        consequent = labels[-1]

        if antecedents not in rules:
            rules[antecedents] = (consequent, dof)
        else:
            last_DOF = rules[antecedents][1]

            if last_DOF < dof: 
                rules[antecedents] = (consequent, dof)

    rules_list = []
    for antecedents, (consequent, dof) in rules.items():
        rules_list.append(RuleSimple(list(antecedents), consequent))

    return RuleBaseRegT1(partitions[:-1], rules_list, partitions[-1])