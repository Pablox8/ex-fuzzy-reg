"""
Utility functions for generating fuzzy partitions and rule bases for regression.

Functions:
    generate_triangular_partitions: Builds triangular fuzzy partitions from data.
    generate_trapezoidal_partitions: Builds trapezoidal fuzzy partitions from data.
    generate_rules: Generates a RuleBaseRegT1 from data using a Wang-Mendel-like approach.
"""

import numpy as np
from ex_fuzzy.rules import RuleSimple

from ex_fuzzy_reg.fuzzy_sets import TriangularFS, TrapezoidalFS
from ex_fuzzy_reg import fuzzy_variable as fv
from ex_fuzzy_reg.rules_reg import RuleBaseRegT1


def generate_triangular_partitions(data: np.ndarray, n_labels: int=3, fs_label_names: list[str]=None, fv_label_names: list[str]=None) -> list[fv.FuzzyVariable]:
    """
    Generate triangular fuzzy partitions for each feature column in data.

    Args:
        data (np.ndarray): Input array of shape (n_samples, n_features).
            The last column is treated as the target variable.
        n_labels (int): Number of fuzzy sets (linguistic labels) per variable. Must be >= 3.
        fs_label_names (list[str], optional): Names for the fuzzy sets.
            Length must equal n_labels. Defaults to "Value 0", "Value 1", ...
        fv_label_names (list[str], optional): Names for the fuzzy variables.
            Length must equal data.shape[1]. Defaults to "Label 0", "Label 1", ...

    Returns:
        list[FuzzyVariable]: One FuzzyVariable per column in data, each containing
            n_labels triangular fuzzy sets.

    Raises:
        ValueError: If n_labels < 3, or if the length of fs_label_names or
            fv_label_names does not match the expected dimensions.
    """
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


def generate_trapezoidal_partitions(data: np.ndarray, n_labels: int=3, fs_label_names: list[str]=None, fv_label_names: list[str]=None) -> list[fv.FuzzyVariable]:
    """
    Generate trapezoidal fuzzy partitions for each feature column in data.

    Uses quantile-based breakpoints to build the trapezoidal parameters, so the
    partitions adapt to the data distribution. 

    Args:
        data (np.ndarray): Input array of shape (n_samples, n_features).
            The last column is treated as the target variable.
        n_labels (int): Number of fuzzy sets (linguistic labels) per variable. Must be >= 3.
        fs_label_names (list[str], optional): Names for the fuzzy sets.
            Length must equal n_labels. Defaults to "Value 0", "Value 1", ...
        fv_label_names (list[str], optional): Names for the fuzzy variables.
            Length must equal data.shape[1]. Defaults to "Label 0", "Label 1", ...

    Returns:
        list[FuzzyVariable]: One FuzzyVariable per column in data, each containing
            n_labels trapezoidal fuzzy sets.

    Raises:
        ValueError: If n_labels < 3, or if the length of fs_label_names or
            fv_label_names does not match the expected dimensions.
    """
    if n_labels < 3:
        raise ValueError("n_labels must be greater or equal to 3.")
    
    if fs_label_names and len(fs_label_names) != n_labels:
        raise ValueError("fs_label_names length must match n_labels")

    if fv_label_names and len(fv_label_names) != data.shape[1]:
        raise ValueError("fv_label_names length must match number of features")

    # ex_fuzzy/utils.py, compute_quantiles
    quantiles_space = np.linspace(0, 100, n_labels)
    quantiles = np.nanpercentile(data, quantiles_space, axis=0)

    partitions = []

    for i in range(data.shape[1]):
        label_min = np.min(data[:, i])
        label_max = np.max(data[:, i])

        fs_list = []

        for j in range(n_labels):
            if j == 0:
                params = [label_min, label_min, quantiles[j + 1, i], quantiles[j + 2, i]]
            elif j == n_labels - 1:
                params = [quantiles[j - 2, i], quantiles[j - 1, i], label_max, label_max]
            else:
                left  = quantiles[j - 1, i]
                mid_left = quantiles[j, i]
                mid_right = quantiles[j + 1, i] if j + 1 < n_labels else mid_left
                right = quantiles[j + 2, i] if j + 2 < n_labels else label_max
                params = [left, mid_left, mid_right, right]
            
            fs_name = fs_label_names[j] if fs_label_names else f"Value {j}"

            fs_list.append(TrapezoidalFS(fs_name, params, [label_min, label_max])) 
        
        fv_label = fv_label_names[i] if fv_label_names else f"Label {i}"
        partitions.append(fv.FuzzyVariable(fv_label, fs_list))

    return partitions


def generate_rules(data: np.ndarray, partitions: list[fv.FuzzyVariable]=None, n_rules: int=0, tolerance: float=0.0) -> RuleBaseRegT1:
    """
    Generate a fuzzy rule base from data using a Wang-Mendel-like heuristic.

    For each sample, the antecedents are the fuzzy labels with the highest membership
    for the input features, and the consequent is the label with the highest membership
    for the target column. When two samples produce the same antecedent combination,
    only the rule with the highest degree of fulfilment (DoF) is kept.

    Args:
        data (np.ndarray): Input array of shape (n_samples, n_features + 1).
            The last column is the target variable.
        partitions (list[FuzzyVariable], optional): Pre-built fuzzy partitions for all
            columns (including the target). If None, triangular partitions are generated
            automatically from data using the default n_labels=3.
        n_rules (int): Maximum number of rules to return (sorted by DoF, descending).
            Pass 0 (default) to return all discovered rules.
        tolerance (float): Minimum DoF threshold. Samples with DoF below this value
            are ignored. Must be in [0, 1]. Defaults to 0.0 (no filtering).

    Returns:
        RuleBaseRegT1: A rule base whose antecedents are the input fuzzy variables
            (all columns except the last) and whose consequent is the target fuzzy
            variable (last column).
    """
    if tolerance < 0.0 or tolerance > 1.0:
        print("Warning: tolerance not in range [0, 1], setting tolerance=0.0.")
        tolerance = 0.0
    
    if n_rules < 0:
        print("Warning: negative n_rules, setting n_rules=0.")
        n_rules = 0

    m = data.shape[0]
    if not partitions:
        partitions = generate_triangular_partitions(data)

    rules = {} # antecendents (tuple): consequent (int), dof (float)

    for i in range(m):
        memberships = np.array([partitions[j].compute_memberships(data[i, j]) for j in range(data.shape[1])])
        labels = np.argmax(memberships, axis=1)

        dof = np.prod(memberships.max(axis=1)) # degree of fulfillment

        if dof < tolerance:
            continue

        antecedents = tuple(labels[:-1])
        consequent = labels[-1]

        if antecedents not in rules:
            rules[antecedents] = (consequent, dof)
        else:
            last_DOF = rules[antecedents][1]

            if last_DOF < dof: # if 2 rules map to the same antecedents, only the one with the highest DOF wins 
                rules[antecedents] = (consequent, dof)
    
    rules_list = []
    dofs = []
    
    for antecedents, (consequent, dof) in rules.items():
        rules_list.append(RuleSimple(list(antecedents), consequent))
        dofs.append(dof)

    if n_rules == 0: # select all rules
        return RuleBaseRegT1(partitions[:-1], rules_list, partitions[-1])
     
    dofs = np.array(dofs)
    valid_rules = np.argsort(dofs)[::-1][:n_rules]
    selected_rules = [rules_list[i] for i in valid_rules]

    return RuleBaseRegT1(partitions[:-1], selected_rules, partitions[-1])