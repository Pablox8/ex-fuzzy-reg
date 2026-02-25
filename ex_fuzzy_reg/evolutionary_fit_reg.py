"""
Evolutionary Optimization for Fuzzy Rule Base Learning

This module implements genetic algorithm-based optimization for learning fuzzy rule bases.
It provides automatic rule discovery, parameter tuning, and structure optimization for
fuzzy inference systems using evolutionary computation techniques.

Main Components:
    - FitRuleBase: Core optimization problem class for genetic algorithms
    - Fitness functions: Multiple objective functions for rule quality assessment
    - Genetic operators: Specialized crossover, mutation, and selection for fuzzy rules
    - Multi-objective optimization: Support for accuracy vs. complexity trade-offs
    - Parallel evaluation: Efficient fitness evaluation using multiple threads
    - Integration with Pymoo: Leverages the Pymoo optimization framework

The module supports automatic learning of:
    - Rule antecedents (which variables and linguistic terms to use)
    - Rule consequents (output class assignments)
    - Rule structure (number of rules, complexity constraints)
    - Membership function parameters (when combined with other modules)

Key Features:
    - Stratified cross-validation for robust fitness evaluation
    - Multiple fitness metrics (accuracy, MCC, F1-score, etc.)
    - Support for Type-1, Type-2, and General Type-2 fuzzy systems
    - Automatic handling of imbalanced datasets
    - Configurable complexity penalties to avoid overfitting
"""
import os 
from typing import Callable, Any, Optional, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.base import BaseEstimator, ClassifierMixin
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import Problem
from pymoo.core.variable import Integer

from ex_fuzzy.rules import RuleSimple
from ex_fuzzy_reg.rules_reg import RuleBaseRegT1

# Handle pymoo version compatibility for parallelization
try:
    # pymoo < 0.6.0
    from pymoo.parallelization.starmap import StarmapParallelization
except ImportError:
    try:
        # pymoo >= 0.6.0
        from pymoo.core.problem import StarmapParallelization
    except ImportError:
        # Fallback: create a simple wrapper if neither import works
        StarmapParallelization = None

# Import backend abstraction
try:
    from . import fuzzy_sets as fs
    from . import fuzzy_variable as fv
    from . import rules_reg

    
except ImportError:
    import ex_fuzzy_reg.fuzzy_sets as fs
    import ex_fuzzy_reg.fuzzy_variable as fv
    import ex_fuzzy_reg.rules as rules_reg


class FitRuleBase(Problem):
    '''
    Class to model as pymoo problem the fitting of a rulebase for a classification problem using Evolutionary strategies. 
    Supports type 1 and iv fs (iv-type 2)
    '''

    """     def _init_optimize_vl(self, fuzzy_type: fs.FUZZY_SETS, n_linguist_variables: int, domain: list[(float, float)] = None, categorical_variables: list[int] = None, X=None):
        '''
        Inits the corresponding fields if no linguistic partitions were given.

        :param fuzzy type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
        :param n_linguistic_variables: number of linguistic variables per antecedent.
        :param domain: list of the limits for each variable. If None (default) the classifier will compute them empirically.
        '''
        try:
            from . import utils
        except ImportError:
            import utils

        self.lvs = None
        self.vl_names = [FitRuleBase.vl_names[n_linguist_variables[nn]] if n_linguist_variables[nn] < 6 else list(map(str, np.arange(nn))) for nn in range(len(n_linguist_variables))]
        

        self.fuzzy_type = fuzzy_type
        self.domain = domain
        self._precomputed_truth = None
        self.categorical_mask = categorical_variables
        self.categorical_boolean_mask =  np.array(categorical_variables) > 0 if categorical_variables is not None else None 
        self.categorical_variables = {}
        for ix, cat in enumerate(categorical_variables):
            if cat > 0:
                self.categorical_variables[ix] = utils.construct_crisp_categorical_partition(np.array(X)[:, ix], self.var_names[ix], fuzzy_type)
        
        self.n_lv_possible = []
        for ix in range(len(self.categorical_mask)):
            if self.categorical_mask[ix] > 0:
                self.n_lv_possible.append(len(self.categorical_variables[ix]))
            else:
                self.n_lv_possible.append(n_linguist_variables[ix])


    def _init_precomputed_vl(self, linguist_variables: list[fv.FuzzyVariable], X: np.array):
        '''
        Inits the corresponding fields if linguistic partitions for each variable are given.

        :param linguistic_variables: list of fuzzyVariables type.
        :param X: np array samples x features.
        '''
        self.lvs = linguist_variables
        self.vl_names = [lv.linguistic_variable_names() for lv in self.lvs]
        self.n_lv_possible = [len(lv.linguistic_variable_names()) for lv in self.lvs]
        self.fuzzy_type = self.lvs[0].fs_type
        self.domain = None
        self._precomputed_truth = rules_reg.compute_antecedents_memberships(linguist_variables, X)

    vl_names = [  # Linguistic variable names prenamed for some specific cases.
        [],
        [],
        ['Low', 'High'],
        ['Low', 'Medium', 'High'],
        ['Low', 'Medium', 'High', 'Very High'],
        ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    ] """

    def __init__(self, X: np.array, y: np.array, n_rules: int, n_ants: int, thread_runner: Optional[Any]=None,
                 antecedents: list[fv.FuzzyVariable]=None, consequent: fv.FuzzyVariable=None, n_linguistic_variables: int=3, fuzzy_type=fs.FUZZY_SETS.t1, domain: list=None, tolerance: float=0.01, alpha: float=0.0, beta: float=0.0, optimize_lv: bool=False, fuzzy_set_type: str='trapezoidal', backend_name: str='pymoo', var_names: list=None) -> None:
        '''
        Constructor method. Initializes the classifier with the number of antecedents, linguist variables and the kind of fuzzy set desired.

        Args:
            X (np.ndarray or pd.DataFrame): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target class vector of shape (n_samples,).
            n_rules (int): Number of rules to optimize in the fuzzy model.
            n_ants (int): Maximum number of antecedents to use in the fuzzy rulebase.
            thread_runner (Optional[Any], optional): An optional threading mechanism for parallel processing. Defaults to None.
            antecedents (list[FuzzyVariable], optional): Precomputed list of fuzzy variables (antecedents). Conflicts with n_linguistic_variables. Defaults to None.
            consequent (FuzzyVariable, optional): The fuzzy variable used as the consequent in the rules. Only used in regression problems. Defaults to None.
            n_linguistic_variables (int, optional): Number of linguistic variables per antecedent. Defaults to 3. Ignored if `antecedents` is provided.
            fuzzy_type (fs.FUZZY_SETS, optional): The fuzzy set or fuzzy set extension used for the linguistic variable. Defaults to fs.FUZZY_SETS.t1.
            domain (list[tuple], optional): A list specifying the lower and upper bounds for each input variable. Defaults to None, where empirical min/max values are used.
            tolerance (float, optional): Tolerance for evaluating the fuzzy model’s performance. Defaults to 0.01.
            alpha (float, optional): Weight for the rulebase size term in the fitness function (penalizes the number of rules). Defaults to 0.0.
            beta (float, optional): Weight for the average rule size term in the fitness function. Defaults to 0.0.
            optimize_lv (bool, optional): Whether to optimize linguistic variables during the learning process. Defaults to False.
            fuzzy_set_type (str, optional): Type of fuzzy set (e.g., 'trapezoidal', 'triangular'). Defaults to 'trapezoidal'.
            backend_name (str, optional): The optimization backend to use. Defaults to 'pymoo'.
            var_names (list[str], optional): List of variable names. Defaults to None, where names are auto-generated or extracted from DataFrame columns.
        '''
        if var_names is not None:
            self.var_names = var_names
            self.X = np.array(X) if not isinstance(X, np.ndarray) else X
        else:
            try:
                self.var_names = list(X.columns)
                self.X = X.values
            except AttributeError:
                self.X = X
                self.var_names = [str(ix) for ix in range(X.shape[1])]

        self.tolerance = tolerance

        self.y = y
        self.n_rules = n_rules
        self.n_ants = n_ants
        self.n_cons = 1 
        self.domain = domain

        self.antecedents = antecedents
        self.consequent = consequent

        if self.antecedents is not None:
            self.n_lv_possible = [len(antecedent.linguistic_variables) for antecedent in self.antecedents]
            self.lvs = [antecedent.linguistic_variables for antecedent in self.antecedents]
        else:
            self.n_lv_possible = [n_linguistic_variables] * self.X.shape[1]
            self.lvs = None

        """         if antecedents is not None:
            self._init_precomputed_vl(antecedents, X)
        else:
            n_linguistic_variables = [n_linguistic_variables] * self.X.shape[1]
            
            self._init_optimize_vl(
                fuzzy_type=fuzzy_type, n_linguist_variables=n_linguistic_variables, categorical_variables=categorical_mask, domain=domain, X=X) """

        if self.domain is None:
            # If all the variables are numerical, then we can compute the min/max of the domain.
            if np.all([np.issubdtype(self.X[:, ix].dtype, np.number) for ix in range(self.X.shape[1])]):
                self.min_bounds = np.min(self.X, axis=0)
                self.max_bounds = np.max(self.X, axis=0)
            else:
                self.min_bounds = np.zeros(self.X.shape[1])
                self.max_bounds = np.zeros(self.X.shape[1])

                for ix in range(self.X.shape[1]):
                    if np.issubdtype(self.X[:, ix].dtype, np.number):
                        self.min_bounds[ix] = np.min(self.X[:, ix])
                        self.max_bounds[ix] = np.max(self.X[:, ix])
                    else:
                        self.min_bounds[ix] = 0
                        self.max_bounds[ix] = len(np.unique(self.X[:, ix][~pd.isna(self.X[:, ix])]))
        else:
            # Handle different domain formats:
            # - List of tuples/arrays: [(min1, max1), (min2, max2), ...] from initial_rules
            # - Tuple of arrays: (min_bounds, max_bounds)
            if isinstance(self.domain, list) and len(self.domain) > 0 and hasattr(self.domain[0], '__len__') and len(self.domain[0]) == 2:
                # Domain is list of (min, max) pairs per feature
                self.min_bounds = np.array([d[0] for d in self.domain])
                self.max_bounds = np.array([d[1] for d in self.domain])
            else:
                self.min_bounds, self.max_bounds = self.domain

        self.antecedents_referencial = [np.linspace(
            self.min_bounds[ix], self.max_bounds[ix], 100) for ix in range(self.X.shape[1])] 

        possible_antecedent_bounds = np.array([[0, self.X.shape[1] - 1]] * self.n_ants * self.n_rules)
        vl_antecedent_bounds = np.array([[ -1, self.n_lv_possible[ax] - 1] for ax in range(self.n_ants)] * self.n_rules)
        antecedent_bounds = np.concatenate((possible_antecedent_bounds, vl_antecedent_bounds))

        vars_antecedent = {ix: Integer(bounds=antecedent_bounds[ix]) for ix in range(len(antecedent_bounds))}
        aux_counter = len(vars_antecedent)

        if self.lvs is None:
            self.feature_domain_bounds = np.array([[0, 99] for ix in range(self.X.shape[1])])
            if self.fuzzy_type == fs.FUZZY_SETS.t1:
                correct_size = [(self.n_lv_possible[ixx]-1) * 4 + 3 for ixx in range(len(self.n_lv_possible))]
            elif self.fuzzy_type == fs.FUZZY_SETS.t2:
                correct_size = [(self.n_lv_possible[ixx]-1) * 6 + 2 for ixx in range(len(self.n_lv_possible))]
            elif self.fuzzy_type == fs.FUZZY_SETS.gt2:
                correct_size = [(self.n_lv_possible[ixx]-1) * 6 + 2 for ixx in range(len(self.n_lv_possible))]
            else:
                raise ValueError(f"Fuzzy type {self.fuzzy_type} not supported for dynamic membership optimization. "
                                "Please provide precomputed linguistic_variables.")
            membership_bounds = np.concatenate(
                [[self.feature_domain_bounds[ixx]] * correct_size[ixx] for ixx in range(len(self.n_lv_possible))])

            vars_memberships = {
                aux_counter + ix: Integer(bounds=membership_bounds[ix]) for ix in range(len(membership_bounds))}
            aux_counter += len(vars_memberships)

        final_consequent_bounds = np.array([[np.min(self.y), np.max(self.y)]] * self.n_rules)
        vars_consequent = {aux_counter + ix: Integer(bounds=final_consequent_bounds[ix]) for ix in range(len(final_consequent_bounds))}

        if self.lvs is None:
            vars = {key: val for d in [vars_antecedent, vars_memberships, vars_consequent] for key, val in d.items()}
            varbound = np.concatenate(
                (antecedent_bounds, membership_bounds, final_consequent_bounds), axis=0)
        else:
            vars = {key: val for d in [vars_antecedent, vars_consequent] for key, val in d.items()}
            varbound = np.concatenate(
                (antecedent_bounds, final_consequent_bounds), axis=0)

        nVar = len(varbound)
        self.single_gen_size = nVar

        """ self.single_gen_size = (n_ants * n_rules) + (n_ants * n_rules) + (n_rules) 
        if optimize_lv:
            n = 4 if fuzzy_set_type == 'trapezoidal' else 3
            self.single_gen_size += ((n_ants + 1) * n_linguistic_variables * n)  """

        self.alpha_ = alpha
        self.beta_ = beta
        self.backend_name = backend_name

        if thread_runner is not None:
            super().__init__(
                vars=vars,
                n_var=nVar,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1],
                elementwise_runner=thread_runner)
        else:
            super().__init__(
                vars=vars,
                n_var=nVar,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1]) 


    # TODO: optimize_lv = True case
    def encode_rulebase(self, rule_base: RuleBaseRegT1, optimize_lv: bool=False):
        antecedents_used = [] # first section
        antecedents_lvs_used = [] # second section
        consequent_lvs_used = [] # fourth section

        for rule in rule_base.rules:
            antecedents_used += list(np.where(np.array(rule.antecedents) == -1, 0, 1))
            antecedents_lvs_used += rule.antecedents
            consequent_lvs_used += [rule.consequent]

        x = antecedents_used + antecedents_lvs_used + consequent_lvs_used
        return np.array(x)


    # TODO: optimize_lv = True case
    def _construct_ruleBase(self, x: np.ndarray, fuzzy_type: fs.FUZZY_SETS=None, optimize_lv: bool=False):
        rules = []
        fourth_pointer = 2 * self.n_ants * self.n_rules

        for i in range(self.n_rules):
            first_pointer = i * self.n_ants
            second_pointer = first_pointer + (self.n_ants * self.n_rules)

            chosen_antecedents = x[first_pointer : first_pointer + self.n_ants]
            ants_lvs_used = x[second_pointer : second_pointer + self.n_ants]

            cons_lv_used = x[fourth_pointer + i]

            rules.append(RuleSimple(ants_lvs_used, cons_lv_used))

        return RuleBaseRegT1(self.antecedents, rules, self.consequent)


    def _evaluate_slow(self, x: np.array, out: dict, *args, **kwargs):
        '''
        :param x: array of train samples. x shape = features
            those features are the parameters to optimize.

        :param out: dict where the F field is the fitness. It is used from the outside.
        '''
        ruleBase = self._construct_ruleBase(x, self.fuzzy_type)

        if len(ruleBase.get_rules()) > 0:
            score = self.fitness_func(ruleBase, self.X, self.y, self.tolerance, self.alpha_, self.beta_, self._precomputed_truth)
        else:
            score = 0.0
        
        out["F"] = 1 - score

    
    def _evaluate(self, x: np.array, out: dict, *args, **kwargs):
        '''
        Faster version of the evaluate function, which does not reconstruct the rule base each time. It computes a functional equivalent with numpy operations,
        which saves considerable time.

        :param x: array of train samples. x shape = features
            those features are the parameters to optimize.
        :param out: dict where the F field is the fitness. It is used from the outside.
        '''
        # Fast path only works for T1 fuzzy sets with membership optimization (lvs is None)
        # When lvs is precomputed, the gene structure is different, so use slow path
        self._evaluate_slow(x, out, *args, **kwargs)


    def fitness_func(self, ruleBase: rules_reg.RuleBase, X:np.array, y:np.array, tolerance:float, alpha:float=0.0, beta:float=0.0, precomputed_truth:np.array=None) -> float:
        '''
        Fitness function for the optimization problem.
        :param ruleBase: RuleBase object
        :param X: array of train samples. X shape = (n_samples, n_features)
        :param y: array of train labels. y shape = (n_samples,)
        :param tolerance: float. Tolerance for the size evaluation.
        :param alpha: float. Weight for the accuracy term.
        :param beta: float. Weight for the average rule size term.
        :param precomputed_truth: np array. If given, it will be used as the truth values for the evaluation.
        :return: float. Fitness value.
        '''
        if precomputed_truth is None:
            precomputed_truth = rules.compute_antecedents_memberships(ruleBase.antecedents, X)

        ev_object = evr.evalRuleBase(ruleBase, X, y, precomputed_truth=precomputed_truth)
        ev_object.add_full_evaluation()
        ruleBase.purge_rules(tolerance)

        if len(ruleBase.get_rules()) > 0: 
            score_acc = ev_object.classification_eval()
            score_rules_size = ev_object.size_antecedents_eval(tolerance)
            score_nrules = ev_object.effective_rulesize_eval(tolerance)

            score = score_acc + score_rules_size * alpha + score_nrules * beta
        else:
            score = 0.0
            
        return score
    

