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
# TODO: organize imports
import os 
from typing import Callable, Any, Optional, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, root_mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from multiprocessing.pool import ThreadPool, Pool
from pymoo.core.problem import Problem
from pymoo.core.variable import Integer

from ex_fuzzy.rules import RuleSimple
from ex_fuzzy_reg import evolutionary_backends_reg as ev_backends
from ex_fuzzy_reg.fuzzy_sets import TriangularFS, TrapezoidalFS
from ex_fuzzy_reg.fuzzy_variable import FuzzyVariable
from ex_fuzzy_reg.rules_reg import RuleBaseRegT1
from ex_fuzzy_reg import rules_reg_utils

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


class BaseFuzzyRulesRegressor(RegressorMixin, BaseEstimator):
    '''
    Class that is used as a regressor for a fuzzy rule based system. Supports precomputed and optimization of the linguistic variables.
    '''

    def __init__(self,  n_rules: int = 30, n_ants: int = 4, fuzzy_type: fs.FUZZY_SETS = fs.FUZZY_SETS.t1, fuzzy_set_type: str='trapezoidal', tolerance: float = 0.0,
                 n_linguistic_variables: int = 3, verbose=False, antecedents: list[fv.FuzzyVariable] = None, consequent: fv.FuzzyVariable = None, categorical_mask: np.array=None,
                 domain: list = None, precomputed_rules: RuleBaseRegT1=None, runner: int=1, allow_unknown:bool=False, backend: str='pymoo', optimize_lv: bool=False) -> None:
        '''
        # TODO: fix the docs
        Inits the optimizer with the corresponding parameters.

        :param nRules: number of rules to optimize.
        :param nAnts: max number of antecedents to use.
        :param fuzzy type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
        :param tolerance: tolerance for the dominance score of the rules.
        :param n_linguist_variables: number of linguistic variables per antecedent.
        :param verbose: if True, prints the progress of the optimization.
        :param linguistic_variables: list of fuzzyVariables type. If None (default) the optimization process will init+optimize them.
        :param domain: list of the limits for each variable. If None (default) the classifier will compute them empirically.
        :param n_class: names of the classes in the problem. If None (default) the classifier will compute it empirically.
        :param precomputed_rules: MasterRuleBase object. If not None, the classifier will use the rules in the object and ignore the conflicting parameters.
        :param runner: number of threads to use. If None (default) the classifier will use 1 thread.
        :param ds_mode: mode for the dominance score. 0: normal dominance score, 1: rules without weights, 2: weights optimized for each rule based on the data.
        :param allow_unknown: if True, the classifier will allow the unknown class in the classification process. (Which would be a -1 value)
        :param backend: evolutionary backend to use. Options: 'pymoo' (default, CPU) or 'evox' (GPU-accelerated). Install with: pip install ex-fuzzy[evox]
        '''
        if precomputed_rules is not None:
            self.n_rules = len(precomputed_rules.get_rules())
            self.n_ants = len(precomputed_rules.get_rules()[0].antecedents)
            self.rule_base = precomputed_rules
        else:
            self.n_rules = n_rules
            self.n_ants = n_ants
            self.categorical_mask = categorical_mask

        self.custom_loss = None
        self.verbose = verbose
        self.tolerance = tolerance
        self.allow_unknown = allow_unknown
        self.optimize_lv = optimize_lv
        
        # Initialize evolutionary backend
        try:
            self.backend = ev_backends.get_backend(backend)
        except ValueError as e:
            if verbose:
                print(f"Warning: {e}. Falling back to pymoo backend.")
            self.backend = ev_backends.get_backend('pymoo')

        if runner > 1 and StarmapParallelization is not None:
            #pool = ThreadPool(runner)
            pool = Pool(runner)
            self.thread_runner = StarmapParallelization(pool.starmap)
        else:
            if runner > 1 and StarmapParallelization is None and verbose:
                print("Warning: Parallelization not available with this pymoo version. Running single-threaded.")
            self.thread_runner = None
        
        if antecedents is not None:
            self.antecedents = antecedents
            self.n_linguistic_variables = len(self.antecedents[0].linguistic_variables)
            self.domain = None
            self.fuzzy_type = self.antecedents[0].fuzzy_type()
            self.fuzzy_set_type = self.antecedents[0].linguistic_variables[0].shape

            if self.n_ants > len(antecedents):
                self.n_ants = len(antecedents)
                if verbose:
                    print('Warning: The number of antecedents is higher than the number of variables. Setting n_ants to the number of linguistic variables. (' + str(len(antecedents)) + ')')
        else:
            self.antecedents = None
            self.fuzzy_type = fuzzy_type
            self.n_linguistic_variables = n_linguistic_variables
            self.domain = domain
            self.fuzzy_set_type = fuzzy_set_type
        
        self.consequent = consequent
        
        # TODO: allow user to set alpha and beta
        self.alpha_ = 0.0
        self.beta_ = 0.0


    def customized_loss(self, loss_function):
        '''
        Function to customize the loss function used for the optimization.

        :param loss_function: function that takes as input the true labels and the predicted labels and returns a float.
        :return: None
        '''
        self.custom_loss = loss_function


    def fit(self, X: np.array, y: np.array, n_gen:int=70, pop_size:int=30,
            checkpoints:int=0, candidate_rules: RuleBaseRegT1=None, initial_rules: RuleBaseRegT1=None, random_state:int=33,
            var_prob:float=0.9, sbx_eta:float=3.0, mut_prob:float=0.2, mutation_eta:float=4.0, tournament_size:int=3, bootstrap_size:int=1000, checkpoint_path:str='',
            p_value_compute:bool=False, checkpoint_callback: Callable[[int, RuleBaseRegT1], None] = None) -> None:
        '''
        Fits a fuzzy rule based classifier using a genetic algorithm to the given data.

        :param X: numpy array samples x features
        :param y: labels. integer array samples (x 1)
        :param n_gen: integer. Number of generations to run the genetic algorithm.
        :param pop_size: integer. Population size for each gneration.
        :param checkpoints: integer. Number of checkpoints to save the best rulebase found so far.
        :param candidate_rules: if these rules exist, the optimization process will choose the best rules from this set. If None (default) the rules will be generated from scratch.
        :param initial_rules: if these rules exist, the optimization process will start from this set. If None (default) the rules will be generated from scratch.
        :param random_state: integer. Random seed for the optimization process.
        :param var_prob: float. Probability of crossover for the genetic algorithm.
        :param sbx_eta: float. Eta parameter for the SBX crossover.
        :param checkpoint_path: string. Path to save the checkpoints. If None (default) the checkpoints will be saved in the current directory.
        :param mutation_eta: float. Eta parameter for the polynomial mutation.
        :param tournament_size: integer. Size of the tournament for the genetic algorithm.
        :param checkpoint_callback: function. Callback function that get executed at each checkpoint ('checkpoints' must be greater than 0), its arguments are the generation number and the rule_base of the checkpoint.
        :return: None. The classifier is fitted to the data.
        '''
        y = y.reshape(-1, 1)
        if isinstance(X, pd.DataFrame):
            lvs_names = list(X.columns)
            X = X.values
        else:
            lvs_names = [str(ix) for ix in range(X.shape[1])]
            
        if candidate_rules is None:
            if initial_rules is not None:
                self.fuzzy_type = initial_rules.fuzzy_type()
                self.n_linguistic_variables = len(initial_rules[0].linguistic_variables)
                self.domain = [fv.domain() for fv in initial_rules[0].antecedents]
                self.n_rules = len(initial_rules.get_rules())
                self.n_ants = len(initial_rules.get_rules()[0].antecedents)
                # Use linguistic variables from the initial rules (don't optimize memberships)
                if self.antecedents is None:
                    self.antecedents = initial_rules[0].antecedents
                if self.consequent is None:
                    self.consequent = initial_rules[0].consequent

            if self.antecedents is None:
                if self.n_ants > X.shape[1]:
                    self.n_ants = X.shape[1]
                    if self.verbose:
                        print('Warning: The number of antecedents is higher than the number of variables. Setting n_ants to the number of variables. (' + str(X.shape[1]) + ')') 

                # If Fuzzy variables need to be optimized.
                problem = FitRuleBaseReg(X, y, n_rules=self.n_rules, n_ants=self.n_ants, tolerance=self.tolerance, 
                                    n_linguistic_variables=self.n_linguistic_variables, fuzzy_type=self.fuzzy_type, fuzzy_set_type=self.fuzzy_set_type, domain=self.domain, thread_runner=self.thread_runner,
                                    alpha=self.alpha_, beta=self.beta_, categorical_mask=self.categorical_mask,
                                    backend_name=self.backend.name(), var_names=lvs_names, optimize_lv=self.optimize_lv)
            else:
                # If Fuzzy variables are already precomputed.
                problem = FitRuleBaseReg(X, y, n_rules=self.n_rules, n_ants=self.n_ants, 
                                    antecedents=self.antecedents, consequent=self.consequent, domain=self.domain, tolerance=self.tolerance, thread_runner=self.thread_runner,
                                    alpha=self.alpha_, beta=self.beta_,
                                    backend_name=self.backend.name(), var_names=lvs_names, optimize_lv=self.optimize_lv)
        else:
            pass # TODO: provisional, change later
            # self.fuzzy_type = candidate_rules.fuzzy_type()
            # self.n_linguistic_variables = candidate_rules.n_linguistic_variables()
            # problem = ExploreRuleBases(X, y, n_classes=len(np.unique(y)), candidate_rules=candidate_rules, thread_runner=self.thread_runner, nRules=self.n_rules)

        if self.custom_loss is not None:
            problem.fitness_func = self.custom_loss

        # Prepare initial population
        if initial_rules is None:
            rules_gene = None  # Will use default random sampling
        else:
            # rules_gene = problem.encode_rulebase(initial_rules, self.antecedents is None)
            rules_gene = problem.encode_rulebase(initial_rules, optimize_lv=self.optimize_lv)
            rules_gene = (np.ones((pop_size, len(rules_gene))) * rules_gene).astype(int)

        # TODO: checkpoints?
        # Use backend for optimization
        """ if checkpoints > 0:
            # Checkpoint mode - delegate to backend if supported
            if self.backend.name() == 'pymoo':
                # Define checkpoint handler
                def handle_checkpoint(gen: int, best_individual: np.array):
                    rule_base = problem._construct_ruleBase(best_individual, self.fuzzy_type)
                    eval_performance = evr.evalRuleBase(rule_base, np.array(X), y)
                    eval_performance.add_full_evaluation()
                    rule_base.purge_rules(self.tolerance)
                    rule_base.rename_cons(self.classes_names)
                    checkpoint_rules = rule_base.print_rules(True, bootstrap_results=True)
                    
                    if checkpoint_callback is None:
                        with open(os.path.join(checkpoint_path, "checkpoint_" + str(gen)), "w") as f:
                            f.write(checkpoint_rules)
                    else:
                        checkpoint_callback(gen, rule_base)
                
                # Call backend's checkpoint optimization
                result = self.backend.optimize_with_checkpoints(
                    problem=problem,
                    n_gen=n_gen,
                    pop_size=pop_size,
                    random_state=random_state,
                    verbose=self.verbose,
                    checkpoint_freq=checkpoints,
                    checkpoint_callback=handle_checkpoint,
                    var_prob=var_prob,
                    sbx_eta=sbx_eta,
                    mutation_eta=mutation_eta,
                    tournament_size=tournament_size,
                    sampling=rules_gene
                )
                
                best_individual = result['X']
                self.performance = 1 - result['F']
            else:
                # EvoX or other backends: checkpoints not supported
                if self.verbose:
                    print(f"Warning: Checkpoints are not yet supported with {self.backend.name()} backend. Running without checkpoints.")
                result = self.backend.optimize(
                    problem=problem,
                    n_gen=n_gen,
                    pop_size=pop_size,
                    random_state=random_state,
                    verbose=self.verbose,
                    var_prob=var_prob,
                    sbx_eta=sbx_eta,
                    mutation_eta=mutation_eta,
                    tournament_size=tournament_size,
                    sampling=rules_gene
                )
                
                best_individual = result['X']
                self.performance = 1 - result['F']
        else: """

        # precalculate memberships for _evaluate_torch_batch if using EvoX
        if self.backend.name() == 'evox':
            problem._cached_memberships = np.stack([
                ant.compute_memberships(X[:, ix]).T
                for ix, ant in enumerate(problem.antecedents)
            ], axis=1)  # (n_samples, n_vars, n_labels)

            problem._cached_consequent_centroids = np.array([
                fs.centroid() for fs in problem.consequent
            ])  # (n_labels,)

        # Normal optimization without checkpoints
        result = self.backend.optimize(
            problem=problem,
            n_gen=n_gen,
            pop_size=pop_size,
            random_state=random_state,
            verbose=self.verbose,
            var_prob=var_prob,
            sbx_eta=sbx_eta,
            mut_prob=mut_prob,
            mutation_eta=mutation_eta,
            tournament_size=tournament_size,
            sampling=rules_gene
        )
        
        best_individual = result['X']
        self.performance = 1 - result['F']

        self.X = X
        self.var_names = [str(ix) for ix in range(X.shape[1])]

        self.rule_base = problem._construct_ruleBase(best_individual, self.fuzzy_type, optimize_lv=self.optimize_lv)

        self.antecedents = self.rule_base.antecedents if self.antecedents is None else self.antecedents
        self.consequent = self.rule_base.consequent if self.consequent is None else self.consequent

        #self.eval_performance = evr.evalRuleBase(
        #self.rule_base, np.array(X), y)
        #self.eval_performance.add_full_evaluation()
        #self.rule_base.purge_rules(self.tolerance)
        #self.eval_performance.add_full_evaluation() # After purging the bad rules we update the metrics.
        
        """ if p_value_compute:
            self.p_value_validation(bootstrap_size) """

    
    def print_rule_bootstrap_results(self) -> None:
        '''
        Prints the bootstrap results for each rule.
        '''
        self.rule_base.print_rule_bootstrap_results()
    

    def p_value_validation(self, bootstrap_size:int=100):
        '''
        Computes the permutation and bootstrapping p-values for the classifier and its rules.

        :param bootstrap_size: integer. Number of bootstraps samples to use.
        '''
        self.p_value_class_structure, self.p_value_feature_coalitions = self.eval_performance.p_permutation_classifier_validation()
        
        self.eval_performance.p_bootstrapping_rules_validation(bootstrap_size)
        

    def load_rule_base(self, rule_base: RuleBaseRegT1) -> None:
        '''
        Loads a master rule base to be used in the prediction process.

        :param rule_base: ruleBase object.
        :return: None
        '''
        self.rule_base = rule_base
        self.n_rules = len(rule_base.get_rules())
        self.n_ants = len(rule_base.get_rules()[0].antecedents)
        
    
    def explainable_predict(self, X: np.array, out_class_names=False) -> np.array:
        '''
        Returns the predicted class for each sample.
        '''
        return self.rule_base.explainable_predict(X, out_class_names=out_class_names)


    def forward(self, X: np.array) -> np.array:
        '''

        Returns the predicted class for each sample.

        :param X: np array samples x features.
        :param out_class_names: if True, the output will be the class names instead of the class index.
        :return: np array samples (x 1) with the predicted class.
        '''
        try:
            X = X.values  # If X was a pandas dataframe
        except AttributeError:
            pass
        
        return self.rule_base.inference_optimized(X) 
        

    def predict(self, X: np.array) -> np.array:
        '''
        Returns the predicted class for each sample.

        :param X: np array samples x features.
        :param out_class_names: if True, the output will be the class names instead of the class index.
        :return: np array samples (x 1) with the predicted class.
        '''
        return self.forward(X)


    def print_rules(self, return_rules:bool=False, bootstrap_results:bool=False) -> None:
        '''
        Print the rules contained in the fitted rulebase.
        '''
        return self.rule_base.print_rules(return_rules, bootstrap_results)


    def plot_fuzzy_variables(self) -> None:
        '''
        Plot the fuzzy partitions in each fuzzy variable.
        '''
        fuzzy_variables = self.rule_base.rule_bases[0].antecedents

        for ix, fv in enumerate(fuzzy_variables):
            vis_rules.plot_fuzzy_variable(fv)


    def rename_fuzzy_variables(self) -> None:
        '''
        Renames the linguist labels so that high, low and so on are consistent. It does so usually after an optimization process.

        :return: None. Names are sorted accorded to the central point of the fuzzy memberships.
        '''

        for ix in range(len(self.rule_base)):
            fuzzy_variables = self.rule_base.rule_bases[ix].antecedents
            for jx, fv in enumerate(fuzzy_variables):
                if fv[0].shape() != 'categorical':
                    new_order_values = []
                    possible_names = FitRuleBase.vl_names[self.n_linguistic_variables[jx]]

                    for zx, fuzzy_set in enumerate(fv.linguistic_variables):
                        studied_fz = fuzzy_set.type()
                        
                        if studied_fz == fs.FUZZY_SETS.temporal:
                            studied_fz = fuzzy_set.inside_type()

                        if studied_fz == fs.FUZZY_SETS.t1:
                            f1 = np.mean(
                                fuzzy_set.membership_parameters[0] + fuzzy_set.membership_parameters[1])
                        elif (studied_fz == fs.FUZZY_SETS.t2):
                            f1 = np.mean(
                                fuzzy_set.secondMF_upper[0] + fuzzy_set.secondMF_upper[1])
                        elif studied_fz == fs.FUZZY_SETS.gt2:
                            sec_memberships = fuzzy_set.secondary_memberships.values()
                            f1 = float(list(fuzzy_set.secondary_memberships.keys())[np.argmax(
                                [fzm.membership_parameters[2] for ix, fzm in enumerate(sec_memberships)])])

                        new_order_values.append(f1)

                    new_order = np.argsort(np.array(new_order_values))
                    fuzzy_sets_vl = fv.linguistic_variables

                    for jx, x in enumerate(new_order):
                        fuzzy_sets_vl[x].name = possible_names[jx]


    def get_rulebase(self) -> list[np.array]:
        '''
        Get the rulebase obtained after fitting the classifier to the data.

        :return: a matrix format for the rulebase.
        '''
        return self.rule_base.get_rulebase_matrix()
    

    def reparametrice_loss(self, alpha:float, beta:float) -> None:
        '''
        Changes the parameters in the loss function. 

        :note: Does not check for convexity preservation. The user can play with these parameters as it wills.
        :param alpha: controls the MCC term.
        :param beta: controls the average rule size loss.
        '''
        self.alpha_ = alpha
        self.beta_ = beta


    def __call__(self, X:np.array) -> np.array:
        '''
        Returns the predicted class for each sample.

        :param X: np array samples x features.
        :return: np array samples (x 1) with the predicted class.
        '''
        return self.predict(X)


class FitRuleBaseReg(Problem):
    '''
    Class to model as pymoo problem the fitting of a rulebase for a classification problem using Evolutionary strategies. 
    Supports type 1 and iv fs (iv-type 2)
    '''
    def _init_optimize_vl(self, fuzzy_type: fs.FUZZY_SETS, n_linguistic_variables: int, domain: list[(float, float)] = None, categorical_variables: list[int] = None, X=None):
        '''
        Inits the corresponding fields if no linguistic partitions were given.

        :param fuzzy type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
        :param n_linguistic_variables: number of linguistic variables per antecedent.
        :param domain: list of the limits for each variable. If None (default) the classifier will compute them empirically.
        '''
        

        self.lvs = None
        #self.vl_names = [FitRuleBaseReg.vl_names[n_linguistic_variables[nn]] if n_linguistic_variables[nn] < 6 else list(map(str, np.arange(nn))) for nn in range(len(n_linguistic_variables))]
        #self.vl_names = FitRuleBaseReg.vl_names if n_linguistic_variables < 6 else []
        
        self.fuzzy_type = fuzzy_type
        self.domain = domain
        self._precomputed_truth = None
        self.categorical_mask = categorical_variables
        self.categorical_boolean_mask =  np.array(categorical_variables) > 0 if categorical_variables is not None else None 

        # TODO: categorical variables
        """ self.categorical_variables = {}
        for ix, cat in enumerate(categorical_variables):
            if cat > 0:
                self.categorical_variables[ix] = utils.construct_crisp_categorical_partition(np.array(X)[:, ix], self.var_names[ix], fuzzy_type)
        
        self.n_lv_possible = []
        for ix in range(len(self.categorical_mask)):
            if self.categorical_mask[ix] > 0:
                self.n_lv_possible.append(len(self.categorical_variables[ix]))
            else:
                self.n_lv_possible.append(n_linguistic_variables[ix]) """


    def _init_precomputed_vl(self, linguistic_variables: list[fv.FuzzyVariable], X: np.array):
        '''
        Inits the corresponding fields if linguistic partitions for each variable are given.

        :param linguistic_variables: list of fuzzyVariables type.
        :param X: np array samples x features.
        '''
        self.lvs = linguistic_variables
        self.vl_names = [lv.linguistic_variable_names() for lv in self.lvs]
        self.fuzzy_type = self.lvs[0].fs_type
        self.domain = None
        self._precomputed_truth = rules_reg.compute_antecedents_memberships_batch(linguistic_variables, X)


    vl_names = [  # Linguistic variable names prenamed for some specific cases.
        [],
        [],
        ['Low', 'High'],
        ['Low', 'Medium', 'High'],
        ['Low', 'Medium', 'High', 'Very High'],
        ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    ]

    # TODO: this will work assuming every fuzzy variable has the same number of linguistic variables.
    def __init__(self, X: np.array, y: np.array, n_rules: int, n_ants: int, thread_runner: Optional[Any]=None,
                 antecedents: list[fv.FuzzyVariable]=None, consequent: fv.FuzzyVariable=None, n_linguistic_variables: int=3, fuzzy_type=fs.FUZZY_SETS.t1, domain: list=None, categorical_mask: np.array=None, tolerance: float=0.01, alpha: float=0.0, beta: float=0.0, optimize_lv: bool=False, fuzzy_set_type: str='trapezoidal', backend_name: str='pymoo', var_names: list=None) -> None:
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
            tolerance (float, optional): Tolerance for evaluating the fuzzy model's performance. Defaults to 0.01.
            alpha (float, optional): Weight for the rulebase size term in the fitness function (penalizes the number of rules). Defaults to 0.0.
            beta (float, optional): Weight for the average rule size term in the fitness function. Defaults to 0.0.
            optimize_lv (bool, optional): Whether to optimize linguistic variables during the learning process. Defaults to False.
            fuzzy_set_type (str, optional): Type of fuzzy set (e.g., 'trapezoidal', 'triangular'). Defaults to 'trapezoidal'.
            backend_name (str, optional): The optimization backend to use. Defaults to 'pymoo'.
            var_names (list[str], optional): List of variable names. Defaults to None, where names are auto-generated or extracted from DataFrame columns.
        '''
        """ if optimize_lv:
            raise NotImplementedError """

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
        self.n_linguistic_variables = n_linguistic_variables
        self.domain = domain
        self.fuzzy_set_type = fuzzy_set_type
        self.optimize_lv = optimize_lv

        self.antecedents = antecedents
        self.consequent = consequent

        if categorical_mask is None:
            self.categorical_mask = np.zeros(X.shape[1])
        else:
            self.categorical_mask = categorical_mask

        if antecedents is not None:
            self._init_precomputed_vl(antecedents, X)
        else:
            self._init_optimize_vl(
                fuzzy_type=fuzzy_type, n_linguistic_variables=n_linguistic_variables, categorical_variables=self.categorical_mask, domain=domain, X=X) 

        if self.domain is None:
            # If all the variables are numerical, then we can compute the min/max of the domain.
            if np.all([np.issubdtype(self.X[:, ix].dtype, np.number) for ix in range(self.X.shape[1])]):
                self.min_bounds = np.append(np.min(self.X, axis=0), np.min(y))
                self.max_bounds = np.append(np.max(self.X, axis=0), np.max(y))
            else:
                self.min_bounds = np.zeros(self.X.shape[1] + self.y.shape[1])
                self.max_bounds = np.zeros(self.X.shape[1] + self.y.shape[1])

                for ix in range(self.X.shape[1]):
                    if np.issubdtype(self.X[:, ix].dtype, np.number):
                        self.min_bounds[ix] = np.min(self.X[:, ix])
                        self.max_bounds[ix] = np.max(self.X[:, ix])
                    else:
                        self.min_bounds[ix] = 0
                        self.max_bounds[ix] = len(np.unique(self.X[:, ix][~pd.isna(self.X[:, ix])]))
               
                self.min_bounds[-1] = np.min(y)
                self.max_bounds[-1] = np.max(y)
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

        self.consequent_referencial = np.linspace(np.min(y), np.max(y), 100) 

        feature_idx_bounds = np.array([[0, self.X.shape[1] - 1]] * self.n_ants * self.n_rules)
        #feature_idx_bounds = np.array([[0, 1]] * self.n_ants * self.n_rules)
        vl_idx_bounds = np.array([[-1, self.n_linguistic_variables - 1]] * self.n_ants * self.n_rules)

        # antecedent bounds
        all_bounds = [feature_idx_bounds, vl_idx_bounds]

        if optimize_lv:
            n = 4 if fuzzy_set_type == 'trapezoidal' else 3
            n_membership_params = (self.n_ants + 1) * n_linguistic_variables * n
            membership_bounds = np.array([[0, 99]] * n_membership_params)
            all_bounds.append(membership_bounds)

        consequent_bounds = np.array([[0, self.n_linguistic_variables-1]] * self.n_rules)
        all_bounds.append(consequent_bounds)

        varbound = np.concatenate(all_bounds, axis=0) 

        self.single_gen_size = len(varbound) 

        self.alpha_ = alpha
        self.beta_ = beta
        self.backend_name = backend_name

        # TODO: figure out what is vars used for
        if thread_runner is not None:
            super().__init__(
                #vars=vars,
                n_var=self.single_gen_size,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1],
                elementwise_runner=thread_runner)
        else:
            super().__init__(
                #vars=vars,
                n_var=self.single_gen_size,
                n_obj=1,
                elementwise=True,
                vtype=int,
                xl=varbound[:, 0],
                xu=varbound[:, 1]) 


    def encode_rulebase(self, rule_base: RuleBaseRegT1, optimize_lv: bool=False):
        n_lv_possible_ants = len(rule_base.antecedents[0].get_linguistic_variables())
        n_lv_possible_cons = len(rule_base.consequent.get_linguistic_variables())

        antecedents_used = [] # first section
        antecedents_lvs_used = [] # second section
        partitions_params = [] # third section
        consequent_lvs_used = [] # fourth section

        for rule in rule_base.rules:
            antecedents_used += list(np.where(np.array(rule.antecedents) == -1, 0, 1))
            antecedents_lvs_used += list(np.where(np.array(rule.antecedents) == -1, 0, rule.antecedents))
            consequent_lvs_used += [rule.consequent]

        if optimize_lv:
            for ix in range(self.n_ants):
                fuzzy_variable = rule_base.antecedents[ix]
                for linguistic_variable in range(n_lv_possible_ants):
                    fz_parameters = fuzzy_variable[linguistic_variable].membership_parameters
                    parameters_closest_idxs = []
                    for jx, fz_parameter in enumerate(fz_parameters):
                        closest_idx = (np.abs(np.asarray(self.antecedents_referencial[ix]) - fz_parameter)).argmin()
                        parameters_closest_idxs.append(closest_idx)
                    partitions_params += parameters_closest_idxs 

            for linguistic_variable in range(n_lv_possible_cons):
                fz_parameters = rule_base.consequent[linguistic_variable].membership_parameters
                parameters_closest_idxs = [] 
                for jx, fz_parameter in enumerate(fz_parameters):
                    closest_idx = (np.abs(np.asarray(self.consequent_referencial) - fz_parameter)).argmin()
                    parameters_closest_idxs.append(closest_idx)
                partitions_params += parameters_closest_idxs

        x = antecedents_used + antecedents_lvs_used + partitions_params + consequent_lvs_used
        return np.array(x)


    def _construct_ruleBase(self, x: np.ndarray, fuzzy_type: fs.FUZZY_SETS=None, optimize_lv: bool=False):
        n = 4 if self.fuzzy_set_type == 'trapezoidal' else 3
        if optimize_lv:
            fourth_pointer = 2 * self.n_ants * self.n_rules + (self.n_ants + 1) * self.n_linguistic_variables * n
        else:
            fourth_pointer = 2 * self.n_ants * self.n_rules

        rules = []
        for i in range(self.n_rules):
            first_pointer = i * self.n_ants
            second_pointer = first_pointer + (self.n_ants * self.n_rules)

            chosen_antecedents_idx = x[first_pointer : first_pointer + self.n_ants].astype(bool)

            ants_lvs_used = x[second_pointer : second_pointer + self.n_ants]

            ants_lvs_used[~chosen_antecedents_idx] = -1

            cons_lv_used = int(x[fourth_pointer + i])

            rules.append(RuleSimple(ants_lvs_used, cons_lv_used))
        
        if optimize_lv:
            linguistic_variables = []
            for i in range(self.n_ants + 1):
                partitions = []
                third_pointer = 2 * self.n_ants * self.n_rules + i * self.n_linguistic_variables * n

                for j in range(self.n_linguistic_variables):
                    params_start_pointer = third_pointer + j * n
                    domain = [self.min_bounds[i], self.max_bounds[i]]
                    partition_params = x[params_start_pointer : params_start_pointer + n]
                    referential = self.antecedents_referencial[i] if i < self.n_ants else self.consequent_referencial
                    actual_params = [referential[int(idx)] for idx in partition_params]
                    if self.fuzzy_set_type == 'trapezoidal':
                        partitions.append(TrapezoidalFS(FitRuleBaseReg.vl_names[n][j], actual_params, domain))
                    else:
                        partitions.append(TriangularFS(FitRuleBaseReg.vl_names[n][j], actual_params, domain))

                linguistic_variables.append(partitions)
            
            antecedents = []
            for i, partitions in enumerate(linguistic_variables[:-1]):
                antecedents.append(FuzzyVariable(f"Antecedent Var {i}", partitions))

            consequent = FuzzyVariable("Consequent", linguistic_variables[-1])

            return RuleBaseRegT1(antecedents, rules, consequent)
        
        antecedents = self.antecedents
        consequent = self.consequent
        if antecedents is None or consequent is None:
            data = np.hstack([self.X, self.y])
            gen_fn = (rules_reg_utils.generate_trapezoidal_partitions
                    if self.fuzzy_set_type == 'trapezoidal'
                    else rules_reg_utils.generate_triangular_partitions)
            all_partitions = gen_fn(data, n_labels=self.n_linguistic_variables,
                                    fv_label_names=self.var_names + ["Consequent"])
            if antecedents is None:
                antecedents = all_partitions[:-1]
            if consequent is None:
                consequent = all_partitions[-1]

        return RuleBaseRegT1(self.antecedents, rules, self.consequent)


    def _evaluate_slow(self, x: np.array, out: dict, *args, **kwargs):
        '''
        :param x: array of train samples. x shape = features
            those features are the parameters to optimize.

        :param out: dict where the F field is the fitness. It is used from the outside.
        '''
        ruleBase = self._construct_ruleBase(x, self.fuzzy_type, optimize_lv=self.optimize_lv)

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


    def _evaluate_torch_batch(self, population: 'torch.Tensor', device: 'torch.device') -> 'torch.Tensor':
        """
        Evalúa toda la población completamente vectorizado en PyTorch/GPU.
        Sin bucles Python — opera sobre tensores (pop_size, n_rules, n_samples).
        Requiere self._cached_memberships y self._cached_consequent_centroids precalculados.

        :param population: tensor (pop_size, n_var) con los genotipos
        :param device: torch device (CPU o GPU)
        :return: tensor (pop_size,) con NRMSE por individuo (minimización)
        """
        import torch

        pop_size, n_var = population.shape
        pop_np = population.cpu().numpy().astype(int)

        n_rules = self.n_rules
        n_ants  = self.n_ants
        fourth_pointer = 2 * n_ants * n_rules

        # decode genes: 
        act_genes = pop_np[:, :n_ants * n_rules].reshape(pop_size, n_rules, n_ants).astype(bool) # (pop_size, n_rules, n_ants)

        lv_genes = pop_np[:, n_ants * n_rules : 2 * n_ants * n_rules].reshape(pop_size, n_rules, n_ants) # (pop_size, n_rules, n_ants)
        lv_genes[~act_genes] = -1 # -1 if inactive

        cons_lv = pop_np[:, fourth_pointer : fourth_pointer + n_rules] # (pop_size, n_rules)

        # load precomputed memberships and centroids using torch:
        memberships_t = torch.tensor(self._cached_memberships, dtype=torch.float32, device=device) # (n_samples, n_vars, n_labels)
        y_t           = torch.tensor(self.y.ravel(), dtype=torch.float32, device=device)
        centroids_t   = torch.tensor(self._cached_consequent_centroids, dtype=torch.float32, device=device)

        n_samples = memberships_t.shape[0]
        n_labels  = centroids_t.shape[0]
        value_range = float(y_t.max() - y_t.min()) if y_t.max() != y_t.min() else 1.0
        fallback    = float(centroids_t.mean())

        # build antecedent tensor:
        lv_genes_t = torch.tensor(lv_genes, dtype=torch.long, device=device)  # (pop, rules, ants)

        # clamp for secure indexing
        lv_safe = torch.clamp(lv_genes_t, 0, n_labels - 1)  # (pop, rules, ants)

        mem_perm = memberships_t.permute(1, 2, 0)  # (n_vars, n_labels, n_samples)

        mem_exp = mem_perm.unsqueeze(0).unsqueeze(0)  # (1, 1, n_ants, n_labels, n_samples)
        mem_exp = mem_exp.expand(pop_size, n_rules, n_ants, n_labels, n_samples)

        # lv_safe: (pop, rules, ants) → (pop, rules, ants, 1, n_samples)
        lv_idx = lv_safe.unsqueeze(-1).unsqueeze(-1).expand(pop_size, n_rules, n_ants, 1, n_samples)

        # Gather sobre dim 3 (n_labels) → (pop, rules, ants, 1, n_samples)
        ant_mem = mem_exp.gather(3, lv_idx).squeeze(3)  # (pop, rules, ants, n_samples)

        # inactive antecedents have membership = 1.0, they don't contribute to min t-norm
        active_mask = (lv_genes_t != -1)
        act_exp = active_mask.unsqueeze(-1).expand_as(ant_mem)  # (pop, rules, ants, n_samples)
        ant_mem = torch.where(act_exp, ant_mem, torch.ones_like(ant_mem))

        # min t-norm
        firing_strengths = ant_mem.min(dim=2).values  # (pop, rules, n_samples)

        # consequent centroids
        cons_idx = torch.tensor(cons_lv, dtype=torch.long, device=device)
        cons_idx = torch.clamp(cons_idx, 0, n_labels - 1)
        rule_centroids = centroids_t[cons_idx]  # (pop_size, n_rules)

        numerator   = torch.einsum('prs,pr->ps', firing_strengths, rule_centroids)
        denominator = firing_strengths.sum(dim=1)  # (pop, n_samples)

        y_pred = torch.where(
            denominator > 0,
            numerator / denominator,
            torch.full_like(denominator, fallback)
        )  # (pop_size, n_samples)

        rmse  = torch.sqrt(torch.mean((y_pred - y_t.unsqueeze(0)) ** 2, dim=1))  # (pop,)
        nrmse = torch.clamp(rmse / value_range, 0.0, 1.0)

        return nrmse  # (pop_size,), already minimized


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

        '''
        this should be the classification_eval method for evalRuleBase object
        other metrics can be used, this is provisional
        '''
        if precomputed_truth is not None:
            y_pred = ruleBase.inference_optimized(X, precomputed_truth)
        else:
            y_pred = ruleBase.inference_optimized(X)
        # 1 - Normalized RMSE
        score = 1 - (root_mean_squared_error(y, y_pred)) / (self.max_bounds[-1] - self.min_bounds[-1])

        score = max(0.0, min(1.0, score))

        # TODO: evalRuleBaseReg ¿?
        """ 
        if precomputed_truth is None:
            precomputed_truth = rules_reg.compute_antecedents_memberships(linguistic_variables, X)
        
        ev_object = evr.evalRuleBase(ruleBase, X, y, precomputed_truth=precomputed_truth)
        ev_object.add_full_evaluation()
        ruleBase.purge_rules(tolerance)

        if len(ruleBase.get_rules()) > 0: 
            score_acc = ev_object.classification_eval()
            score_rules_size = ev_object.size_antecedents_eval(tolerance)
            score_nrules = ev_object.effective_rulesize_eval(tolerance)

            score = score_acc + score_rules_size * alpha + score_nrules * beta
        else:
            score = 0.0 """
            
        return score
    