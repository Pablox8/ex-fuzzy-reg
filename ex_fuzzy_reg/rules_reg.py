"""
Fuzzy rule base implementations for regression.

Contains:
    - RuleBaseRegT1: Type-1 Mamdani rule base for regression.
    - RuleBaseRegTSK: Takagi-Sugeno-Kang rule base for regression.
    - Helper functions: compute_antecedents_memberships, compute_antecedents_memberships_batch.
"""

import numpy as np
from ex_fuzzy.rules import RuleSimple, RuleBase

from ex_fuzzy_reg import fuzzy_sets as fs
from ex_fuzzy_reg import fuzzy_variable as fv


# modifiers_names = {0.5: 'Somewhat', 1.0: '', 1.3: 'A little', 1.7: 'Slightly', 2.0: 'Very', 3.0: 'Extremely', 4.0: 'Very very'}

def compute_antecedents_memberships(antecedents: list[fv.FuzzyVariable], x: np.ndarray) -> np.ndarray:
    """
    Compute membership degrees for input values across all fuzzy variables.
    
    This function calculates the membership degrees of input values for each linguistic
    variable in the antecedents. It returns a structured representation that can be
    used for efficient rule evaluation and inference.
    
    Args:
        antecedents (list[fs.fuzzyVariable]): List of fuzzy variables representing
            the antecedents (input variables) of the fuzzy system
        x (np.ndarray): Input vector with values for each antecedent variable.
            Shape should be (n_samples, n_variables) or (n_variables,) for single sample
            
    Returns: 
        np.ndarray: a list with the antecedent truth values for each one. Each list is comprised of a list with n elements, where n is the number of linguistic variables in each variable.
    """
    x = np.array(x)
    cache_antecedent_memberships = []

    for ix, antecedent in enumerate(antecedents):
        cache_antecedent_memberships.append(
            antecedent.compute_memberships(x[:, ix]))

    return np.array(cache_antecedent_memberships)

def compute_antecedents_memberships_batch(antecedents: list[fv.FuzzyVariable], X: np.ndarray) -> np.ndarray:
    # X shape: (n_samples, n_vars)
    # Output shape:  (n_samples, n_vars, n_labels)
    result = np.stack([
        antecedent.compute_memberships(X[:, ix]).T  # (n_samples, n_labels)
        for ix, antecedent in enumerate(antecedents)
    ], axis=1)
    return result

# TODO: add tests and document for this module
class RuleBaseRegT1(RuleBase):
    '''
    Rule base for Type-1 Mamdani regression.
    
    Supports multiple rules with a single fuzzy consequent variable.
    Evaluation uses a configurable t-norm (default: minimum) across antecedents.
    '''
    def __init__(self, antecedents: list[fv.FuzzyVariable], rules: list[RuleSimple], consequent: fv.FuzzyVariable, tnorm = np.min) -> None:
        '''
        Constructor of the RuleBaseT1 class.

        Args:
            antecedents (list[FuzzyVariable]): list of fuzzy variables that are the antecedents of the rules.
            rules (list[RuleSimple]): list of rules.
            consequent (FuzzyVariable): fuzzy variable that is the consequent of the rules. 
            tnorm: t-norm used to compute the fuzzy output.
        
        Note: 
            super().__init__() is intentionally not called. RuleBase expects ex-fuzzy's fuzzyVariable; this class uses
            ex_fuzzy_reg's FuzzyVariable. We inherit only for method access, not for constructor behaviour.
        '''
        self.rules = rules
        self.antecedents = antecedents
        self.consequent = consequent
        self.tnorm = tnorm # tnorm must have axis argument like numpy min and prod


    def compute_antecedents_memberships(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns a list of of dictionaries that contains the memberships for each x value to the ith antecedents, nth linguistic variable.

        Args:
            x: vector (only one sample) with the values of the inputs. 
        
        Returns: 
            np.ndarray: a list with the antecedent truth values for each one. Each list is comprised of a list with n elements, where n is the number of linguistic variables in each variable.
        '''
        if len(self.rules) > 0:
            cache_antecedent_memberships = []

            for ix, antecedent in enumerate(self.antecedents):
                # Check if x is pandas 
                if hasattr(x, 'values'):
                    x = x.values

                cache_antecedent_memberships.append(antecedent.compute_memberships(x[ix]))

            return np.array(cache_antecedent_memberships)

        else:
            if self.fuzzy_type() == fs.FUZZY_SETS.t1:
                return np.zeros((x.shape[0], 1))
            # NOT USED FOR NOW
            # elif self.fuzzy_type() == fs.FUZZY_SETS.t2:
            #     return np.zeros((x.shape[0], 1, 2))
            # elif self.fuzzy_type() == fs.FUZZY_SETS.gt2:
            #     return np.zeros((x.shape[0], len(self.alpha_cuts), 2))
    
    def compute_antecedents_memberships_batch(self, X: np.ndarray) -> np.ndarray:
        # X shape: (n_samples, n_vars)
        # Output shape:  (n_samples, n_vars, n_labels)
        result = np.stack([
            antecedent.compute_memberships(X[:, ix]).T  # (n_samples, n_labels)
            for ix, antecedent in enumerate(self.antecedents)
        ], axis=1)
        return result


    def compute_cut_heights(self, antecedents_memberships: np.ndarray) -> np.ndarray:
        cut_heights = []
        
        for idx, rule in enumerate(self.rules):
            antecedent_indices = rule.antecedents            
            memberships_for_rule = []
            
            for var_idx, mf_idx in enumerate(antecedent_indices):
                if mf_idx != -1:
                    membership_value = antecedents_memberships[var_idx, mf_idx]
                    memberships_for_rule.append(membership_value)
            
            if memberships_for_rule:
                cut_height = self.tnorm(memberships_for_rule)
            else:
                cut_height = 0.0  

            cut_heights.append(cut_height)
        
        return np.array(cut_heights)

    
    def compute_cut_heights_batch(self, memberships_batch: np.ndarray) -> np.ndarray:
        """
        Args:
            memberships_batch: (n_samples, n_variables, n_mfs)
        Returns:
            (n_samples, n_rules)
        """
        n_samples = memberships_batch.shape[0]
        all_rule_strengths = []

        for rule in self.rules:
            ants = np.array(rule.antecedents)
            valid_vars = np.where(ants != -1)[0]
            valid_mfs = ants[valid_vars]
            
            if valid_vars.size > 0:
                rule_memberships = memberships_batch[:, valid_vars, valid_mfs]
                
                strength = self.tnorm(rule_memberships, axis=1)
            else:
                # Rule with no antecedents fires at 1.0 (identity) or 0.0
                strength = np.ones(n_samples)

            all_rule_strengths.append(strength)

        return np.column_stack(all_rule_strengths)
    

    def inference(self, x: np.ndarray) -> np.ndarray:
        '''
        Computes the output of the t1 inference system.

        Args:
            x: array with the values of the inputs.
        
        Returns: 
            np.ndarray: array with the output of the inference system for each sample.
        '''
        output = []

        # cases when p_x = p_y = []
        consequent_fallback = np.mean([np.mean(lv.membership_parameters) for lv in self.consequent.linguistic_variables])

        for sample in x:
            antecedents_memberships = self.compute_antecedents_memberships(sample)
            cut_heights = self.compute_cut_heights(antecedents_memberships)

            cut_consequents = []
            for idx, rule in enumerate(self.rules):
                rule_consequent = rule.consequent
                cut_height = cut_heights[idx]

                cut_consequents.append(fs.cut(self.consequent.__getitem__(rule_consequent), cut_height))
            
            aggregated_consequents = fs.trapezoidal_triangular_union(cut_consequents)
            p_x, p_y = aggregated_consequents

            if len(p_x) == 0 or np.sum(p_y) == 0:
                x_crisp = consequent_fallback
            else:
                x_crisp = fs.centroid_defuzzification(p_x, p_y)
            output.append(x_crisp)
        
        return np.array(output).reshape(-1, 1)


    def inference_optimized(self, X: np.ndarray, precomputed_truth=None) -> np.ndarray:
        """Vectorized Mamdani inference for regression."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if precomputed_truth is None:
            memberships_batch = self.compute_antecedents_memberships_batch(X)
        else:
            memberships_batch = precomputed_truth
        
        firing_strengths = self.compute_cut_heights_batch(memberships_batch)
        
        consequent_values = np.array([
            self.consequent[rule.consequent].centroid() 
            for rule in self.rules
        ])

        numerator = np.dot(firing_strengths, consequent_values)
        denominator = np.sum(firing_strengths, axis=1)

        fallback = np.mean(consequent_values) # avoid division by zero

        y_pred = np.divide(
            numerator, 
            denominator, 
            out=np.full(numerator.shape, fallback), 
            where=denominator > 0
        )
        return y_pred


    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Same as inference() in the t1 case.

        Return a vector of size (samples, )

        Args:
            x: array with the values of the inputs.
        
        Returns: 
            np.ndarray: array with the deffuzified output for each sample.
        '''
        return self.inference(x)


    def fuzzy_type(self) -> fs.FUZZY_SETS:
        '''
        Returns the correspoing type of the RuleBase using the enum type in the fuzzy_sets module.

        Returns: 
            FUZZY_SETS: the corresponding fuzzy set type of the RuleBase.
        '''
        return fs.FUZZY_SETS.t1
    

    def purge_empty_rules(self) -> None:
        '''
        Remove rules where every antecedent is -1.
        
        These rules fire at strength 1.0 for every sample because the t-norm
        neutral element fills each don't-care slot, corrupting every prediction.
        '''
        self.rules = [
            rule for rule in self.rules
            if any(a != -1 for a in rule.antecedents)
        ]


class ConsequentTSK:
    def __init__(self, params: np.ndarray) -> None:
       self.params = params


    @property
    def order(self):
        return 0 if len(self.params) == 1 else 1


    def compute_consequent(self, x: np.ndarray) -> np.ndarray:
        if self.order == 0:
            return self.params[0]

        return np.dot(x, self.params[1:]) + self.params[0]


class RuleSimpleTSK:
    def __init__(self, antecedents: list[int], consequent: ConsequentTSK) -> None:
        self.antecedents = antecedents
        self.consequent = consequent 

    
    def inference(self, x: np.ndarray) -> np.ndarray:
        return self.consequent.compute_consequent(x)
    
    
class RuleBaseRegTSK:
    def __init__(self, antecedents: list[fv.FuzzyVariable], rules: list[RuleSimpleTSK], tnorm = np.min) -> None:
        self.rules = rules
        self.antecedents = antecedents
        self.tnorm = tnorm


    def compute_rules_truth_values(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns a list of of dictionaries that contains the memberships for each x value to the ith antecedents, nth linguistic variable.
        x must be a vector (only one sample)

        Args:
            x: vector (only one sample) with the values of the inputs.
        
        Returns: 
            np.ndarray: a list with the antecedent truth values for each one. Each list is comprised of a list with n elements, where n is the number of linguistic variables in each variable.
        '''
        truth_values = []

        for rule in self.rules:
            rule_antecedent_memberships = []

            for idx, ling_var_idx in enumerate(rule.antecedents):
                if ling_var_idx != -1:
                    rule_antecedent_memberships.append(self.antecedents[idx].linguistic_variables[ling_var_idx].membership(x[idx])) 

            truth_values.append(self.tnorm(rule_antecedent_memberships))

        return np.array(truth_values)


    def compute_rules_consequents(self, x: np.ndarray) -> np.ndarray:
        consequents = []

        for rule in self.rules: 
            consequents.append(rule.inference(x)) 

        return np.array(consequents)


    def inference(self, x: np.ndarray) -> np.ndarray:
        output = []

        for sample in x:
            rules_consequents = self.compute_rules_consequents(sample)
            rules_truth_values = self.compute_rules_truth_values(sample)

            x_agg = np.dot(rules_consequents, rules_truth_values) / np.sum(rules_truth_values)
            output.append(x_agg)

        return output


    def print_rules(self, return_rules:bool=False) -> None:
        '''
        Print the rules from the rule base.

        :param return_rules: if True, the rules are returned as a string.
        '''
        all_rules = ''
        for ix, rule in enumerate(self.rules):
            str_rule = generate_tsk_rule_string(rule, self.antecedents)
            
            all_rules += str_rule + '\n'

        if not return_rules:
            print(all_rules)
        else:
            return all_rules

 
def generate_tsk_rule_string(rule: RuleSimpleTSK, antecedents: list) -> str:
    initiated = False
    str_rule = 'IF '

    for jx, antecedent in enumerate(antecedents):
        keys = antecedent.linguistic_variable_names()

        if antecedents[jx] != -1:
            if not initiated:
                initiated = True
            else:
                str_rule += ' AND '
            
            str_rule += str(antecedent.name) + ' IS ' + str(keys[rule.antecedents[jx]])
        
    str_rule += ' THEN y = '
    for i, param in enumerate(rule.consequent.params):
        if i == 0:
            str_rule += str(param) + ' + '
        elif i == len(rule.consequent.params)-1:
            str_rule += str(param) + '*' + str(antecedents[i-1].name)
        else:
            str_rule += str(param) + '*' + str(antecedents[i-1].name) + ' + '
    
    return str_rule

