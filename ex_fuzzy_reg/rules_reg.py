from tkinter import W
import numpy as np
from ex_fuzzy.rules import RuleSimple, RuleBase

from ex_fuzzy_reg import fuzzy_sets as fs
from ex_fuzzy_reg import fuzzy_variable as fv


# modifiers_names = {0.5: 'Somewhat', 1.0: '', 1.3: 'A little', 1.7: 'Slightly', 2.0: 'Very', 3.0: 'Extremely', 4.0: 'Very very'}

# TODO: add tests and document for this module
class RuleBaseRegT1(RuleBase):
    '''
    Class optimized to work with multiple rules at the same time. Supports only one consequent.
    (Use one rulebase per consequent to study classification problems. Check MasterRuleBase class for more documentation)

    This class supports t1 fs.
    '''
    def __init__(self, antecedents: list[fv.FuzzyVariable], rules: list[RuleSimple], consequent: fv.FuzzyVariable = None, tnorm = np.prod) -> None:
        '''
        Constructor of the RuleBaseT1 class.

        Args:
            antecedents (list[FuzzyVariable]): list of fuzzy variables that are the antecedents of the rules.
            rules (list[RuleSimple]): list of rules.
            consequent (FuzzyVariable): fuzzy variable that is the consequent of the rules. ONLY on regression problems.
            tnorm: t-norm used to compute the fuzzy output.
        '''
        self.rules = rules
        self.antecedents = antecedents
        self.consequent = consequent
        self.tnorm = tnorm


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
            elif self.fuzzy_type() == fs.FUZZY_SETS.t2:
                return np.zeros((x.shape[0], 1, 2))
            elif self.fuzzy_type() == fs.FUZZY_SETS.gt2:
                return np.zeros((x.shape[0], len(self.alpha_cuts), 2))


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
                cut_height = np.min(memberships_for_rule)
            else:
                cut_height = 0.0  

            cut_heights.append(cut_height)
        
        return np.array(cut_heights)


    def inference(self, x: np.ndarray) -> np.ndarray:
        '''
        Computes the output of the t1 inference system.

        Args:
            x: array with the values of the inputs.
        
        Returns: 
            np.ndarray: array with the output of the inference system for each sample.
        '''
        output = []

        for sample in x:
            antecedents_memberships = self.compute_antecedents_memberships(sample)
            cut_heights = self.compute_cut_heights(antecedents_memberships)

            cut_consequents = []
            for idx, rule in enumerate(self.rules):
                rule_consequent = rule.consequent
                cut_height = cut_heights[idx]

                cut_consequents.append(fs.cut(self.consequent.__getitem__(rule_consequent), cut_height))
            
            aggregated_consequents = fs.trapezoidal_union(cut_consequents)
            p_x, p_y = aggregated_consequents

            x_crisp = fs.centroid_defuzzification(p_x, p_y)
            output.append(x_crisp)
        
        return np.array(output).reshape(-1, 1)


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


    # TODO: print_rules 
    def print_rules(self) -> None:
        pass
