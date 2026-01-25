import numpy as np
from ex_fuzzy.rules import RuleSimple, RuleBase

from ex_fuzzy_reg import fuzzy_sets as fs
from ex_fuzzy_reg import fuzzy_variable as fv


# modifiers_names = {0.5: 'Somewhat', 1.0: '', 1.3: 'A little', 1.7: 'Slightly', 2.0: 'Very', 3.0: 'Extremely', 4.0: 'Very very'}


class RuleBaseRegT1(RuleBase):
    '''
    Class optimized to work with multiple rules at the same time. Supports only one consequent.
    (Use one rulebase per consequent to study classification problems. Check MasterRuleBase class for more documentation)

    This class supports t1 fs.
    '''

    def __init__(self, antecedents: list[fv.FuzzyVariable], rules: list[RuleSimple], consequent: fv.FuzzyVariable = None, tnorm=np.prod) -> None:
        '''
        Constructor of the RuleBaseT1 class.

        :param antecedents: list of fuzzy variables that are the antecedents of the rules.
        :param rules: list of rules.
        :param consequent: fuzzy variable that is the consequent of the rules. ONLY on regression problems.
        :param tnorm: t-norm used to compute the fuzzy output.
        '''
        self.rules = rules
        self.antecedents = antecedents
        self.consequent = consequent
        self.tnorm = tnorm


    def compute_antecedents_memberships(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns a list of of dictionaries that contains the memberships for each x value to the ith antecedents, nth linguistic variable.
        x must be a vector (only one sample)

        :param x: vector with the values of the inputs.
        :return: a list with the antecedent truth values for each one. Each list is comprised of a list with n elements, where n is the number of linguistic variables in each variable.
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


    def inference(self, X: np.ndarray) -> np.ndarray:
        '''
        Computes the output of the t1 inference system.

        Return an array in shape samples.

        :param x: array with the values of the inputs.
        :return: array with the output of the inference system for each sample.
        '''
        output = []

        for sample in X:
            antecedents_memberships = self.compute_antecedents_memberships(sample)
            cut_heights = self.compute_cut_heights(antecedents_memberships)

            cut_consequents = []
            for idx, rule in enumerate(self.rules):
                rule_consequent = rule.consequent
                cut_height = cut_heights[idx]

                cut_consequents.append(fs.cut(self.consequent.__getitem__(rule_consequent), cut_height))
            
            aggregated_consequents = fs.union(cut_consequents)
            p_x, p_y = aggregated_consequents

            x_crisp = fs.centroid_defuzzification(p_x, p_y)
            output.append(x_crisp)
        
        return np.array(output).reshape(-1, 1)


    def forward(self, X: np.ndarray) -> np.ndarray:
        '''
        Same as inference() in the t1 case.

        Return a vector of size (samples, )

        :param x: array with the values of the inputs.
        :return: array with the deffuzified output for each sample.
        '''
        return self.inference(X)


    def fuzzy_type(self) -> fs.FUZZY_SETS:
        '''
        Returns the correspoing type of the RuleBase using the enum type in the fuzzy_sets module.

        :return: the corresponding fuzzy set type of the RuleBase.
        '''
        return fs.FUZZY_SETS.t1


class ConsequentTSK:
    def __init__(self, params: np.ndarray) -> None:
       self.params = params


    def compute_consequent(self, x: np.ndarray):
        if len(self.params) == 1:
            return self.params[0]

        return np.dot(x, params) 


    