import abc
import numbers
import copy

from _pytest.cacheprovider import cache
import numpy as np
try:
    from . import fuzzy_sets as fs
    from . import fuzzy_variable as fv
    # from . import centroid
except ImportError:
    import fuzzy_sets as fs
    import fuzzy_variable as fv
    # import centroid

modifiers_names = {0.5: 'Somewhat', 1.0: '', 1.3: 'A little', 1.7: 'Slightly', 2.0: 'Very', 3.0: 'Extremely', 4.0: 'Very very'}


class RuleError(Exception):
    """
    Exception raised when a fuzzy rule is incorrectly defined or invalid.
    
    This exception is used throughout the rules module to indicate various
    rule-related errors such as invalid antecedent specifications, inconsistent
    membership functions, or malformed rule structures.
    
    Attributes:
        message (str): Human-readable description of the error
        
    Example:
        >>> try:
        ...     rule = RuleSimple([-1, -1, -1], 0)  # Invalid: all don't-care antecedents
        ... except RuleError as e:
        ...     print(f"Rule error: {e}")
    """

    def __init__(self, message: str) -> None:
        """
        Initialize the RuleError exception.

        Args:
            message (str): Descriptive error message explaining what went wrong
        """
        super().__init__(message)


class RuleSimple():
    """
    Simplified Rule Representation for Optimized Computation.
    
    This class represents fuzzy rules in a simplified format optimized for
    computational efficiency in rule base operations. It uses integer encoding
    for antecedents and consequents to minimize memory usage and speed up
    rule evaluation processes.
    
    Attributes:
        antecedents (list[int]): Integer-encoded antecedents where:
            - -1: Variable not used in the rule
            - 0-N: Index of the linguistic variable used for the ith input
        consequent (int): Integer index of the consequent linguistic variable
        modifiers (np.array): Optional modifiers for rule adaptation
        
    Example:
        >>> # Rule: IF x1 is Low AND x2 is High THEN y is Medium
        >>> # Assuming Low=0, High=1, Medium=1
        >>> rule = RuleSimple([0, 1], consequent=1)
        >>> print(rule.antecedents)  # [0, 1]
        >>> print(rule.consequent)  # 1
        
    Note:
        This simplified representation is designed for high-performance
        rule evaluation in large rule bases where memory and speed are critical.
    """

    def __init__(self, antecedents: list[int], consequent: int = 0, modifiers: np.array = None) -> None:
        """
        Creates a rule with the given antecedents and consequent.
        
        Args:
            antecedents (list[int]): List of integers indicating the linguistic 
                variable used for each input (-1 for unused variables)
            consequent (int, optional): Integer indicating the linguistic variable 
                used for the consequent. Defaults to 0.
            modifiers (np.array, optional): Array of modifier values for rule adaptation.
                Defaults to None.
                
        Example:
            >>> # Create a rule with two antecedents and one consequent
            >>> rule = RuleSimple([0, 2, -1], consequent=1)  # x1=0, x2=2, x3=unused, y=1
        """
        self.antecedents = list(map(int, antecedents))
        self.consequent = int(consequent)
        self.modifiers = modifiers

    def __getitem__(self, ix):
        """
        Returns the antecedent value for the given index.
        
        Args:
            ix (int): Index of the antecedent to return
            
        Returns:
            int: The antecedent value at the specified index
            
        Example:
            >>> rule = RuleSimple([0, 1, 2])
            >>> print(rule[1])  # 1
        """
        return self.antecedents[ix]

    def __setitem__(self, ix, value):
        """
        Sets the antecedent value for the given index.
        
        Args:
            ix (int): Index of the antecedent to set
            value (int): Value to set at the specified index
            
        Example:
            >>> rule = RuleSimple([0, 1, 2])
            >>> rule[1] = 3  # Change second antecedent to 3
        """
        self.antecedents[ix] = value

    def __str__(self):
        """
        Returns a string representation of the rule.
        
        Returns:
            str: Human-readable string representation of the rule
            
        Example:
            >>> rule = RuleSimple([0, 1], consequent=2)
            >>> print(rule)  # Rule: antecedents: [0, 1] consequent: 2
        """
        aux = 'Rule: antecedents: ' + str(self.antecedents) + ' consequent: ' + str(self.consequent)

        try:
            if self.modifiers is not None:
                aux += ' modifiers: ' + str(self.modifiers)
        except AttributeError:
            pass
        
        try:
            aux += ' score: ' + str(self.score)
        except AttributeError:
            pass

        try:
            aux += ' weight: ' + str(self.weight)
        except AttributeError:
            pass
    
        try:
            aux += ' accuracy: ' + str(self.accuracy)
        except AttributeError:
            pass

        try:
            aux += ' p-value class structure: ' + str(self.p_value_class_structure)
        except AttributeError:
            pass

        try:
            aux += ' p-value feature coalitions: ' + str(self.p_value_feature_coalitions)
        except AttributeError:
            pass

        try:
            aux += ' p-value bootstrapping membership validation: ' + str(self.boot_p_value)
        except AttributeError:
            pass

        try:
            aux += ' bootstrapping confidence conf. interval: ' + str(self.boot_confidence_interval)
        except AttributeError:
            pass

        try:
            aux += ' bootstrapping support conf. interval: ' + str(self.boot_support_interval)
        except AttributeError:
            pass
            

        return aux

    def __len__(self):
        """
        Returns the number of antecedents in the rule.
        
        Returns:
            int: Number of antecedents in the rule
            
        Example:
            >>> rule = RuleSimple([0, 1, 2])
            >>> print(len(rule))  # 3
        """
        return len(self.antecedents)

    def __eq__(self, other: 'RuleSimple'):
        """
        Returns True if the two rules are equal.
        
        Args:
            other (RuleSimple): Another rule to compare with
            
        Returns:
            bool: True if rules have identical antecedents and consequent
            
        Example:
            >>> rule1 = RuleSimple([0, 1], consequent=2)
            >>> rule2 = RuleSimple([0, 1], consequent=2)
            >>> print(rule1 == rule2)  # True
        """
        return self.antecedents == other.antecedents and self.consequent == other.consequent

    def __hash__(self):
        '''
        Returns the hash of the rule.
        '''
        return hash(str(self))


class RuleBase():
    '''
    Class optimized to work with multiple rules at the same time. Right now supports only one consequent. 
    (Solution: use one rulebase per consequent to study)
    '''

    def __init__(self, antecedents: list[fv.FuzzyVariable], rules: list[RuleSimple], consequent: fv.FuzzyVariable, tnorm=np.prod) -> None:
        '''
        Creates a rulebase with the given antecedents, rules and consequent.

        :param antecedents: list of fuzzy sets.
        :param rules: list of rules.
        :param consequent: fuzzy set.
        :param tnorm: t-norm to use in the inference process.
        '''
        self.rules = rules
        self.antecedents = antecedents
        self.consequent = consequent
        self.tnorm = tnorm


    def get_rules(self) -> list[RuleSimple]:
        '''
        Returns the list of rules in the rulebase.
        '''
        return self.rules


    def add_rule(self, new_rule: RuleSimple):
        '''
        Adds a new rule to the rulebase.
        :param new_rule: rule to add.
        '''
        self.rules.append(new_rule)


    def add_rules(self, new_rules: list[RuleSimple]):
        '''
        Adds a list of new rules to the rulebase.

        :param new_rules: list of rules to add.
        '''
        self.rules += new_rules


    def remove_rule(self, ix: int) -> None:
        '''
        Removes the rule in the given index.
        :param ix: index of the rule to remove.
        '''
        del self.rules[ix]


    def remove_rules(self, delete_list: list[int]) -> None:
        '''
        Removes the rules in the given list of indexes.

        :param delete_list: list of indexes of the rules to remove.
        '''
        self.rules = [rule for ix, rule in enumerate(
            self.rules) if ix not in delete_list]


    def get_rulebase_matrix(self):
        '''
        Returns a matrix with the antecedents values for each rule.
        '''
        res = np.zeros((len(self.rules), len(self.antecedents)))

        for ix, rule in enumerate(self.rules):
            res[ix] = rule

        return res


    def print_rules(self, return_rules:bool=False, bootstrap_results:bool=True) -> None:
        '''
        Print the rules from the rule base.

        :param return_rules: if True, the rules are returned as a string.
        '''
        all_rules = ''
        for ix, rule in enumerate(self.rules):
            str_rule = generate_rule_string(rule, self.antecedents, bootstrap_results)
            
            all_rules += str_rule + '\n'

        if not return_rules:
            print(all_rules)
        else:
            return all_rules


    def compute_antecedents_memberships(self, x: np.array) -> np.ndarray:
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


    @abc.abstractmethod
    def inference(self, x: np.array) -> np.array:
        '''
        Computes the fuzzy output of the fl inference system.

        Return an array in shape samples x 2 (last is iv dimension)

        :param x: array with the values of the inputs.
        :return: array with the memberships of the consequents for each rule.
        '''
        raise NotImplementedError


    @abc.abstractmethod
    def forward(self, x: np.array) -> np.array:
        '''
        Computes the deffuzified output of the fl inference system.

        Return a vector of size (samples, )

        :param x: array with the values of the inputs.
        :return: array with the deffuzified output.
        '''
        raise NotImplementedError


    @abc.abstractmethod
    def fuzzy_type(self) -> fs.FUZZY_SETS:
        '''
        Returns the corresponding type of the RuleBase using the enum type in the fuzzy_sets module.

        :return: the type of fuzzy set used in the RuleBase.
        '''
        raise NotImplementedError


    def __len__(self):
        '''
        Returns the number of rules in the rule base.
        '''
        return len(self.rules)


    def __getitem__(self, item: int) -> RuleSimple:
        '''
        Returns the corresponding rulebase.

        :param item: index of the rule.
        :return: the corresponding rule.
        '''
        return self.rules[item]


    def __setitem__(self, key: int, value: RuleSimple) -> None:
        '''
        Set the corresponding rule.

        :param key: index of the rule.
        :param value: new rule.
        '''
        self.rules[key] = value
    

    def __iter__(self):
        '''
        Returns an iterator for the rule base.
        '''
        return iter(self.rules)
    

    def __eq__(self, other):
        '''
        Returns True if the two rule bases are equal.
        '''
        return self.rules == other.rules
    

    def __hash__(self):
        '''
        Returns the hash of the rule base.
        '''
        return hash(str(self))
    

    def __add__(self, other):
        '''
        Adds two rule bases.
        '''
        return RuleBase(self.antecedents, self.rules + other.rules, self.consequent, self.tnorm)
    

    def n_linguistic_variables(self) -> int:
        '''
        Returns the number of linguistic variables in the rule base.
        '''
        return [len(amt) for amt in self.antecedents]


    def copy(self):
        '''
        Creates a copy of the RuleBase.
        
        :param deep: if True, creates a deep copy. If False, creates a shallow copy.
        :return: a copy of the RuleBase.
        '''
        # Deep copy all components
        copied_rules = copy.deepcopy(self.rules)
        copied_antecedents = copy.deepcopy(self.antecedents)
        copied_consequent = copy.deepcopy(self.consequent) if self.consequent is not None else None
        
        # Create new instance based on the type
        if isinstance(self, RuleBaseT1):
            return RuleBaseT1(copied_antecedents, copied_rules, copied_consequent, self.tnorm)
        elif isinstance(self, RuleBaseT2):
            return RuleBaseT2(copied_antecedents, copied_rules, copied_consequent, self.tnorm)
        elif isinstance(self, RuleBaseGT2):
            return RuleBaseGT2(copied_antecedents, copied_rules, copied_consequent, self.tnorm)
        else:
            # Base RuleBase class
            new_rb = RuleBase.__new__(RuleBase)
            new_rb.rules = copied_rules
            new_rb.antecedents = copied_antecedents
            new_rb.consequent = copied_consequent
            new_rb.tnorm = self.tnorm
            return new_rb


class RuleBaseT1(RuleBase):
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


    def inference(self, x: np.array) -> float:
        '''
        Computes the output of the t1 inference system.

        Return an array in shape samples.

        :param x: array with the values of the inputs.
        :return: array with the output of the inference system for each sample.
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
            
            aggregated_consequents = fs.union(cut_consequents)
            p_x, p_y = aggregated_consequents

            x_crisp = fs.centroid_defuzzification(p_x, p_y)
            output.append(x_crisp)
        
        return np.array(output)


    def forward(self, x: np.array) -> np.array:
        '''
        Same as inference() in the t1 case.

        Return a vector of size (samples, )

        :param x: array with the values of the inputs.
        :return: array with the deffuzified output for each sample.
        '''
        return self.inference(x)


    def fuzzy_type(self) -> fs.FUZZY_SETS:
        '''
        Returns the correspoing type of the RuleBase using the enum type in the fuzzy_sets module.

        :return: the corresponding fuzzy set type of the RuleBase.
        '''
        return fs.FUZZY_SETS.t1



def generate_rule_string(rule: RuleSimple, antecedents: list, bootstrap_results: bool=True) -> str:
    '''
    Generates a string with the rule.

    :param rule: rule to generate the string.
    :param antecedents: list of fuzzy variables.
    :param modifiers: array with the modifiers for the antecedents.
    '''
    def format_p_value(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**' 
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'
        
    initiated = False
    str_rule = 'IF '
    for jx, antecedent in enumerate(antecedents):
        keys = antecedent.linguistic_variable_names()

        if rule[jx] != -1:
            if not initiated:
                initiated = True
            else:
                str_rule += ' AND '

            str_rule += str(antecedent.name) + ' IS ' + str(keys[rule[jx]])
            
            try:
                relevant_modifier = rule.modifiers[jx]
                if relevant_modifier != 1:
                    if relevant_modifier in modifiers_names.keys():
                        str_mod = modifiers_names[relevant_modifier]
                    else:
                        str_mod = str(relevant_modifier)

                    str_rule += ' (MOD ' + str_mod + ')'
            except AttributeError:
                pass
            except TypeError:
                pass


    try:
        score = rule.score if antecedents[0].fuzzy_type() == fs.FUZZY_SETS.t1 else np.mean(rule.score)
        str_rule += ' WITH DS ' + str(score)

        # If the classification scores have been computed, print them.
        try:
            str_rule += ', ACC ' + str(rule.accuracy)

        except AttributeError:
            pass
        # Check if they have weights
        try:
            str_rule += ', WGHT ' + str(rule.weight)
        except AttributeError:
            pass
    except AttributeError:
        try:
            str_rule += ' THEN consequent vl is ' + str(rule.consequent)
        except AttributeError:
            pass
    
    if bootstrap_results:
        try:
            p_value_class_structure = rule.p_value_class_structure
            p_value_feature_coalition = rule.p_value_feature_coalitions


            pvc = format_p_value(p_value_class_structure)
            pvf = format_p_value(p_value_feature_coalition)

            str_rule += ' (p-value Permutation Class Structure: ' + pvc + ', Feature Coalition: ' + pvf 
            p_value_bootstrap = rule.boot_p_value

            pbs = format_p_value(p_value_bootstrap)
            str_rule += ' Membership Validation: ' + pbs
            str_rule += ')'

            str_rule += ' Confidence Interval: ' + str(rule.boot_confidence_interval)
            str_rule += ' Support Interval: ' + str(rule.boot_support_interval)
        except AttributeError:
            pass
    return str_rule