from typing import Generator
import numpy as np

from ex_fuzzy_reg.fuzzy_sets import FS, FUZZY_SETS


class FuzzyVariable():
    """
    Fuzzy Variable Container and Management Class.
    
    This class represents a fuzzy variable composed of multiple fuzzy sets
    (linguistic variables). It provides methods to compute memberships across
    all fuzzy sets and manage the linguistic terms of the variable.
    
    Attributes:
        linguistic_variables (list): List of fuzzy sets that define the variable
        name (str): Name of the fuzzy variable
        units (str): Units of measurement (optional, for display purposes)
        fs_type (FUZZY_SETS): Type of fuzzy sets (t1 or t2)
        
    Example:
        >>> # Create fuzzy sets for temperature
        >>> low_temp = gaussianFS([15, 5], "Low", 100)
        >>> medium_temp = gaussianFS([25, 5], "Medium", 100)
        >>> high_temp = gaussianFS([35, 5], "High", 100)
        >>> 
        >>> # Create fuzzy variable
        >>> temp_var = fuzzyVariable("Temperature", [low_temp, medium_temp, high_temp], "°C")
        >>> memberships = temp_var.membership([20, 25, 30])
        >>> print(memberships.shape)  # (3, 3) - 3 inputs, 3 fuzzy sets
        
    Note:
        All fuzzy sets in the variable must be of the same type (t1 or t2).
    """

    def __init__(self, name: str, fuzzy_sets: list[FS], units: str = None) -> None:
        """
        Creates a fuzzy variable with the specified fuzzy sets.
        
        Args:
            name (str): Name of the fuzzy variable
            fuzzy_sets (list[FS]): List of fuzzy sets that comprise the linguistic variables
            units (str, optional): Units of the fuzzy variable for display purposes
            
        Raises:
            ValueError: If fuzzy_sets is empty or contains mixed types
            
        Example:
            >>> fs1 = gaussianFS([0, 1], "Low", 10)
            >>> fs2 = gaussianFS([5, 1], "High", 10)
            >>> var = fuzzyVariable("Speed", [fs1, fs2], "m/s")
        """
        self.linguistic_variables = []
        self.name = name
        self.units = units

        if len(fuzzy_sets) == 0:
            raise ValueError("Specified fuzzy sets list is empty.")
        
        types = {fuzzy_set.type() for fuzzy_set in fuzzy_sets}
        if len(types) > 1:
            raise ValueError("All fuzzy sets must be of the same type.")

        for ix, fuzzy_set in enumerate(fuzzy_sets):
            self.linguistic_variables.append(fuzzy_set)

        self.fs_type = self.linguistic_variables[0].type()


    def append(self, fuzzy_set: FS) -> None:
        '''
        Appends a fuzzy set to the fuzzy variable.

        :param fuzzy_set: FS. Fuzzy set to append.
        '''
        self.linguistic_variables.append(fuzzy_set)


    def linguistic_variable_names(self) -> list:
        '''
        Returns the name of the linguistic variables.

        :return: list of strings.
        '''
        return [fuzzy_set.name for fuzzy_set in self.linguistic_variables]


    def get_linguistic_variables(self) -> list[FS]:
        '''
        Returns the name of the linguistic variables.

        :return: list of strings.
        '''
        return self.linguistic_variables


    def compute_memberships(self, x: np.array) -> list:
        '''
        Computes the membership to each of the FS in the fuzzy variables.

        :param x: numeric value or array. Computes the membership to each of the FS in the fuzzy variables.
        :return: list of floats. Membership to each of the FS in the fuzzy variables.
        '''
        res = []
        try:
            x = np.clip(x, self.linguistic_variables[0].domain[0], self.linguistic_variables[0].domain[1])
        except Exception as e:
            pass

        for fuzzy_set in self.linguistic_variables:
            res.append(fuzzy_set.membership(x))

        return np.array(res)


    def domain(self) -> list[float]:
        '''
        Returns the domain of the fuzzy variable.

        :return: list of floats.
        '''
        return self.linguistic_variables[0].domain


    def fuzzy_type(self) -> FUZZY_SETS:
        '''
        Returns the fuzzy type of the domain

        :return: the type of the fuzzy set present in the fuzzy variable.
        '''
        return self.fs_type
    

    def _permutation_validation(self, mu_A:np.array, mu_B:np.array, p_value_need:float=0.05) -> bool:
        '''
        Validates the fuzzy variable using permutation test to check if the fuzzy sets are statistically different.

        :param mu_A: np.array. Memberships of the first fuzzy set.
        :param mu_B: np.array. Memberships of the second fuzzy set.
        :return: bool. True if the fuzzy sets are statistically different, False otherwise.
        '''
        from scipy.stats import permutation_test

        # Perform permutation test
        result = permutation_test((mu_A, mu_B), lambda x, y: np.mean(np.abs(x - y)), n_resamples=1000)
        statistic, p_value, null_distribution = result.statistic, result.pvalue, result.null_distribution

        return p_value < p_value_need
    

    def validate(self, X, verbose:bool=False) -> bool:
        '''
        Validates the fuzzy variable. Checks that all the fuzzy sets have the same type and domain.

        :param X: np.array. Input data to validate the fuzzy variable.
        :return: bool. True if the fuzzy variable is valid, False otherwise.
        '''
        if len(self.linguistic_variables) == 0:
            return False
        
        # Get the fuzzy sets memberships
        memberships = self.compute_memberships(X)
        memberships = np.array(memberships)

        cond1 = True
        # Property 1: Only one of the fuzzy sets memberships can be 1 at the same time
        if np.equal(memberships, 1).sum(axis=0).max() > 1:
            cond1 = False
        if not cond1 and verbose:
            print('Property 1 violated: More than one fuzzy set has a membership of 1 at the same time.')

        # Property 2: All fuzzy sets are fuzzy numbers is fullfilled if they are trapezoidal or gaussian
        cond2 = all([fuzzy_set.shape() in ['trapezoidal', 'triangular', 'gaussian'] for fuzzy_set in self.linguistic_variables])
        if not cond2 and verbose:
            print('Property 2 violated: Not all fuzzy sets are fuzzy numbers (trapezoidal or gaussian).')

        # Property 3: At least one fuzzy set must non zero in every point of the domain
        cond3 = np.any(memberships > 0, axis=0).all()
        if not cond3 and verbose:
            print('Property 3 violated: At least one fuzzy set must be non-zero in every point of the domain.')

        # Property 4: Given any two points of the domain, if a < b, the membership f_n+1(b)>f_n+1(a) can only hold if f_n(a)>f_n(b). So, a fuzzy set can only grow if the previous fuzzy set is decreasing.
        cond4 = True
        for i in range(len(self.linguistic_variables) - 1):
            if np.any(memberships[i, :] < memberships[i + 1, :]) and np.any(memberships[i, :] > memberships[i + 1, :]):
                cond4 = False
                break
        
        if not cond4 and verbose:
            print('Property 4 violated: Fuzzy sets must be non-decreasing in the domain. If a fuzzy set grows, the previous fuzzy set must be decreasing.')
        
        # Property 5: The smallest fuzzy set must be the first one and the biggest fuzzy set must be the last one. The smallest should start at the left of the domain and the biggest should end at the right of the domain.
        cond5 = (self.compute_memberships(self[0].domain[0])[0] >= 0.99) and (self.compute_memberships(self[0].domain[1])[-1] >= 0.99)
        if not cond5 and verbose:
            print('Property 5 violated: The smallest fuzzy set must be the first one and the biggest fuzzy set must be the last one. The smallest should start at the left of the domain and the biggest should end at the right of the domain.')

        # Property 6: The population of the fuzzy sets-induced memberships must be statistically different from each other
        cond6 = True
        if len(self.linguistic_variables) > 1:
            for i in range(len(self.linguistic_variables) - 1):
                if not self._permutation_validation(memberships[i, :], memberships[i + 1, :], p_value_need=0.05):
                    cond6 = False
                    break
        if not cond6 and verbose:
            print('Property 6 violated: The fuzzy sets must be statistically different from each other. Use permutation test to check this. (' + str(i) + ',' + str(i+1) + ')')

        valid = cond1 and cond2 and cond3 and cond4 and cond5 and cond6

        if verbose and valid:
            print('Fuzzy variable ' + self.name + ' is valid.')

        return valid


    def __str__(self) -> str:
        '''
        Returns the name of the fuzzy variable, its type and its parameters.
        
        :return: string.
        '''
        return f'{self.name} ({self.fs_type.name}) - {self.linguistic_variable_names()}'


    def __getitem__(self, item) -> FS:
        '''
        Returns the corresponding fs.

        :param item: int. Index of the FS.
        :return: FS. The corresponding FS.
        '''
        return self.linguistic_variables[item]


    def __setitem__(self, item: int, elem: FS) -> None:
        '''
        Sets the corresponding fs.

        :param item: int. Index of the FS.
        :param elem: FS. The FS to set.
        '''
        self.linguistic_variables[item] = elem
    

    def __iter__(self) -> Generator[FS, None, None]:
        '''
        Returns the corresponding fs.

        :param item: int. Index of the FS.
        :return: FS. The corresponding FS.
        '''
        for fuzzy_set in self.linguistic_variables:
            yield fuzzy_set


    def __len__(self) -> int:
        '''
        Returns the number of linguistic variables.

        :return: int. Number of linguistic variables.
        '''
        return len(self.linguistic_variables)
    

    def __call__(self, x: np.array) -> list:
        '''
        Computes the membership to each of the FS in the fuzzy variables.

        :param x: numeric value or array.
        :return: list of floats. Membership to each of the FS in the fuzzy variables.
        '''
        return self.compute_memberships(x)