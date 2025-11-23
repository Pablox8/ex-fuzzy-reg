import enum
from typing import Generator

import numpy as np
import pandas as pd


def _get_torch():
    """Lazy import of torch to avoid expensive imports when not needed."""
    try:
        import torch
        return torch
    except ImportError:
        return None

def _is_torch_tensor(x):
    """Check if x is a torch tensor without importing torch unless necessary."""
    # First check the type name to avoid torch import if not needed
    if type(x).__name__ != 'Tensor':
        return False
    
    # Only import torch if we have a potential tensor
    torch = _get_torch()
    if torch is None:
        return False
    
    return isinstance(x, torch.Tensor)


class FUZZY_SETS(enum.Enum):
    """
    Enumeration defining the types of fuzzy sets supported by the library.
    
    This enum is used throughout the library to specify which type of fuzzy set
    should be created or used in operations.
    
    Attributes:
        t1: Type-1 fuzzy sets with crisp membership functions
        t2: Type-2 interval fuzzy sets with upper and lower membership bounds  
        gt2: General Type-2 fuzzy sets with full secondary membership functions
        
    Example:
        >>> fz_type = FUZZY_SETS.t1
        >>> if fz_type == FUZZY_SETS.t1:
        ...     print("Using Type-1 fuzzy sets")
    """
    t1 = 'Type 1'
    t2 = 'Type 2'
    gt2 = 'General Type 2'


    def __eq__(self, __value: object) -> bool:
        return self.value == __value.value


class FS():
    """
    Base class for Type-1 fuzzy sets (Zadeh fuzzy sets).
    
    This class implements the fundamental Type-1 fuzzy set with crisp membership functions.
    It serves as the base class for more specialized fuzzy set types like triangular,
    gaussian, and categorical fuzzy sets.
    
    Attributes:
        name (str): The linguistic name of the fuzzy set (e.g., "low", "medium", "high")
        membership_parameters (list[float]): Parameters defining the membership function
        domain (list[float]): Two-element list defining the universe of discourse [min, max]
        
    Example:
        >>> fs = FS("medium", [1, 2, 3, 4], [0, 5])  # Trapezoidal fuzzy set
        >>> membership = fs.membership(2.5)
        >>> print(membership)  # Should be 1.0 (fully in the set)
        
    Note:
        This class uses trapezoidal membership functions by default. For other shapes,
        use specialized subclasses like gaussianFS or triangularFS.
    """

    def __init__(self, name: str, membership_parameters: list[float], domain: list[float]=None) -> None:
        """
        Initialize a Type-1 fuzzy set.

        Args:
            name (str): Linguistic name for the fuzzy set
            membership_parameters (list[float]): Four parameters [a, b, c, d] defining 
                the trapezoidal membership function where:
                - a: left foot (membership starts rising from 0)
                - b: left shoulder (membership reaches 1.0)
                - c: right shoulder (membership starts falling from 1.0)
                - d: right foot (membership reaches 0)
            domain (list[float]): Two-element list [min, max] defining the universe
                of discourse for this fuzzy set
                
        Example:
            >>> fs = FS("medium", [2, 3, 7, 8], [0, 10])
            >>> # Creates a trapezoidal set: rises from 2-3, flat 3-7, falls 7-8
        """
        self.name = name
        self.domain = domain
        self.membership_parameters = membership_parameters


    def membership(self, x: np.array) -> np.array:
        """
        Compute membership degrees for input values.

        This method calculates the membership degree(s) for the given input value(s)
        using the membership function defined by this fuzzy set's parameters.

        Args:
            x (np.array): Input value(s) for which to compute membership degrees.
                Can be a single value, list, or numpy array.

        Returns:
            np.array: Membership degree(s) in the range [0, 1]. Shape matches input.

        """
        pass


    def type(self) -> FUZZY_SETS:
        """
        Return the fuzzy set type identifier.

        Returns:
            FUZZY_SETS: The type identifier (FUZZY_SETS.t1 for Type-1 fuzzy sets)
        """
        pass
    

    def __str__(self) -> str:
        '''
        Returns the name of the fuzzy set, its type and its parameters.
        
        :return: string.
        '''
        pass
    

    def shape(self) -> str:
        '''
        Returns the shape of the fuzzy set.

        :return: string.
        '''
        pass


class TrapezoidalFS(FS):
    def __init__(self, name: str, membership_parameters: list[float], domain: list[float], height: float=1.0) -> None:
        super().__init__(name, membership_parameters, domain)
        self.height = height


    def membership(self, x: np.array, epsilon=10E-5) -> np.array:
        a, b, c, d = self.membership_parameters
        h = self.height

        # Special case: a singleton trapezoid
        if a == d:
            # If the y are numpy arrays, we need to use the numpy function
            if isinstance(x, np.ndarray):
                return np.equal(x, a).astype(float)
            if _is_torch_tensor(x):
                torch = _get_torch()
                return torch.eq(x, a).float()
                
        if b == a:
            b += epsilon
        if c == d:
            d += epsilon

        aux1 = h*(x - a) / (b - a)
        aux2 = -h*(x - d) / (d - c)
        
        if _is_torch_tensor(x):
            torch = _get_torch()
            return torch.clamp(torch.min(aux1, aux2), 0.0, h)

        if isinstance(x, np.ndarray):
            return np.clip(np.minimum(aux1, aux2), 0.0, h)        
        elif isinstance(x, list):
            return [np.clip(min(aux1, aux2), 0.0, h) for elem in x]
        elif isinstance(x, pd.Series):
            return np.clip(np.minimum(aux1, aux2), 0.0, h)
        else: # Single value
            return np.clip(min(aux1, aux2), 0.0, h)


    def type(self) -> FUZZY_SETS:
        return FUZZY_SETS.t1
    

    def __str__(self) -> str:
        return f'{self.name} ({self.type().name}) - {self.membership_parameters} - {self.height}'
    

    def shape(self) -> str:
        return 'trapezoid'


class TriangularFS(FS):
    def __init__(self, name: str, membership_parameters: list[float], domain: list[float], height: float=1.0) -> None:
        super().__init__(name, membership_parameters, domain)
        self.height = height


    def membership(self, x: np.array, epsilon=10E-5) -> np.array:
        a, b, c = self.membership_parameters
        h = self.height

        if b == a:
            b += epsilon
        if b == c:
            b -= epsilon

        aux1 = h*(x - a) / (b - a)
        aux2 = -h*(x - c) / (c - b)
        
        if _is_torch_tensor(x):
            torch = _get_torch()
            val = torch.min(aux1, aux2)
            return torch.clamp(val, 0.0, h)

        if isinstance(x, np.ndarray):
            val = np.minimum(aux1, aux2)
            return np.clip(val, 0.0, h)

        elif isinstance(x, list):
            return [np.clip(min(a1, a2), 0.0, h) 
                    for a1, a2 in zip(aux1, aux2)]

        elif isinstance(x, pd.Series):
            val = np.minimum(aux1, aux2)
            return np.clip(val, 0.0, h)

        else:  # valor escalar
            val = min(aux1, aux2)
            return np.clip(val, 0.0, h)


    def type(self) -> FUZZY_SETS:
        return FUZZY_SETS.t1
    

    def __str__(self) -> str:
        return f'{self.name} ({self.type().name}) - {self.membership_parameters} - {self.height}'
    

    def shape(self) -> str:
        return 'triangular'


class GaussianFS(FS):
    def __init__(self, name: str, membership_parameters: list[float], universe_size: int) -> None:
        super().__init__(name, membership_parameters)
        self.universe_size = universe_size


    def membership(self, x: np.array) -> np.array:
        mean, standard_deviation = self.membership_parameters
        return np.exp(- ((x - mean) / standard_deviation) ** 2)

    
    def type(self) -> FUZZY_SETS:
        return FUZZY_SETS.t1
    

    def __str__(self) -> str:
        return f'{self.name} ({self.type().name}) - {self.membership_parameters}'
    

    def shape(self) -> str:
        return 'gaussian'


class CategoricalFS(FS):
    pass

class GT2(FS):
    pass

class IVFS(FS):
    pass

class CategoricalIVFS(IVFS):
    pass

class GaussianIVFS(IVFS):
    pass


def cut(fs1: FS, h: float) -> TrapezoidalFS:
    if fs1.shape() != 'trapezoid' and fs1.shape() != 'triangular':
        print('The fuzzy set must be either trapezoid or triangular')
        return None

    if h == 0:
        fs2 = TrapezoidalFS(f"cut {fs1.name}",  [0], fs1.domain, h)
        return fs2
    if h == 1:
        fs2 = TrapezoidalFS(f"cut {fs1.name}", fs1.membership_parameters, fs1.domain, h)
        return fs2
    
    m_params = fs1.membership_parameters

    z1 = m_params[0] + h * (m_params[1] - m_params[0])

    if fs1.shape() == 'triangular':
        z2 = m_params[1] + h * (m_params[2] - m_params[1])
    else:
        z2 = m_params[2] + h * (m_params[3] - m_params[2])

    fs2 = TrapezoidalFS(f"cut {fs1.name}", [m_params[0], z1, z2, m_params[-1]], fs1.domain, h)
    return fs2


def compute_intersection_x(s1, s2) -> float:
    x1, y1 = s1[0]
    x2, y2 = s1[1]
    m = (y2 - y1) / (x2 - x1)

    x1_p, y1_p = s2[0]
    x2_p, y2_p = s2[1]
    m_p = (y2_p - y1_p) / (x2_p - x1_p)

    x = (m*x1 - m_p*x1_p + y1_p - y1) / (m - m_p)
    return x


def segments_may_intersect(s1, s2) -> None:
    x2, _ = s1[1]
    x1_p, _ = s2[0]

    return x2 >= x1_p


def union(trapezoids: list[FS]) -> tuple:
    if not trapezoids:
        return [], []

    if len(trapezoids) == 1:
        p_x = trapezoids[0].membership_parameters
        h = trapezoids[0].height
        p_y = [0, h, h, 0]
        return p_x, p_y
    
    p_x = set()
    segments = []

    for i, trapezoid in enumerate(trapezoids):
        t_x = trapezoid.membership_parameters
        h = trapezoids[i].height
        x1, x2, x3, x4 = t_x[0], t_x[1], t_x[2], t_x[3]
        h1, h2, h3, h4 = 0, h, h, 0
        p_x.update([x1, x2, x3, x4])
        
        segments.append(([(x1,h1),(x2,h2)], i))
        segments.append(([(x2,h2),(x3,h3)], i))
        segments.append(([(x3,h3),(x4,h4)], i))
      
    for i in range(len(segments)):
        s1, idx1 = segments[i]
        for j in range(i + 1, len(segments)):
            s2, idx2 = segments[j]
            
            # only intersect segments from different trapezoids
            if idx1 != idx2 and segments_may_intersect(s1, s2):
                try:
                    x_intersect = compute_intersection_x(s1, s2)
                    
                    # check if intersection lies within both segments
                    if (s1[0][0] <= x_intersect <= s1[1][0] and s2[0][0] <= x_intersect <= s2[1][0]):
                        p_x.add(x_intersect)

                except ZeroDivisionError: # parallel lines, no intersection
                    pass

    p_x = np.sort(np.array(list(p_x)))
    
    y_values = np.array([fs.membership(p_x) for fs in trapezoids])
    p_y = np.max(y_values, axis=0)
    
    return p_x, p_y


def centroid_defuzzification(p_x, p_y) -> float:
    if len(p_x) != len(p_y):
        return -np.inf

    num = 0
    den = 0

    for i in range(len(p_x)-1):
        # segment points
        a, y_a = (p_x[i], p_y[i])
        b, y_b = (p_x[i+1], p_y[i+1])
        
        # line equation
        m = (y_b - y_a) / (b - a)
        n = -m*a + y_a

        # algebraic integration
        ba = b - a
        ba2 = b**2 / 2 - a**2 / 2
        ba3 = b**3 / 3 - a**3 / 3

        num += (m*ba3 + n*ba2) 
        den += (m*ba2 + n*ba)

    x_crisp = num / den
    return x_crisp


