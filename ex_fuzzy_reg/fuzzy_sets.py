"""
Fuzzy Sets Library

Provides base and concrete implementations of Type-1 fuzzy sets:
- Trapezoidal
- Triangular
- Gaussian

Supports NumPy arrays and optionally PyTorch tensors.
"""
import enum
import abc
import math

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

"""
Handles the logic for a trapezoidal membership function calculation.

Used for both trapezoidal and triangular fuzzy sets, since a TriangularFS with parameters [a, b, c] is equivalent to 
a TrapezoidalFS with parameters [a, b, b, c].
"""
def _trapezoidal_membership_logic(x: np.ndarray, params: list[float], h: float):
    x = np.asanyarray(x) # if x is already a numpy array, passes it through
    a, b, c, d = params

    if a == d:
        return np.where(np.isclose(x, a), h, 0.0)
            
    with np.errstate(divide='ignore', invalid='ignore'):
        aux1 = np.where(b > a, h*(x - a) / (b - a), 1.0)
        aux2 = np.where(d > c, -h*(x - d) / (d - c), 1.0)
    
    return np.clip(np.minimum(aux1, aux2), 0.0, h) 


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


class FS(abc.ABC):
    """
    Base class for fuzzy set implementation.
    
    It serves as the base class for more specialized fuzzy set types like trapezoidal, triangular,
    gaussian, and categorical fuzzy sets.
    
    Attributes:
        name (str): The linguistic name of the fuzzy set (e.g., "low", "medium", "high")
        membership_parameters (list[float]): Parameters defining the membership function.
            The interpretation depends on the subclass (e.g., trapezoidal uses [a, b, c, d])
        domain (list[float]): Two-element list defining the universe of discourse [min, max]
    """

    def __init__(self, name: str, membership_parameters: list[float], domain: list[float]=None, height: float=1.0) -> None:
        """
        Initialize a fuzzy set.

        Args:
            name (str): Linguistic name for the fuzzy set
            membership_parameters (list[float]): Parameters defining the membership function 
            domain (list[float]): Two-element list [min, max] defining the universe
                of discourse for this fuzzy set
        """
        self.name = name
        self.domain = domain
        self.membership_parameters = membership_parameters
        self.height = height


    @abc.abstractmethod
    def membership(self, x: np.ndarray) -> np.ndarray:
        """
        Compute membership degrees for input values.

        This method calculates the membership degree(s) for the given input value(s)
        using the membership function defined by this fuzzy set's parameters.

        Args:
            x (np.ndarray): Input value(s) for which to compute membership degrees.
                Can be a single value, list, or numpy array.

        Returns:
            np.ndarray: Membership degree(s) in the range [0, 1]. Shape matches input.
        """
        raise NotImplementedError


    @abc.abstractmethod
    def type(self) -> FUZZY_SETS:
        """
        Returns the fuzzy set type identifier.

        Returns:
            FUZZY_SETS: The type identifier.
        """
        raise NotImplementedError
    

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns the name of the fuzzy set, its type and its parameters.
        
        Returns: 
            string: name, type and parameters of the fuzzy set.
        """
        raise NotImplementedError
    

    @abc.abstractmethod
    def shape(self) -> str:
        """
        Returns the shape of the fuzzy set.

        Returns: 
            string: shape of the fuzzy set.
        """
        raise NotImplementedError


    def is_empty(self) -> bool:
        """
        Returns True if the fuzzy set is empty (all parameters are 0).

        Returns: 
            bool: if the fuzzy set is empty or not.
        """
        return self.height == 0


class TrapezoidalFS(FS):
    """
    Trapezoidal fuzzy set implementation.
    """
    def __init__(self, name: str, membership_parameters: list[float], domain: list[float], height: float=1.0) -> None:
        """
        Initialize a trapezoidal fuzzy set.

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
            height (float): Max value of the membership function. It is 1 by default.
        """
        super().__init__(name, membership_parameters, domain, height)


    def membership(self, x: ArrayLike) -> np.ndarray:
        """
        Compute membership degrees for input values.

        This method calculates the membership degree(s) for the given input value(s)
        using the membership function defined by this fuzzy set's parameters.

        Args:
            x (array-like): Input value(s) for which to compute membership degrees.
                Can be a single value, list, or numpy array.

        Returns:
            np.ndarray: Membership degree(s) in the range [0, 1]. Shape matches input.
        """
        x = np.asanyarray(x) # if x is already a numpy array, passes it through
        return _trapezoidal_membership_logic(x, self.membership_parameters, self.height)     


    def type(self) -> FUZZY_SETS:
        """
        Returns the fuzzy set type identifier.

        Returns:
            FUZZY_SETS: The type identifier.
        """
        return FUZZY_SETS.t1
    

    def __str__(self) -> str:
        """
        Returns the name of the fuzzy set, its type and its parameters.
        
        Returns: 
            string: name, type and parameters of the fuzzy set.
        """
        return f'{self.name} ({self.type().name}) - {self.membership_parameters} - {self.height}'
    

    def shape(self) -> str:
        """
        Returns the shape of the fuzzy set.

        Returns: 
            string: shape of the fuzzy set ('trapezoidal' in this case).
        """
        return 'trapezoidal'


class TriangularFS(FS):
    """
    Triangular fuzzy set implementation.
    """
    def __init__(self, name: str, membership_parameters: list[float], domain: list[float], height: float=1.0) -> None:
        """
        Initialize a triangular fuzzy set.

        Args:
            name (str): Linguistic name for the fuzzy set
            membership_parameters (list[float]): Three parameters [a, b, c] defining 
                the triangular membership function where:
                - a: left foot (membership starts rising from 0)
                - b: peak (membership reaches 1.0)
                - c: right foot (membership reaches 0)
            domain (list[float]): Two-element list [min, max] defining the universe
                of discourse for this fuzzy set
            height (float): Max value of the membership function. It is 1 by default.
        """
        super().__init__(name, membership_parameters, domain, height)

    
    def membership(self, x: ArrayLike) -> np.ndarray:
        """
        Compute membership degrees for input values.

        This method calculates the membership degree(s) for the given input value(s)
        using the membership function defined by this fuzzy set's parameters.

        Args:
            x (array-like): Input value(s) for which to compute membership degrees.
                Can be a single value, list, or numpy array.

        Returns:
            np.ndarray: Membership degree(s) in the range [0, 1]. Shape matches input.
        """
        x = np.asanyarray(x) # if x is already a numpy array, passes it through
        a, b, c = self.membership_parameters
        h = self.height

        # TriangularFS [a, b, c] is equivalent to TrapezoidalFS [a, b, b, c]
        return _trapezoidal_membership_logic(x, [a, b, b, c], h)


    def type(self) -> FUZZY_SETS:
        """
        Returns the fuzzy set type identifier.

        Returns:
            FUZZY_SETS: The type identifier.
        """
        return FUZZY_SETS.t1
    

    def __str__(self) -> str:
        """
        Returns the name of the fuzzy set, its type and its parameters.
        
        Returns: 
            string: name, type and parameters of the fuzzy set.
        """
        return f'{self.name} ({self.type().name}) - {self.membership_parameters} - {self.height}'
    

    def shape(self) -> str:
        """
        Returns the shape of the fuzzy set.

        Returns: 
            string: shape of the fuzzy set ('triangular' in this case).
        """
        return 'triangular'


class GaussianFS(FS):
    """
    Gaussian fuzzy set implementation.
    """
    def __init__(self, name: str, membership_parameters: list[float], universe_size: int, height: float=1.0) -> None:
        """
        Initialize a Gaussian fuzzy set.

        Args:
            name (str): Linguistic name for the fuzzy set
            membership_parameters (list[float]): Two parameters [mean, std] defining 
                the gaussian membership function where:
                - mean: center of the gaussian curve where the membership function is maximum
                - std: spread of the gaussian curve. Must be greater than 0
            universe_size (int): Number of elements considered in the discrete membership function
            height: max value for the membership function. Defaults to 1
        """
        mean, std = membership_parameters
        if std <= 0:
            raise ValueError("std must be greater than 0.")
        super().__init__(name, membership_parameters, domain=None, height=height)
        self.universe_size = universe_size


    def membership(self, x: ArrayLike) -> np.ndarray:
        """
        Compute membership degrees for input values.

        This method calculates the membership degree(s) for the given input value(s)
        using the membership function defined by this fuzzy set's parameters.
        
        The membership function is defined as:

        μ(x) = h * exp(- (x - mean)^2 / (2 * std^2))

        Args:
            x (array-like): Input value(s) for which to compute membership degrees.
                Can be a single value, list, or numpy array.

        Returns:
            np.ndarray: Membership degree(s) in the range (0, 1]. Shape matches input.
        """
        x = np.asanyarray(x)
        mean, std = self.membership_parameters
        h = self.height

        return h * np.exp(-(x - mean)**2 / (2*std**2))
    

    def type(self) -> FUZZY_SETS:
        """
        Returns the fuzzy set type identifier.

        Returns:
            FUZZY_SETS: The type identifier.
        """
        return FUZZY_SETS.t1
    

    def __str__(self) -> str:
        """
        Returns the name of the fuzzy set, its type and its parameters.
        
        Returns: 
            string: name, type and parameters of the fuzzy set.
        """
        return f'{self.name} ({self.type().name}) - {self.membership_parameters}'
    

    def shape(self) -> str:
        """
        Returns the shape of the fuzzy set.

        Returns: 
            string: shape of the fuzzy set ('gaussian' in this case).
        """
        return 'gaussian'


# TODO: Implement CategoricalFS
class CategoricalFS(FS):
    pass


# TODO: Implement GT2
class GT2(FS):
    pass


# TODO: Implement IVFS
class IVFS(FS):
    pass


# TODO: Implement CategoricalIVFS
class CategoricalIVFS(IVFS):
    pass


# TODO: Implement GaussianIVFS
class GaussianIVFS(IVFS):
    pass


def cut(fs1: FS, h: float) -> FS:
    """
    Clips (truncates) the given fuzzy set at height h.

    This method limits the membership function to a maximum value of h,
    producing a new TrapezoidalFS whose peak membership equals h.

    Args:
        fs1 (FS): Input fuzzy set to be cut. Must be either trapezoid or triangular. 
        h (float): Height used to cut the given fuzzy set. Must be in range [0, 1].

    Returns:
        TrapezoidalFS: Resulting fuzzy set from cutting the given one.

    Note:
        - If fs1 is triangular, the result is a trapezoidal fuzzy set.
        - If h == 0, the resulting fuzzy set has zero membership everywhere.
        - If h == 1, the original shape is preserved. 
    """
    if h < 0 or h > 1:
        raise ValueError("h must be in range [0, 1].")

    if h == 0:
        fs2 = TrapezoidalFS(f"cut {fs1.name}",  [0, 0, 0, 0], fs1.domain, h)
        return fs2
    if h == 1: 
        return fs1

    if fs1.shape() == 'gaussian':
        mean, std = fs1.membership_parameters
        orig_h = fs1.height

        # value of x where membership(x) = h
        x_offset = std * math.sqrt(-2 * math.log(h / orig_h))
        z1 = mean - x_offset
        z2 = mean + x_offset

        # 3-sigma covers approximately 99.7% of the area
        a = mean - 3*std
        d = mean + 3*std

        a = min(a, z1)
        d = max(d, z2)

        return TrapezoidalFS(f"cut {fs1.name}", [a, z1, z2, d], fs1.domain, h)

    m_params = fs1.membership_parameters

    z1 = m_params[0] + h * (m_params[1] - m_params[0])

    if fs1.shape() == 'triangular':
        z2 = m_params[1] + h * (m_params[2] - m_params[1])
    else:
        z2 = m_params[2] + h * (m_params[3] - m_params[2])

    fs2 = TrapezoidalFS(f"cut {fs1.name}", [m_params[0], z1, z2, m_params[-1]], fs1.domain, h)
    return fs2


def compute_intersection_x(s1, s2) -> float | None:
    """
    In the xy plane, computes the x component of the intersection between two segments.

    Args:
        s1 (list[tuple[float, float]]): First segment [(x1, y1), (x2, y2)].
        s2 (list[tuple[float, float]]): Second segment [(x1_p, y1_p), (x2_p, y2_p)].

    Returns:
        float | None: x component of the intersection between the segments, or None if the segments don't intersect. 

    Note:
        A segment AB is defined by two points A(x1, y1) and B(x2, y2).
    """
    x1, y1 = s1[0]
    x2, y2 = s1[1]
    x1_p, y1_p = s2[0]
    x2_p, y2_p = s2[1]

    is_vertical1 = x2 == x1
    is_vertical2 = x2_p == x1_p

    if is_vertical1 and is_vertical2:
        return x1 if x1 == x1_p else None

    if is_vertical1:
        m_p = (y2_p - y1_p) / (x2_p - x1_p)        
        y = m_p * (x1 - x1_p) + y1_p
        
        if min(y1, y2) <= y <= max(y1, y2) and min(y1_p, y2_p) <= y <= max(y1_p, y2_p):
            return x1
        return None

    if is_vertical2:
        m = (y2 - y1) / (x2 - x1)
        y = m * (x1_p - x1) + y1
        
        if min(y1, y2) <= y <= max(y1, y2) and min(y1_p, y2_p) <= y <= max(y1_p, y2_p):
            return x1_p
        return None
        
    m = (y2 - y1) / (x2 - x1)
    m_p = (y2_p - y1_p) / (x2_p - x1_p)

    if m == m_p:
        return None  # parallel segments

    x = (m*x1 - m_p*x1_p + y1_p - y1) / (m - m_p)
    if min(x1, x2) <= x <= max(x1, x2) and min(x1_p, x2_p) <= x <= max(x1_p, x2_p):
        return x
    return None


def segments_may_intersect(s1, s2) -> bool:
    """
    In the xy plane, checks if the x-projections of two line segments overlap.

    Args:
        s1 (list[tuple[float, float]]): First segment [(x1, y1), (x2, y2)].
        s2 (list[tuple[float, float]]): Second segment [(x1_p, y1_p), (x2_p, y2_p)].
    
    Returns:
        bool: True if the x-ranges of the segments overlap, False otherwise.
    
    Note:
        This is only a necessary condition for intersection, not a guarantee.
        Segments may still not intersect if their y-ranges or slopes differ.
    """
    return (max(s1[0][0], s1[1][0]) >= min(s2[0][0], s2[1][0]) and  
            max(s2[0][0], s2[1][0]) >= min(s1[0][0], s1[1][0]))


def trapezoidal_triangular_union(fuzzy_sets: list[FS]) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a list of trapezoidal and triangular fuzzy sets, computes and returns the points (x, y) representing the union of all fuzzy sets passsed.

    Args:
        fuzzy_sets (list[FS]): fuzzy sets to compute the union of.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: x and y component of the calculated union points. Both have the same shape.
            - x (np.ndarray): Sorted x-coordinates covering all fuzzy sets and their intersections.
            - y (np.ndarray): Corresponding y-values representing the maximum membership at each x.
        Returns (None, None) if no fuzzy sets are passed or all passed fuzzy sets are empty.
    """
    if not fuzzy_sets:
        return [], [] # no fuzzy sets passed
    
    if len(fuzzy_sets) == 1:
        p_x = fuzzy_sets[0].membership_parameters
        h = fuzzy_sets[0].height

        if fuzzy_sets[0].shape() == 'trapezoidal':
            p_y = [0, h, h, 0]
        elif fuzzy_sets[0].shape() == 'triangular':
            p_y = [0, h, 0]
        
        return p_x, p_y

    p_x = set()
    segments = []

    for i, fs in enumerate(fuzzy_sets):
        if not fs.is_empty():
            fs_x = fs.membership_parameters
            h = fs.height

            if fs.shape() == 'trapezoidal':
                x1, x2, x3, x4 = fs_x[0], fs_x[1], fs_x[2], fs_x[3]
                h1, h2, h3, h4 = 0, h, h, 0
                
                p_x.update([x1, x2, x3, x4])
            
                segments.append(([(x1,h1),(x2,h2)], i))
                segments.append(([(x2,h2),(x3,h3)], i))
                segments.append(([(x3,h3),(x4,h4)], i))

            elif fs.shape() == 'triangular':
                x1, x2, x3 = fs_x[0], fs_x[1], fs_x[2]
                h1, h2, h3 = 0, h, 0
                
                p_x.update([x1, x2, x3])
            
                segments.append(([(x1,h1),(x2,h2)], i))
                segments.append(([(x2,h2),(x3,h3)], i))

    if not p_x:
        return [], [] # all fuzzy sets are empty

    for i in range(len(segments)):
        s1, idx1 = segments[i]
        for j in range(i + 1, len(segments)):
            s2, idx2 = segments[j]
            
            # only intersect segments from different trapezoids
            if idx1 != idx2 and segments_may_intersect(s1, s2):
                x_intersect = compute_intersection_x(s1, s2)
                if x_intersect is not None:
                    p_x.add(x_intersect)

    p_x = np.sort(np.array(list(p_x)))
    
    y_values = np.array([fs.membership(p_x) for fs in fuzzy_sets if not fs.is_empty()])
    p_y = np.max(y_values, axis=0)
    
    return p_x, p_y
 

def centroid_defuzzification(p_x: ArrayLike, p_y: ArrayLike) -> float:
    """
    Computes the x-coordinate of the centroid (center of gravity) of a 
    fuzzy set with its membership function defined by linear segments.

    The points (p_x[i], p_y[i]) are treated as connected points forming straight-line segments.
    The centroid is calculated by integrating over each segment.

    Args:
        p_x (ArrayLike): 1D array of x-component of consecutive vertices.
        p_y (ArrayLike): 1D array of y-component of consecutive vertices.
        Both arrays must be non-empty and have the same shape.
    
    Returns:
        float: x-coordinate of the centroid. 
        Returns 0 if the total area is zero (which would otherwise result in NaN).
    
    Note:
        Vertical segments are ignored because they contribute no area to the integral.
    """
    if len(p_x) == 0 or len(p_y) == 0:
        raise ValueError("p_x and p_y must not be empty.")

    if len(p_x) != len(p_y):
        raise ValueError("p_x and p_y must have the same shape.")
    
    p_x = np.asarray(p_x)
    p_y = np.asarray(p_y)

    # segment points
    a  = p_x[:-1]
    b  = p_x[1:] 
    y_a = p_y[:-1]
    y_b = p_y[1:]

    non_vertical = (b - a) != 0
    a = a[non_vertical]
    b = b[non_vertical]
    y_a = y_a[non_vertical]
    y_b = y_b[non_vertical]

    # line equation
    m = (y_b - y_a) / (b - a)
    n = y_a - m * a

    # precompute powers
    ba   = b - a
    ba2  = (b**2 - a**2) / 2
    ba3  = (b**3 - a**3) / 3

    # algebraic integration
    num = m * ba3 + n * ba2
    den = m * ba2 + n * ba

    if np.sum(den) == 0:
        return np.nan

    x_crisp = np.sum(num) / np.sum(den)

    return x_crisp


def first_of_maxima_defuzzification(p_x: ArrayLike, p_y: ArrayLike) -> float:
    return p_x[np.argmax(p_y)]


def last_of_maxima_defuzzification(p_x: ArrayLike, p_y: ArrayLike) -> float:
    max_y = np.max(p_y)
    return p_x[p_y == max_y][-1]


def mean_of_maxima_defuzzification(p_x: ArrayLike, p_y: ArrayLike) -> float:
    max_y = np.max(p_y)
    max_xs = p_x[p_y == max_y]
    return np.mean(max_xs)