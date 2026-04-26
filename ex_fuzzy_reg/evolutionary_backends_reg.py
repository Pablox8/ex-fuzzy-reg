"""
Backend abstraction layer for evolutionary optimization (Regression).

This module provides a unified interface for different evolutionary computation backends,
allowing users to choose between pymoo (CPU-based) and EvoX (GPU-accelerated with PyTorch).
This version is adapted for regression problems (FitRuleBaseReg / RuleBaseRegT1) and uses
NRMSE-based fitness instead of MCC.

Backends:
    - PyMooBackend: Default CPU-based backend using the pymoo library
    - EvoXBackend: GPU-accelerated backend using EvoX with PyTorch

Usage:
    Users can specify the backend when creating a regressor:

    # Using default pymoo backend
    reg = BaseFuzzyRulesRegressor(backend='pymoo')

    # Using EvoX with GPU acceleration
    reg = BaseFuzzyRulesRegressor(backend='evox')
"""

from abc import ABC, abstractmethod
from typing import Callable, Any
import numpy as np


class EvolutionaryBackend(ABC):
    """Abstract base class for evolutionary optimization backends."""

    @abstractmethod
    def optimize(self, problem: Any, n_gen: int, pop_size: int,
                 random_state: int, verbose: bool, **kwargs) -> dict:
        """
        Run evolutionary optimization.

        Args:
            problem: The optimization problem to solve
            n_gen: Number of generations
            pop_size: Population size
            random_state: Random seed
            verbose: Whether to print progress
            **kwargs: Backend-specific parameters

        Returns:
            dict with keys:
                - 'X': Best solution found (numpy array)
                - 'F': Best fitness value  (minimization objective, i.e. NRMSE)
                - 'pop': Final population
                - 'algorithm': Algorithm object (backend-specific)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (dependencies installed)."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend."""
        pass


class PyMooBackend(EvolutionaryBackend):
    """Backend using pymoo for CPU-based evolutionary optimization."""

    def __init__(self):
        self._available = self._check_availability()
        if self._available:
            print("PyMoo backend using CPU")

    def _check_availability(self) -> bool:
        try:
            import pymoo
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    def name(self) -> str:
        return "pymoo"

    def optimize(self, problem: Any, n_gen: int, pop_size: int,
                 random_state: int, verbose: bool,
                 var_prob: float = 0.9, sbx_eta: float = 3.0, mut_prob: float = 0.2,
                 mutation_eta: float = 4.0, tournament_size: int = 3,
                 sampling: Any = None, **kwargs) -> dict:
        """
        Optimize using pymoo's genetic algorithm.
        
        Args:
            problem: pymoo Problem instance
            n_gen: Number of generations
            pop_size: Population size
            random_state: Random seed
            verbose: Print progress
            var_prob: Crossover probability
            sbx_eta: SBX crossover eta parameter
            mut_prob: Mutation probability
            mutation_eta: Polynomial mutation eta parameter
            tournament_size: Tournament selection size
            sampling: Initial population sampling strategy
            **kwargs: Additional pymoo-specific parameters
            
        Returns:
            dict with optimization results; 'F' is the minimization objective (NRMSE)
        """
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.optimize import minimize
        from pymoo.operators.repair.rounding import RoundingRepair
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PolynomialMutation

        if sampling is None:
            sampling = IntegerRandomSampling()

        from pymoo.core.callback import Callback
        class BestOfAllTimeCallback(Callback):
            def __init__(self):
                super().__init__()
                self.best_x = None
                self.best_f = float("inf")
            def notify(self, algorithm):
                # Get current best from the population
                f = algorithm.pop.get("F")
                idx = f.argmin()
                if f[idx] < self.best_f:
                    self.best_f = f[idx][0]
                    self.best_x = algorithm.pop.get("X")[idx].copy()

        best_tracker = BestOfAllTimeCallback()

        algorithm = GA(
            pop_size=pop_size,
            crossover=SBX(prob=var_prob, eta=sbx_eta, repair=RoundingRepair()),
            mutation=PolynomialMutation(prob=mut_prob, eta=mutation_eta, repair=RoundingRepair()),
            tournament_size=tournament_size,
            sampling=sampling,
            eliminate_duplicates=False
        ) 

        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            seed=random_state,
            copy_algorithm=False,
            callback=best_tracker,
            verbose=verbose
        )

        return {
            'X': best_tracker.best_x if best_tracker.best_x is not None else res.X,
            'F': best_tracker.best_f if best_tracker.best_x is not None else res.F,
            'pop': res.pop,
            'res': res
        }

    # TODO: checkpoints?
    def optimize_with_checkpoints(self, problem: Any, n_gen: int, pop_size: int,
                                   random_state: int, verbose: bool,
                                   checkpoint_freq: int, checkpoint_callback: Callable,
                                   var_prob: float = 0.3, sbx_eta: float = 3.0,
                                   mutation_eta: float = 7.0, tournament_size: int = 3,
                                   sampling: Any = None, **kwargs) -> dict:
        """
        Optimize with checkpoint callbacks at specified intervals.

        Args:
            problem: pymoo Problem instance (FitRuleBaseReg)
            n_gen: Number of generations
            pop_size: Population size
            random_state: Random seed
            verbose: Print progress
            checkpoint_freq: Call checkpoint_callback every N generations
            checkpoint_callback: Callable(gen: int, best_individual: np.array) called at checkpoints
            var_prob: Crossover probability
            sbx_eta: SBX crossover eta parameter
            mutation_eta: Polynomial mutation eta parameter
            tournament_size: Tournament selection size
            sampling: Initial population sampling strategy
            **kwargs: Additional parameters

        Returns:
            dict with optimization results; 'F' is the minimization objective (NRMSE)
        """
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.operators.repair.rounding import RoundingRepair
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PolynomialMutation

        if sampling is None:
            sampling = IntegerRandomSampling()

        algorithm = GA(
            pop_size=pop_size,
            crossover=SBX(prob=var_prob, eta=sbx_eta, repair=RoundingRepair()),
            mutation=PolynomialMutation(eta=mutation_eta, repair=RoundingRepair()),
            tournament_size=tournament_size,
            sampling=sampling,
            eliminate_duplicates=False
        )

        if verbose:
            print('=================================================')
            print('n_gen  |  n_eval  |     f_avg     |     f_min    ')
            print('=================================================')

        algorithm.setup(problem, seed=random_state, termination=('n_gen', n_gen))

        for k in range(n_gen):
            algorithm.next()
            res = algorithm

            if verbose:
                print('%-6s | %-8s | %-8s | %-8s' % (
                    res.n_gen, res.evaluator.n_eval,
                    res.pop.get('F').mean(), res.pop.get('F').min()
                ))

            if k % checkpoint_freq == 0:
                pop = algorithm.pop
                fitness_last_gen = pop.get('F')
                best_solution_arg = np.argmin(fitness_last_gen)
                best_individual = pop.get('X')[best_solution_arg, :]

                # Call user-provided checkpoint callback
                checkpoint_callback(k, best_individual)

        # Extract final results
        pop = algorithm.pop
        fitness_last_gen = pop.get('F')
        best_solution = np.argmin(fitness_last_gen)
        best_individual = pop.get('X')[best_solution, :]

        return {
            'X': best_individual,
            'F': fitness_last_gen[best_solution],
            'pop': pop,
            'algorithm': algorithm,
            'res': algorithm
        }


class EvoXBackend(EvolutionaryBackend):
    """Backend using EvoX for GPU-accelerated evolutionary optimization with PyTorch.

    Adapted for regression: fitness is NRMSE (lower is better / minimization).
    """

    def __init__(self):
        self._available = self._check_availability()
        if self._available:
            self._setup_pytorch()

    def _check_availability(self) -> bool:
        try:
            import evox
            import torch
            return True
        except Exception:
            # Catch all exceptions: ImportError, RuntimeError (e.g., torch.compile
            # not supported on Python 3.14+), or any other initialization errors
            return False

    def _setup_pytorch(self):
        """Setup PyTorch configuration for GPU usage."""
        import torch

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            print(f"EvoX backend using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._device = torch.device('cpu')
            print(f"EvoX backend using CPU (GPU not available)")

    def is_available(self) -> bool:
        return self._available

    def name(self) -> str:
        return "evox"

    def _compute_nrmse_torch(self, y_pred: 'torch.Tensor', y_true: 'torch.Tensor',
                              y_min: float, y_max: float) -> float:
        """
        Compute Normalized Root Mean Squared Error (NRMSE) using PyTorch.

        Replaces the MCC computation used in the classification backend.
        NRMSE = RMSE / (y_max - y_min), clamped to [0, 1].

        :param y_pred: predicted values tensor
        :param y_true: true values tensor
        :param y_min: minimum value of the target range
        :param y_max: maximum value of the target range
        :return: NRMSE as Python float (0 = perfect, 1 = worst)
        """
        import torch

        y_pred = y_pred.float()
        y_true = y_true.float()

        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
        value_range = y_max - y_min

        if value_range == 0:
            return 0.0

        nrmse = rmse / value_range
        return float(torch.clamp(nrmse, 0.0, 1.0).cpu().item())

    def _batch_evaluate_torch(self, population: 'torch.Tensor', problem: Any, device: 'torch.device') -> 'torch.Tensor':
        """
        Batch evaluate population using PyTorch with GPU-accelerated NRMSE computation.

        :param population: population tensor (pop_size, n_var)
        :param problem: FitRuleBaseReg instance with optional _evaluate_torch_batch method
        :param device: torch device
        :return: fitness tensor (pop_size,) — minimization objective (NRMSE)
        """
        import torch

        # TRUE BATCHED EVALUATION - evaluate entire population at once
        if hasattr(problem, '_evaluate_torch_batch'):
            # Use fully batched evaluation if available
            score = problem._evaluate_torch_batch(population, device)
            return score
        
        # NumPy evaluation
        fitness_list = []
        for ind in population:
            out = {}
            problem._evaluate(ind.cpu().numpy().astype(int), out)
            fitness_list.append(out['F'])
        return torch.tensor(fitness_list, dtype=torch.float32, device=device)

    def optimize(self, problem: Any, n_gen: int, pop_size: int,
             random_state: int, verbose: bool,
             var_prob: float = 0.9, sbx_eta: float = 3.0, mut_prob: float = 0.9,
             mutation_eta: float = 4.0, tournament_size: int = 3,
             sampling: Any = None, **kwargs) -> dict:
        """
        Optimize using EvoX's genetic algorithm with PyTorch backend.

        Adapted for regression: fitness is NRMSE (minimization objective).

        Args:
            problem: FitRuleBaseReg instance compatible with EvoX
            n_gen: Number of generations
            pop_size: Population size
            random_state: Random seed
            verbose: Print progress
            var_prob: Crossover probability
            sbx_eta: SBX crossover distribution index
            mut_prob: Mutation probability
            mutation_eta: Polynomial mutation distribution index
            tournament_size: Tournament selection size
            sampling: Initial population (numpy array)
            **kwargs: Additional EvoX-specific parameters
    
        Returns:
            dict with optimization results, 'F' is the minimization objective (NRMSE)
        """
        import torch
        from evox.operators import mutation, crossover

        # Extract problem information
        n_var = problem.n_var
        xl = problem.xl
        xu = problem.xu

        # Set random seed for PyTorch
        torch.manual_seed(random_state)

        # Get device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize population
        if sampling is not None and isinstance(sampling, np.ndarray):
            # Initial population provided
            population = torch.tensor(sampling, dtype=torch.int32, device=device)
        else:
            # Random initialization with per-variable bounds
            population = torch.zeros((pop_size, n_var), dtype=torch.int32, device=device)
            for var_idx in range(n_var):
                population[:, var_idx] = torch.randint(
                    low=int(xl[var_idx]),
                    high=int(xu[var_idx]) + 1,
                    size=(pop_size,),
                    dtype=torch.int32,
                    device=device
                )

        # Convert bounds to PyTorch tensors for mutation
        lb_torch = torch.tensor(xl, dtype=torch.float32, device=device)
        ub_torch = torch.tensor(xu, dtype=torch.float32, device=device)

        # Check if problem has torch evaluation method
        has_torch_eval = hasattr(problem, '_evaluate_torch_batch')

        # Initial evaluation
        if has_torch_eval:
            # Use PyTorch evaluation for GPU acceleration with batched NRMSE computation
            fitness = self._batch_evaluate_torch(population, problem, device)
        else:
            # Fallback to numpy evaluation
            fitness_list = []
            for ind in population:
                out = {}
                problem._evaluate(ind.cpu().numpy().astype(int), out)
                fitness_list.append(out['F'])
            fitness = torch.tensor(fitness_list, dtype=torch.float32, device=device)


        # Initialize global best
        global_best_idx = torch.argmin(fitness)
        global_best_x = population[global_best_idx].clone()
        global_best_f = float(fitness[global_best_idx])

        for gen in range(n_gen):
            # Selection - VECTORIZED tournament selection (select pop_size parents for mating)
            # Generate all random tournaments at once
            tournament_candidates = torch.randint(0, pop_size, (pop_size, tournament_size), device=device)
            tournament_fitness = fitness[tournament_candidates] # (pop_size, tournament_size)
            selected_idx = tournament_candidates[torch.arange(pop_size, device=device), torch.argmin(tournament_fitness, dim=1)]
            selected_pop = population[selected_idx].float()

            # Shuffle to cross randomly
            shuffle_idx = torch.randperm(pop_size, device=device)
            selected_pop = selected_pop[shuffle_idx]

            # Crossover using EvoX simulated_binary function
            offspring = crossover.simulated_binary(selected_pop, pro_c=var_prob, dis_c=sbx_eta)
            offspring = torch.clamp(offspring, lb_torch.unsqueeze(0), ub_torch.unsqueeze(0))
            offspring = torch.round(offspring)

            # Mutation using EvoX polynomial_mutation function
            offspring = mutation.polynomial_mutation(
                offspring, lb=lb_torch, ub=ub_torch, pro_m=mut_prob, dis_m=mutation_eta
            )
            offspring = torch.clamp(offspring, lb_torch.unsqueeze(0), ub_torch.unsqueeze(0))
            offspring = torch.round(offspring).int()

            # Evaluate offspring
            if has_torch_eval:
                # Use PyTorch evaluation for GPU acceleration with batched NRMSE computation
                offspring_fitness = self._batch_evaluate_torch(offspring, problem, device)
            else:
                # Fallback to numpy evaluation
                fitness_list = []
                for ind in offspring:
                    out = {}
                    problem._evaluate(ind.cpu().numpy().astype(int), out)
                    fitness_list.append(out['F'])
                offspring_fitness = torch.tensor(fitness_list, dtype=torch.float32, device=device)

            # Elitist survival selection: combine parents and offspring, select best pop_size
            combined_pop = torch.cat([population, offspring], dim=0)
            combined_fitness = torch.cat([fitness, offspring_fitness], dim=0)

            # Select best pop_size individuals
            sorted_indices = torch.argsort(combined_fitness)[:pop_size]
            population = combined_pop[sorted_indices]
            fitness = combined_fitness[sorted_indices]

            # Update global best (already sorted population, index 0 is best)
            current_best_f = float(fitness[0])
            if current_best_f < global_best_f:
                global_best_f = current_best_f
                global_best_x = population[0].clone()

            if verbose and gen % max(1, n_gen // 10) == 0:
                print(f'Gen {gen:4d} | Best fitness: {global_best_f:.6f} | '
                    f'Avg fitness: {torch.mean(fitness):.6f}')

        if verbose:
            print(f'Optimization complete. Best fitness: {global_best_f:.6f}')

        return {
            'X': global_best_x.cpu().numpy().astype(int),
            'F': global_best_f,
            'pop': population.cpu().numpy(),
            'fitness': fitness.cpu().numpy(),
            'algorithm': None
        }


def get_backend(backend_name: str = 'pymoo') -> EvolutionaryBackend:
    """
    Get an evolutionary backend by name.

    Args:
        backend_name: Name of the backend ('pymoo' or 'evox')

    Returns:
        EvolutionaryBackend instance

    Raises:
        ValueError: If backend is not available or unknown
    """
    backends = {
        'pymoo': PyMooBackend,
        'evox': EvoXBackend,
    }

    if backend_name not in backends:
        raise ValueError(
            f"Unknown backend '{backend_name}'. Available backends: {list(backends.keys())}"
        )

    backend = backends[backend_name]()

    if not backend.is_available():
        raise ValueError(
            f"Backend '{backend_name}' is not available. "
            f"Please install required dependencies. "
            f"For EvoX: pip install ex-fuzzy[evox]"
        )

    return backend


def list_available_backends() -> list[str]:
    """
    List all available backends.

    Returns:
        List of backend names that are currently available
    """
    all_backends = ['pymoo', 'evox']
    available = []

    for name in all_backends:
        try:
            backend = get_backend(name)
            if backend.is_available():
                available.append(name)
        except ValueError:
            pass

    return available
