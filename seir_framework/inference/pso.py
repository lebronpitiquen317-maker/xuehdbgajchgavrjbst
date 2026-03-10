import numpy as np
from typing import Dict, List, Tuple, Callable
from ..model.base import CompartmentalModel
from .likelihood import ObservationModel

class AdaptivePSO:
    """
    Adaptive Particle Swarm Optimization for parameter initialization.
    Optimizes deterministic model fit to data.
    """
    
    def __init__(self, 
                 model: CompartmentalModel, 
                 data: np.ndarray, 
                 obs_model: ObservationModel,
                 param_bounds: Dict[str, Tuple[float, float]],
                 population_size: int = 30):
        """
        Args:
            model: Model instance
            data: Observed data array (incidence counts)
            obs_model: Observation likelihood model
            param_bounds: Dict of {param_name: (min, max)}
            population_size: Number of particles in swarm
        """
        self.model = model
        self.data = data
        self.obs_model = obs_model
        self.bounds = param_bounds
        self.pop_size = population_size
        self.param_names = list(param_bounds.keys())
        self.dim = len(self.param_names)
        
        # Initialize swarm
        self.positions = np.zeros((population_size, self.dim))
        self.velocities = np.zeros((population_size, self.dim))
        
        for i, (name, (low, high)) in enumerate(param_bounds.items()):
            self.positions[:, i] = np.random.uniform(low, high, population_size)
            self.velocities[:, i] = np.random.uniform(-(high-low), (high-low), population_size) * 0.1
            
        self.pbest_pos = self.positions.copy()
        self.pbest_scores = np.full(population_size, np.inf)
        self.gbest_pos = self.positions[0].copy()
        self.gbest_score = np.inf
        
    def _evaluate_cost(self, params_vec: np.ndarray, initial_state: np.ndarray) -> float:
        """
        Evaluate Negative Log Likelihood of deterministic model.
        """
        # Construct params dict
        params = self.model.param_dict.copy()
        for i, name in enumerate(self.param_names):
            params[name] = params_vec[i]
            
        # Run deterministic model
        # Assume data starts at t=0, steps=1 day
        t_start = 0
        t_end = len(self.data)
        
        # Update model params
        self.model.update_parameters(params)
        
        try:
            _, history = self.model.run(t_start, t_end, initial_state, mode='deterministic')
            
            # Calculate incidence (assuming C is last index)
            # history shape: (steps+1, n_comp)
            # incidence shape: (steps,)
            incidence = history[1:, -1] - history[:-1, -1]
            incidence = np.maximum(incidence, 0)
            
            # Calculate NLL
            # Sum of -log_likelihood for each data point
            nll = 0
            for t in range(len(self.data)):
                # Handle missing data if nan
                if np.isnan(self.data[t]):
                    continue
                    
                obs = self.data[t]
                pred = incidence[t]
                
                # Likelihood expects scalar/vector params.
                # Here params is a dict of scalars.
                ll = self.obs_model.log_likelihood(obs, pred, params)
                nll -= ll
                
            return nll
        except Exception as e:
            # unstable region
            return np.inf

    def optimize(self, initial_state: np.ndarray, max_iter: int = 50) -> Dict[str, float]:
        """
        Run PSO optimization.
        """
        w = 0.9 # Inertia
        c1 = 1.5 # Cognitive
        c2 = 1.5 # Social
        
        for it in range(max_iter):
            # Adapt inertia
            w = 0.9 - 0.5 * (it / max_iter)
            
            for i in range(self.pop_size):
                cost = self._evaluate_cost(self.positions[i], initial_state)
                
                if cost < self.pbest_scores[i]:
                    self.pbest_scores[i] = cost
                    self.pbest_pos[i] = self.positions[i].copy()
                    
                    if cost < self.gbest_score:
                        self.gbest_score = cost
                        self.gbest_pos = self.positions[i].copy()
            
            # Update velocities and positions
            r1 = np.random.random((self.pop_size, self.dim))
            r2 = np.random.random((self.pop_size, self.dim))
            
            self.velocities = (w * self.velocities + 
                               c1 * r1 * (self.pbest_pos - self.positions) + 
                               c2 * r2 * (self.gbest_pos - self.positions))
            
            self.positions += self.velocities
            
            # Bound constraints
            for j, (name, (low, high)) in enumerate(self.bounds.items()):
                self.positions[:, j] = np.clip(self.positions[:, j], low, high)
                
            # print(f"Iter {it}: Best Cost {self.gbest_score:.4f}")
            
        # Return best params
        best_params = {}
        for i, name in enumerate(self.param_names):
            best_params[name] = self.gbest_pos[i]
            
        return best_params
