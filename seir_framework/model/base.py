import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union

class CompartmentalModel(ABC):
    """
    Abstract base class for compartmental epidemiological models.
    Supports both deterministic (ODE) and stochastic (SDE/Tau-leaping) dynamics.
    """

    def __init__(self, 
                 compartments: List[str], 
                 parameters: Dict[str, float],
                 dt: float = 1.0,
                 seed: int = None):
        """
        Initialize the model.

        Args:
            compartments: List of compartment names (e.g., ['S', 'E', 'I', 'R'])
            parameters: Dictionary of static parameters
            dt: Time step size for simulation
            seed: Random seed for reproducibility
        """
        self.compartments = compartments
        self.param_dict = parameters
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.state_idx = {name: i for i, name in enumerate(compartments)}
        self.n_compartments = len(compartments)

    @abstractmethod
    def get_derivatives(self, t: float, state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Compute deterministic derivatives (dS/dt, dE/dt, etc.).
        
        Args:
            t: Current time
            state: Current state vector
            params: Current parameters (including time-varying ones)
            
        Returns:
            np.ndarray: Array of derivatives
        """
        pass

    @abstractmethod
    def step_stochastic(self, t: float, state: np.ndarray, params: Dict[str, float], dt: float) -> np.ndarray:
        """
        Perform a single stochastic step (e.g., Euler-Maruyama or Tau-leaping).
        
        Args:
            t: Current time
            state: Current state vector
            params: Current parameters
            dt: Time step
            
        Returns:
            np.ndarray: New state vector
        """
        pass

    def update_parameters(self, new_params: Dict[str, float]):
        """Update internal parameters."""
        self.param_dict.update(new_params)

    def run(self, 
            t_start: float, 
            t_end: float, 
            initial_state: np.ndarray, 
            time_varying_params: Optional[Dict[str, callable]] = None,
            mode: str = 'deterministic') -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the simulation.

        Args:
            t_start: Start time
            t_end: End time
            initial_state: Initial population in each compartment
            time_varying_params: Dictionary mapping param name to function f(t) -> float
            mode: 'deterministic' or 'stochastic'

        Returns:
            Tuple[np.ndarray, np.ndarray]: (time_points, state_history)
        """
        steps = int((t_end - t_start) / self.dt)
        time_points = np.linspace(t_start, t_end, steps + 1)
        state_history = np.zeros((steps + 1, self.n_compartments))
        state_history[0] = initial_state
        
        current_state = initial_state.copy()
        
        for i, t in enumerate(time_points[:-1]):
            # Resolve parameters for current time
            current_params = self.param_dict.copy()
            if time_varying_params:
                for key, func in time_varying_params.items():
                    current_params[key] = func(t)
            
            if mode == 'deterministic':
                # RK4 integration for stability
                k1 = self.get_derivatives(t, current_state, current_params)
                k2 = self.get_derivatives(t + 0.5*self.dt, current_state + 0.5*self.dt*k1, current_params)
                k3 = self.get_derivatives(t + 0.5*self.dt, current_state + 0.5*self.dt*k2, current_params)
                k4 = self.get_derivatives(t + self.dt, current_state + self.dt*k3, current_params)
                current_state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            elif mode == 'stochastic':
                current_state = self.step_stochastic(t, current_state, current_params, self.dt)
                
            # Ensure non-negativity
            current_state = np.maximum(current_state, 0)
            state_history[i+1] = current_state
            
        return time_points, state_history
