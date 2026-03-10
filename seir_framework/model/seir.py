import numpy as np
from typing import Dict, List, Union
from .base import CompartmentalModel

class SEIRModel(CompartmentalModel):
    """
    Standard SEIR model with time-varying transmission.
    State vector: [S, E, I, R, C]
    S: Susceptible
    E: Exposed
    I: Infectious
    R: Recovered
    C: Cumulative Incidence (E -> I transitions)
    """

    def __init__(self, N: int, params: Dict[str, float], dt: float = 1.0, seed: int = None):
        """
        Args:
            N: Total population size
            params: {'beta': ..., 'sigma': ..., 'gamma': ...}
            dt: Time step
            seed: Random seed
        """
        super().__init__(['S', 'E', 'I', 'R', 'C'], params, dt, seed)
        self.N = N

    def get_derivatives(self, t: float, state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Deterministic ODEs.
        dS/dt = -beta * S * I / N
        dE/dt = beta * S * I / N - sigma * E
        dI/dt = sigma * E - gamma * I
        dR/dt = gamma * I
        dC/dt = sigma * E  (Cumulative cases - usually measured at symptom onset)
        """
        S, E, I, R, C = state
        
        beta = params.get('beta')
        sigma = params.get('sigma')
        gamma = params.get('gamma')
        
        # Force of infection
        # Add small epsilon to I to prevent absorbing state issues if desired, 
        # but standard model is beta * S * I / N
        force_infection = beta * I / self.N
        
        dS = -force_infection * S
        dE = force_infection * S - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        dC = sigma * E # Accumulate transitions from E to I
        
        return np.array([dS, dE, dI, dR, dC])

    def step_stochastic(self, t: float, state: np.ndarray, params: Dict[str, Union[float, np.ndarray]], dt: float) -> np.ndarray:
        """
        Tau-leaping approximation for stochastic dynamics.
        Supports vectorized execution if state is (N_particles, 5).
        """
        # Handle vectorization
        if state.ndim == 1:
            S, E, I, R, C = state
        else:
            S, E, I, R, C = state.T
        
        beta = params.get('beta')
        sigma = params.get('sigma')
        gamma = params.get('gamma')
        
        # 1. Infection: S -> E
        # Rate = beta * S * I / N
        rate_inf = beta * S * I / self.N
        # Use np.maximum for safety with arrays
        n_inf = self.rng.poisson(np.maximum(0, rate_inf * dt))
        # Clamp to available S
        n_inf = np.minimum(n_inf, S)
        
        # 2. Progression: E -> I
        # Rate = sigma * E
        rate_prog = sigma * E
        n_prog = self.rng.poisson(np.maximum(0, rate_prog * dt))
        # Clamp to available E
        n_prog = np.minimum(n_prog, E)
        
        # 3. Recovery: I -> R
        # Rate = gamma * I
        rate_rec = gamma * I
        n_rec = self.rng.poisson(np.maximum(0, rate_rec * dt))
        # Clamp to available I
        n_rec = np.minimum(n_rec, I)
        
        # Update states
        S_new = S - n_inf
        E_new = E + n_inf - n_prog
        I_new = I + n_prog - n_rec
        R_new = R + n_rec
        C_new = C + n_prog # Cumulative cases tracks E->I transitions
        
        if state.ndim == 1:
            return np.array([S_new, E_new, I_new, R_new, C_new])
        else:
            return np.stack([S_new, E_new, I_new, R_new, C_new], axis=1)
