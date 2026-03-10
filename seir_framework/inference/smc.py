import numpy as np
from typing import Dict, List, Callable, Optional, Union
from ..model.base import CompartmentalModel
from .likelihood import ObservationModel

class ParticleFilter:
    """
    Sequential Monte Carlo (Particle Filter) for epidemiological inference.
    
    Features:
    - Joint state and parameter estimation.
    - Adaptive resampling based on ESS.
    - Time-varying parameter evolution (Random Walk).
    """
    
    def __init__(self, 
                 model: CompartmentalModel, 
                 observation_model: ObservationModel,
                 n_particles: int = 1000,
                 ess_threshold: float = 0.5):
        """
        Args:
            model: The compartmental model instance.
            observation_model: The likelihood model.
            n_particles: Number of particles.
            ess_threshold: Threshold for resampling (fraction of n_particles).
        """
        self.model = model
        self.obs_model = observation_model
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold * n_particles
        
        self.particles_state = None # Shape (n_particles, n_compartments)
        self.particles_params = {} # Dict of (n_particles,) arrays
        self.weights = np.ones(n_particles) / n_particles
        
        # History storage
        self.history = []
        
        # Parameter evolution config
        # Key: param_name, Value: sigma (std dev of log-space random walk)
        self.param_walk_sigma = {} 
        
    def initialize(self, 
                   initial_state_priors: Dict[str, Union[float, Callable]],
                   param_priors: Dict[str, Callable]):
        """
        Initialize particles from priors.
        
        Args:
            initial_state_priors: Dict mapping compartment name to value or generator.
                                  e.g. {'S': 990, 'E': 0, 'I': 10, 'R': 0, 'C': 0}
            param_priors: Dict mapping param name to generator function (returning scalar or array).
        """
        # Initialize States
        self.particles_state = np.zeros((self.n_particles, self.model.n_compartments))
        for i, name in enumerate(self.model.compartments):
            val = initial_state_priors.get(name, 0)
            if callable(val):
                self.particles_state[:, i] = val(size=self.n_particles)
            else:
                self.particles_state[:, i] = val
                
        # Initialize Parameters
        for name, prior_gen in param_priors.items():
            if callable(prior_gen):
                self.particles_params[name] = prior_gen(size=self.n_particles)
            else:
                self.particles_params[name] = np.full(self.n_particles, prior_gen)
                
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.history = []

    def set_parameter_walk(self, param_name: str, sigma: float):
        """
        Enable random walk for a parameter.
        param(t+1) = param(t) * exp(N(0, sigma))
        """
        self.param_walk_sigma[param_name] = sigma

    def step(self, t: float, dt: float, observed_data: float = None):
        """
        Perform one step of the particle filter.
        
        1. Evolve parameters (Random Walk).
        2. Evolve state (Stochastic Model Step).
        3. Compute Weights (Likelihood) if data is present.
        4. Resample if needed.
        """
        # 1. Evolve Parameters
        for name, sigma in self.param_walk_sigma.items():
            if name in self.particles_params:
                # Log-normal random walk
                noise = np.random.normal(0, sigma, self.n_particles)
                self.particles_params[name] *= np.exp(noise)
        
        # Store previous cumulative cases to calculate incidence
        # Assuming 'C' is the last compartment as in SEIRModel
        prev_C = self.particles_state[:, -1].copy()
        
        # 2. Evolve State
        # Model expects params dict
        self.particles_state = self.model.step_stochastic(
            t, self.particles_state, self.particles_params, dt
        )
        
        # Calculate Incidence (New Cases)
        current_C = self.particles_state[:, -1]
        incidence = current_C - prev_C
        # Ensure non-negative incidence due to potential numerical artifacts (though C is monotonic)
        incidence = np.maximum(incidence, 0)
        
        # 3. Update Weights
        if observed_data is not None:
            # Calculate log likelihood for each particle
            # We construct a params dict for the observation model from particle params
            # Note: Observation model might need 'rho' or 'kappa' which are in particles_params
            
            # Vectorized likelihood computation would be ideal, but our ObsModel is scalar.
            # We can implement a vectorized helper or loop.
            # Let's check if ObsModel can handle vectors.
            # scipy.stats usually handles vectors.
            
            # Prepare params for likelihood
            # We need to pass the vectors directly to likelihood.
            obs_params = {k: v for k, v in self.particles_params.items()}
            
            # Compute log weights
            log_weights = self.obs_model.log_likelihood(observed_data, incidence, obs_params)
            
            # Update weights: w_new = w_old * likelihood
            # Working in log space for stability? 
            # w = w * exp(log_lik)
            # But standard way: w_new = w_old * likelihood
            
            # Handle -inf
            log_weights = np.nan_to_num(log_weights, nan=-np.inf)
            
            # Avoid underflow by shifting
            max_log_w = np.max(log_weights)
            if max_log_w == -np.inf:
                 # All particles have 0 likelihood. This is bad. 
                 # Could happen if outliers. Reset weights or fail.
                 # For robustness, add small epsilon? Or just let it fail/resample.
                 print(f"Warning: All particles have zero likelihood at t={t}")
                 # Reset to uniform?
                 weights_unnorm = np.ones(self.n_particles)
            else:
                weights_unnorm = self.weights * np.exp(log_weights - max_log_w)
            
            # Normalize
            if np.sum(weights_unnorm) > 0:
                self.weights = weights_unnorm / np.sum(weights_unnorm)
            else:
                self.weights = np.ones(self.n_particles) / self.n_particles
                
        # 4. Resample
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.ess_threshold:
            self.resample()
            
        # Store history (summary stats or full particles?)
        # Storing full particles for all steps might be heavy. 
        # But required for "full particle ensemble" output.
        # We'll store a copy.
        snapshot = {
            't': t,
            'state_mean': np.average(self.particles_state, axis=0, weights=self.weights),
            'params_mean': {k: np.average(v, weights=self.weights) for k, v in self.particles_params.items()},
            'ess': ess,
            # Store subset or full? Let's store full for now.
            'states': self.particles_state.copy(),
            'weights': self.weights.copy(),
            'params': {k: v.copy() for k, v in self.particles_params.items()},
            'incidence': incidence.copy()
        }
        self.history.append(snapshot)

    def resample(self):
        """
        Systematic resampling.
        """
        positions = (np.arange(self.n_particles) + np.random.random()) / self.n_particles
        indexes = np.zeros(self.n_particles, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.n_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        
        self.particles_state = self.particles_state[indexes]
        for k in self.particles_params:
            self.particles_params[k] = self.particles_params[k][indexes]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def get_posterior_estimates(self):
        """
        Return history of estimates.
        """
        return self.history
