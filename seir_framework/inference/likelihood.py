import numpy as np
from scipy import stats
from abc import ABC, abstractmethod

class ObservationModel(ABC):
    """
    Abstract base class for observation models (Likelihoods).
    """
    @abstractmethod
    def log_likelihood(self, observed: float, expected: float, params: dict) -> float:
        """
        Compute log p(y | x, theta).
        
        Args:
            observed: The observed data point (y_t)
            expected: The model-predicted value (true incidence x_t)
            params: Dictionary containing parameters like 'rho', 'kappa'
        """
        pass

class PoissonLikelihood(ObservationModel):
    """
    Poisson observation model.
    y ~ Poisson(rho * expected)
    """
    def log_likelihood(self, observed: float, expected: float, params: dict) -> float:
        rho = params.get('rho', 1.0)
        mu = rho * expected
        # Avoid log(0)
        mu = np.maximum(mu, 1e-9)
        return stats.poisson.logpmf(observed, mu)

class NegativeBinomialLikelihood(ObservationModel):
    """
    Negative Binomial observation model.
    y ~ NegBin(mean = rho * expected, dispersion = kappa)
    Variance = mean + mean^2 / kappa
    """
    def log_likelihood(self, observed: float, expected: float, params: dict) -> float:
        rho = params.get('rho', 1.0)
        kappa = params.get('kappa', 10.0) # Dispersion parameter
        
        mu = rho * expected
        mu = np.maximum(mu, 1e-9)
        
        # Convert to scipy parameterization (n, p)
        # Var = mu + mu^2 / kappa = mu / p
        # mu/p = mu + mu^2/kappa => 1/p = 1 + mu/kappa => p = 1 / (1 + mu/kappa) = kappa / (kappa + mu)
        # n = mu * p / (1-p) ... actually simpler: mean=n(1-p)/p.
        # If we use n=kappa, p=kappa/(kappa+mu):
        # mean = kappa * (mu/(kappa+mu)) / (kappa/(kappa+mu)) * ((kappa+mu)/kappa - 1) ... ?
        # Let's verify: n=k, p=k/(k+m).
        # Mean = k * (1 - k/(k+m)) / (k/(k+m)) = k * (m/(k+m)) / (k/(k+m)) = m. Correct.
        
        n = kappa
        p = kappa / (kappa + mu)
        
        return stats.nbinom.logpmf(observed, n, p)
