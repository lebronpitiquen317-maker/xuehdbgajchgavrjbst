import numpy as np
import pandas as pd
from typing import List, Dict, Any

def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will change output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    
    return np.interp(quantiles, weighted_quantiles, values)

class Diagnostics:
    def __init__(self, history: List[Dict[str, Any]]):
        self.history = history
        self.times = [h['t'] for h in history]
        
    def get_state_quantiles(self, compartment_idx: int, quantiles=[0.025, 0.5, 0.975]):
        """
        Get quantiles for a state compartment over time.
        """
        results = []
        for h in self.history:
            states = h['states'][:, compartment_idx]
            weights = h['weights']
            qs = weighted_quantile(states, quantiles, weights)
            results.append(qs)
        return np.array(results) # Shape (T, len(quantiles))

    def get_incidence_quantiles(self, quantiles=[0.025, 0.5, 0.975]):
        """
        Get quantiles for incidence over time.
        """
        results = []
        for h in self.history:
            if 'incidence' in h:
                vals = h['incidence']
                weights = h['weights']
                qs = weighted_quantile(vals, quantiles, weights)
                results.append(qs)
        return np.array(results)

    def get_parameter_quantiles(self, param_name: str, quantiles=[0.025, 0.5, 0.975]):
        """
        Get quantiles for a parameter over time.
        """
        results = []
        for h in self.history:
            if param_name in h['params']:
                vals = h['params'][param_name]
                weights = h['weights']
                qs = weighted_quantile(vals, quantiles, weights)
                results.append(qs)
        return np.array(results)
        
    def get_ess_trace(self):
        return np.array([h['ess'] for h in self.history])
