"""Base classes for all mfcc estimators."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MagnitudeMeanFeatureMfcc(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = X.mean(axis=2)
        return Xt
    
class MagnitudeStdFeatureMfcc(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = X.std(axis=2)
        return Xt
    
from mtsa.correlation_networks import PearsonCorrelationNetwork
class CorrelationFeatureMfcc(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def get_triu(G):
            return G[np.triu_indices(len(G), k = 1)]
        corr = PearsonCorrelationNetwork()
        Xt = map(corr.get_correlation, X)
        Xt = map(get_triu, Xt)
        Xt= np.array(list(Xt)) 
        return Xt