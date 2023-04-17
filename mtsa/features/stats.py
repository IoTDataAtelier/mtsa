"""Base classes for all mfcc estimators."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce
import itertools as ite

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
    

FEATURES = [
    ("M", MagnitudeMeanFeatureMfcc()), 
    ("S", MagnitudeStdFeatureMfcc()), 
    ("C", CorrelationFeatureMfcc())
]

def get_features(self):
    number_features = np.arange(1, len(FEATURES)+1)
    def combinations(r): return ite.combinations(FEATURES, r)
    all_combinations = map(combinations, number_features)
    features = reduce(lambda x, y: ite.chain(x, y), all_combinations)
    return features

