"""Base classes for all mfcc estimators."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import kurtosis, skew
from scipy.fft import rfft
from functools import reduce
import itertools as ite
from mtsa.correlation_networks import PearsonCorrelationNetwork

#region feature extraction strategies

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
    
class RootMeanSquareFeature(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = np.sqrt(np.mean(x**2))
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt
    
class SquareRootOfAmplitude(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = (np.mean(np.sqrt(np.absolute(x)))) ** 2
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt
    
class Kurtosis(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = kurtosis(x)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt
    
class Skewness(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = skew(x)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt
    
class Peak2Peak(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = np.max(x) - np.min(x)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt
    
class CrestFactor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.root_mean_square_feature = RootMeanSquareFeature()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = np.max(np.absolute(x)) / self.root_mean_square_feature.transform(x)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt
    
class ImpulseValue(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = np.max(np.absolute(x)) / np.mean(np.absolute(x))
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt
    
class MarginFactor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.square_root_of_amplitude = SquareRootOfAmplitude()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = np.max(np.absolute(x)) / self.square_root_of_amplitude.transform(x)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt
    
class ShapeFactor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.root_mean_square_feature = RootMeanSquareFeature()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = self.root_mean_square_feature.transform(x) / np.mean(np.absolute(x))
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt

class KurtosisFactor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.kurtosis = Kurtosis()
        self.root_mean_square_feature = RootMeanSquareFeature()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = self.kurtosis.transform(x) / (self.root_mean_square_feature.transform(x) ** 4)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt
    
class FrequencyCenter(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = 2 * np.abs(rfft(x)) / x.size
            xt = np.mean(xt)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt
    
class RootMeanSquareFrequency(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        def transform_sample(x):
            xt = 2 * np.abs(rfft(x)) / x.size
            xt = np.sqrt(np.mean(xt**2))
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
        
        return Xt
    
    
class RootFrequencyVariance(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.frequency_center = FrequencyCenter()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):        
        def transform_sample(x):
            xt = 2 * np.abs(rfft(x)) / x.size
            xt = np.sqrt(np.mean((xt - self.frequency_center.transform(xt)) ** 2))
            return xt
            
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[transform_sample(x)] for x in X])
        else:      
            Xt = transform_sample(X)
            
        return Xt

#endregion

#region methods

def get_features(self):
    number_features = np.arange(1, len(FEATURES)+1)
    def combinations(r): return ite.combinations(FEATURES, r)
    all_combinations = map(combinations, number_features)
    features = reduce(lambda x, y: ite.chain(x, y), all_combinations)
    return features

#endregion

#region pre-arranged set of features

FEATURES = [
    ("M", MagnitudeMeanFeatureMfcc()), 
    ("S", MagnitudeStdFeatureMfcc()), 
    ("C", CorrelationFeatureMfcc())
]

StatisticalSignalDescriptors = [
    ("rootMeanSquareFeature", RootMeanSquareFeature()),
    ("squareRootOfAmplitude", SquareRootOfAmplitude()),
    ("kurtosis", Kurtosis()),
    ("skewness", Skewness()),
    ("peak2Peak", Peak2Peak()),
    ("crestFactor", CrestFactor()),
    ("impulseValue", ImpulseValue()),
    ("marginFactor", MarginFactor()),
    ("shapeFactor", ShapeFactor()),
    ("kurtosisFactor", KurtosisFactor()),
    ("frequencyCenter", FrequencyCenter()),
    ("rootMeanSquareFrequenc", RootMeanSquareFrequency()),
    ("rootFrequencyVariance", RootFrequencyVariance())
]

#endregion

