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
        Xt = np.sqrt(np.mean(X**2))
        return Xt
    
class SquareRootOfAmplitude(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = (np.mean(np.sqrt(np.absolute(X)))) ** 2
        return Xt
    
class Kurtosis(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = kurtosis(X)
        return Xt
    
class Skewness(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = skew(X)
        return Xt
    
class Peak2Peak(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = np.max(X) - np.min(X)
        return Xt
    
class CrestFactor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.root_mean_square_feature = RootMeanSquareFeature()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = np.max(np.absolute(X)) / self.root_mean_square_feature.transform(X)
        return Xt
    
class ImpulseValue(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt =np.max(np.absolute(X)) / np.mean(np.absolute(X))
        return Xt
    
class MarginFactor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.square_root_of_amplitude = SquareRootOfAmplitude()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = np.max(np.absolute(X)) / self.square_root_of_amplitude.transform(X)
        return Xt
    
class ShapeFactor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.root_mean_square_feature = RootMeanSquareFeature()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.root_mean_square_feature.transform(X) / np.mean(np.absolute(X))
        return Xt

class KurtosisFactor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.kurtosis = Kurtosis()
        self.root_mean_square_feature = RootMeanSquareFeature()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.kurtosis.transform(X) / (self.root_mean_square_feature.transform(X) ** 4)
        return Xt
    
class FrequencyCenter(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = 2 * np.abs(rfft(X)) / X.size
        return np.mean(Xt)
    
class RootMeanSquareFrequency(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = 2 * np.abs(rfft(X)) / X.size
        return np.sqrt(np.mean(Xt**2))
    
class RootFrequencyVariance(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.frequency_center = FrequencyCenter()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = 2 * np.abs(rfft(X)) / X.size
        return np.sqrt(np.mean((Xt - self.frequency_center.transform(Xt)) ** 2))

class FeatureExtractionMixer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.rootMeanSquareFeature = RootMeanSquareFeature()
        self.squareRootOfAmplitude = SquareRootOfAmplitude()
        self.kurtosis = Kurtosis()
        self.skewness = Skewness()
        self.peak2Peak = Peak2Peak()
        self.crestFactor = CrestFactor()
        self.impulseValue = ImpulseValue()
        self.marginFactor = MarginFactor()
        self.shapeFactor = ShapeFactor()
        self.kurtosisFactor = KurtosisFactor()
        self.frequencyCenter = FrequencyCenter()
        self.rootMeanSquareFrequency = RootMeanSquareFrequency()
        self.rootFrequencyVariance = RootFrequencyVariance()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        result = []
        result.append(self.rootMeanSquareFeature.transform(X))
        result.append(self.squareRootOfAmplitude.transform(X))
        result.append(self.kurtosis.transform(X))
        result.append(self.skewness.transform(X))
        result.append(self.peak2Peak.transform(X))
        result.append(self.crestFactor.transform(X))
        result.append(self.impulseValue.transform(X))
        result.append(self.marginFactor.transform(X))
        result.append(self.shapeFactor.transform(X))
        result.append(self.kurtosisFactor.transform(X))
        result.append(self.frequencyCenter .transform(X))
        result.append(self.rootMeanSquareFrequency.transform(X))
        result.append(self.rootFrequencyVariance.transform(X))
        result = np.array(result, dtype=float)
        return result
    
#endregion

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

