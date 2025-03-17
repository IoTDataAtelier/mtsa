from sklearn import metrics
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import OneClassSVM
from mtsa.features.mel import Array2Mfcc
from mtsa.features.stats import CrestFactor, FeatureExtractionMixer, FrequencyCenter, ImpulseValue, Kurtosis, KurtosisFactor, MarginFactor, Peak2Peak, RootFrequencyVariance, RootMeanSquareFeature, RootMeanSquareFrequency, ShapeFactor, Skewness, SquareRootOfAmplitude
from mtsa.utils import Wav2Array
from mtsa.features.stats import FEATURES
import time

class OSVM(BaseEstimator, OutlierMixin):
    def __init__(self, 
                 use_array2mfcc = False, 
                 sampling_rate=None,
                 use_featureUnion= False):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.final_model = OneClassSVM(kernel="rbf", nu =0.1)
        self.use_array2mfcc = use_array2mfcc
        self.use_featureUnion = use_featureUnion
        self.model = self._build_model()
        self.last_fit_time = 0
        
    @property
    def name(self):
        return "OSVM " + "+".join([f[0] for f in self.features])
        
    def fit(self, X, y=None):    
        start = time.perf_counter()     
        self.model.fit(X)
        end = time.perf_counter()
        self.last_fit_time = end - start
        return self.last_fit_time #seconds
        
    def score_samples(self, X):
        pred = self.model.predict(X)
        pred[pred == 1] = 1
        pred[pred == -1] = 0
        pred = pred.astype(float)
        return pred
    
    def score(self, X, Y):
        X = self.score_samples
        fpr, tpr, thresholds = metrics.roc_curve(Y, X)
        auc = metrics.auc(fpr, tpr)
        return auc

    def predict(self, X):
        return self.model.predict(X)

    def _build_model(self):
        wav2array = Wav2Array(sampling_rate=self.sampling_rate)
        array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)
        features = FeatureUnion(FEATURES)
        featureExtractionMixer = FeatureExtractionMixer()
        
        featureExtractionMixer.append_strategy(RootMeanSquareFeature())
        featureExtractionMixer.append_strategy(SquareRootOfAmplitude())
        featureExtractionMixer.append_strategy(Kurtosis())
        featureExtractionMixer.append_strategy(Skewness())
        featureExtractionMixer.append_strategy(Peak2Peak())
        featureExtractionMixer.append_strategy(CrestFactor())
        featureExtractionMixer.append_strategy(ImpulseValue())
        featureExtractionMixer.append_strategy(MarginFactor())
        featureExtractionMixer.append_strategy(ShapeFactor())
        featureExtractionMixer.append_strategy(KurtosisFactor())
        featureExtractionMixer.append_strategy(FrequencyCenter())
        featureExtractionMixer.append_strategy(RootMeanSquareFrequency())
        featureExtractionMixer.append_strategy(RootFrequencyVariance())
        
        
        if self.use_array2mfcc and self.use_featureUnion:
            model = Pipeline(
                steps=[
                    ("wav2array", wav2array),
                    ("array2mfcc", array2mfcc),
                    ("features", features),
                    ("final_model", self.final_model),
                ]
            )
        elif self.use_array2mfcc:
            model = Pipeline(
                steps=[
                    ("wav2array", wav2array),
                    ("array2mfcc", array2mfcc),
                    ("final_model", self.final_model),
                ]
            )
        elif self.use_featureUnion:
            model = Pipeline(
                steps=[
                    ("wav2array", wav2array),
                    ("features", features),
                    ("final_model", self.final_model),
                ]
            )
        else:
            model = Pipeline(
                steps=[
                    ("wav2array", wav2array),
                    ("featureExtractionMixer", featureExtractionMixer),
                    ("final_model", self.final_model),
                ]
            )

        return model