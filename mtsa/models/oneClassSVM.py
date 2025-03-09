from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import OneClassSVM

from mtsa.features.mel import Array2Mfcc
from mtsa.utils import Wav2Array

class OSVM(BaseEstimator, OutlierMixin):
    def __init__(self, 
                 use_array2mfcc = False, 
                 use_featureUnion= False):
        super().__init__()
        self.final_model = OneClassSVM(kernel="rbf", nu =0.1)
        self.use_array2mfcc = use_array2mfcc
        self.use_featureUnion = use_featureUnion
        
    def fit(self, X, y=None):        
        self.final_model.fit(X)
        
    def score_samples(self, X):
        pred = self.final_model.predict(X)
        pred[pred == 1] = 1
        pred[pred == -1] = 0
        pred = pred.astype(float)
        return pred

    def _build_model(self):
        wav2array = Wav2Array(sampling_rate=self.sampling_rate)
        array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)
        features = FeatureUnion(self.features)
        
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
                    ("final_model", self.final_model),
                ]
            )
        
        return model