from sklearn import metrics
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import OneClassSVM
from mtsa.features.mel import Array2Mfcc
from mtsa.features.stats import CrestFactor, FeatureExtractionMixer, FrequencyCenter, ImpulseValue, Kurtosis, KurtosisFactor, MarginFactor, Peak2Peak, RootFrequencyVariance, RootMeanSquareFeature, RootMeanSquareFrequency, ShapeFactor, Skewness, SquareRootOfAmplitude
from mtsa.utils import Wav2Array
from mtsa.features.stats import FEATURES

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
        
    @property
    def name(self):
        return "OSVM " + "+".join([f[0] for f in self.features])
        
    def fit(self, X, y=None):        
        return self.model.fit(X)
        
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
        
        # rootMeanSquareFeature = RootMeanSquareFeature()
        # squareRootOfAmplitude = SquareRootOfAmplitude()
        # kurtosis = Kurtosis()
        # skewness = Skewness()
        # peak2Peak = Peak2Peak()
        # crestFactor = CrestFactor()
        # impulseValue = ImpulseValue()
        # marginFactor = MarginFactor()
        # shapeFactor = ShapeFactor()
        # kurtosisFactor = KurtosisFactor()
        # frequencyCenter = FrequencyCenter()
        # rootMeanSquareFrequency = RootMeanSquareFrequency()
        # rootFrequencyVariance = RootFrequencyVariance()
        
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
        # else:
        #     model = Pipeline(
        #         steps=[
        #             ("wav2array", wav2array),
        #             ("rootMeanSquareFeature",rootMeanSquareFeature),
        #             ("squareRootOfAmplitude",squareRootOfAmplitude),
        #             ("kurtosis",kurtosis),
        #             ("skewness",skewness),
        #             ("peak2Peak",peak2Peak),
        #             ("crestFactor",crestFactor),
        #             ("impulseValue",impulseValue),
        #             ("marginFactor",marginFactor),
        #             ("shapeFactor",shapeFactor),
        #             ("kurtosisFactor",kurtosisFactor),
        #             ("frequencyCenter",frequencyCenter),
        #             ("rootMeanSquareFrequenc",rootMeanSquareFrequency),
        #             ("rootFrequencyVariance",rootFrequencyVariance),
        #             ("final_model", self.final_model),
        #         ]
        #     )
        
        return model