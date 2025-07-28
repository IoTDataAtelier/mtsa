from sklearn import metrics
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import OneClassSVM
from mtsa.features.mel import Array2Mfcc
from mtsa.features.stats import StatisticalSignalDescriptors
from mtsa.utils import Wav2Array
from mtsa.features.stats import FEATURES
import time

class OSVM(BaseEstimator, OutlierMixin):
    def __init__(self, 
                 use_array2mfcc = False, 
                 sampling_rate=None,
                 use_stats_features= True,
                 isForWaveData = True,
                 use_MFCC_features=False,
                 kernel = "rbf",
                 nu = 0.1,
                 degree = 3,
                 mono = True):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.kernel = kernel
        self.nu = nu
        self.degree = degree
        self.final_model = OneClassSVM(kernel=self.kernel, nu=self.nu, degree=self.degree)
        self.use_array2mfcc = use_array2mfcc
        self.use_stats_features = use_stats_features
        self.use_MFCC_features = use_MFCC_features
        self.isForWaveData = isForWaveData
        self.mono = mono
        self.model = self._build_model()
        self.last_fit_time = 0
        
    @property
    def name(self):
        return "OSVM"
        
    def fit(self, X, y=None):    
        start = time.perf_counter()     
        self.model.fit(X)
        end = time.perf_counter()
        self.last_fit_time = end - start
        return self
        
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
        array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)
        wav2array = Wav2Array(sampling_rate=self.sampling_rate, mono=self.mono)
        stats_features = FeatureUnion(StatisticalSignalDescriptors)
        MFCC_features = FeatureUnion(FEATURES)
        model = Pipeline(steps=[])
        
        if self.isForWaveData:
            model.steps.append(("wav2array", wav2array))
            
        if self.use_array2mfcc:
            model.steps.append(("array2mfcc", array2mfcc))
        
        if self.use_stats_features:
            model.steps.append(("statsFeatures", stats_features))
            
        if self.use_MFCC_features:
            model.steps.append(("MFCCfeatures", MFCC_features))
            
        model.steps.append(("final_model", self.final_model))

        return model