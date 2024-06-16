from sklearn import metrics
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from functools import reduce
from sklearn.base import BaseEstimator, OutlierMixin
from functools import reduce
from mtsa.features.mel import Array2Mfcc
from mtsa.models.GANF_components.ganfBaseModel import GANFBaseModel
from mtsa.utils import Wav2Array
from sklearn.pipeline import Pipeline
from mtsa.utils import Wav2Array

class GANF(nn.Module, BaseEstimator, OutlierMixin):

    def __init__(self, 
                 sampling_rate=None,
                 random_state = None,
                 isForWaveData = False,
                 batch_size = None,
                 learning_rate = None,
                 mono = True,
                 use_array2mfcc = False,
                 index_CUDA_device = '0'
                 ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.random_state = random_state
        self.isForWaveData =isForWaveData
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mono = mono
        self.final_model = GANFBaseModel(index_CUDA_device=index_CUDA_device)
        self.use_array2mfcc = use_array2mfcc
        self.model = self._build_model()

    @property
    def name(self):
        return "GANF " + "+".join([f[0] for f in self.features])
        
    def fit(self, X, y=None, batch_size = None, epochs= None, max_iteraction= None, learning_rate = None, debug_dataframe = None):
        mono = (self.mono and not self.use_array2mfcc)
        if batch_size is None:
            batch_size = self.batch_size
        if learning_rate is None:
            learning_rate = self.learning_rate

        return self.model.fit(X, y, 
                              final_model__batch_size=batch_size,
                              final_model__epochs=epochs,
                              final_model__max_iteraction=max_iteraction,
                              final_model__learning_rate = learning_rate,
                              final_model__debug_dataframe = debug_dataframe,
                              final_model__isWaveData = self.isForWaveData,
                              final_model__mono = mono)

    def transform(self, X, y=None):
        l = list()
        l.append(X)
        l.extend(self.model.steps[:-1])
        Xt = reduce(lambda x, y: y[1].transform(x), l)
        return Xt
    
    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        return np.array(
            list(
                map(
                    self.model.score_samples, 
                    [[x] for x in X])
                )
            )
    
    def score(self, X, Y):
        X = self.score_samples
        fpr, tpr, thresholds = metrics.roc_curve(Y, X)
        auc = metrics.auc(fpr, tpr)
        return auc
    
    def get_adjacent_matrix(self):
        return self.final_model.get_adjacent_matrix()

    def _build_model(self):
        array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)
        wav2array = Wav2Array(sampling_rate=self.sampling_rate, mono=self.mono)
        #mono = (self.mono and not self.use_array2mfcc)
        if self.use_array2mfcc and self.isForWaveData:
            model = Pipeline(
                steps=[
                    ("wav2array", wav2array),
                    ("array2mfcc", array2mfcc),
                    ("final_model", self.final_model),
                    ]
                )
        elif  self.isForWaveData:
            model = Pipeline(
                steps=[
                    ("wav2array", wav2array),
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