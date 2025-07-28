import numpy as np
import time
from mtsa.features.mel import Array2Mfcc
from mtsa.utils import Wav2Array
from typing import List, Optional
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, OutlierMixin
from mtsa.models.ransyncoders_components.ransyncoders_base import RANSynCodersBase

class RANSynCoders(BaseEstimator, OutlierMixin):
    def __init__(self, 
                 n_estimators: int = 5,
                 max_features: int = 5,
                 encoding_depth: int = 1,
                 latent_dim: int = 1,
                 decoding_depth: int = 2,
                 activation: str = 'relu',
                 output_activation: str = 'sigmoid',
                 delta: float = 0.05,  # quantile bound for regression
                 # Syncrhonization inputs
                 synchronize: bool = False, #False
                 force_synchronization: bool = True,  # if synchronization is true but no significant frequencies found
                 min_periods: int = 3,  # if synchronize and forced, this is the minimum bound on cycles to look for in train set
                 freq_init: Optional[List[float]] = None,  # initial guess for the dominant angular frequency
                 max_freqs: int = 5,  # the number of sinusoidal signals to fit
                 min_dist: int = 60,  # maximum distance for finding local maximums in the PSD
                 trainable_freq: bool = False,  # whether to make the frequency a variable during layer weight training
                 bias: bool = True,  # add intercept (vertical displacement)
                 sampling_rate: int = 16000,
                 mono: bool = False,
                 is_acoustic_data: bool = False,
                 normal_classifier: int = 0,
                 abnormal_classifier: int = 1,
                ) -> None:
            super().__init__()
            self.last_fit_time = 0
            
            # Rancoders inputs:
            self.n_estimators = n_estimators
            self.max_features = max_features
            self.encoding_depth = encoding_depth
            self.latent_dim = latent_dim
            self.decoding_depth = decoding_depth
            self.activation = activation
            self.output_activation = output_activation
            self.delta = delta
            
            # Syncrhonization inputs
            self.synchronize = synchronize
            self.force_synchronization = force_synchronization
            self.min_periods = min_periods
            self.freq_init = freq_init  # in radians (angular frequency)
            self.max_freqs = max_freqs
            self.min_dist = min_dist
            self.trainable_freq = trainable_freq
            self.bias = bias
            self.sampling_rate = sampling_rate
            self.mono = mono
            self.is_acoustic_data = is_acoustic_data
            self.normal_classifier = normal_classifier
            self.abnormal_classifier = abnormal_classifier
            self.model = self.build_model()

    @property
    def name(self):
        return "RANSynCoders"
        
    def fit(self, 
            X,
            y = None, 
            timestamps_matrix: np.ndarray = None,
            batch_size: int = 180, 
            learning_rate: float = 0.001,
            epochs: int = 10,
            freq_warmup: int = 5,  # number of warmup epochs to prefit the frequency
            sin_warmup: int = 5,  # number of warmup epochs to prefit the sinusoidal representation 
            pos_amp: bool = True,  # whether to constraint amplitudes to be +ve only
            shuffle: bool = True
        ):
        start = time.perf_counter()  
        self.model.fit(X, 
                              y, 
                              final_model__timestamps_matrix = timestamps_matrix,
                              final_model__batch_size = batch_size, 
                              final_model__learning_rate = learning_rate,
                              final_model__epochs = epochs,
                              final_model__freq_warmup = freq_warmup,
                              final_model__sin_warmup = sin_warmup,
                              final_model__pos_amp = pos_amp,
                              final_model__shuffle = shuffle,
                            )
        end = time.perf_counter()
        self.last_fit_time = end - start
        return self
    
    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        if self.is_acoustic_data:    
            return np.array(
                list(
                    map(
                        self.model.score_samples, 
                        [[x] for x in X])
                    )
                )
            
        return self.model.score_samples(X)
    
    def set_timestamps_matrix_to_predict(self, timestamps_matrix: np.ndarray):
        self.final_model.set_timestamps_matrix_to_predict(timestamps_matrix)
    
    def save(self):
        return self.final_model.save()

    def get_config(self):
        return self.final_model.get_config()

    def get_final_model(self):
        return RANSynCodersBase(n_estimators = self.n_estimators, 
                                max_features = self.max_features, 
                                encoding_depth = self.encoding_depth, 
                                latent_dim = self.latent_dim, 
                                decoding_depth = self.decoding_depth, 
                                activation = self.activation,
                                output_activation = self.output_activation,
                                delta = self.delta,
                                synchronize = self.synchronize,
                                force_synchronization = self.force_synchronization,
                                min_periods = self.min_periods,  
                                freq_init = self.freq_init,  
                                max_freqs = self.max_freqs,  
                                min_dist = self.min_dist, 
                                trainable_freq = self.trainable_freq,  
                                bias = self.bias,
                                sampling_rate = self.sampling_rate,
                                mono = self.mono,
                                is_acoustic_data = self.is_acoustic_data,
                                normal_classifier = self.normal_classifier,
                                abnormal_classifier = self.abnormal_classifier
                            )
        
    def build_model(self):
        steps = self.__get_pipeline_steps()
        model = Pipeline(steps=steps)
        
        return model
    
    def __get_pipeline_steps(self):
        wav2array = Wav2Array(sampling_rate=self.sampling_rate, mono=self.mono)
        array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)
       
        self.final_model = self.get_final_model()
        
        if self.is_acoustic_data:
            steps = [("wav2array", wav2array),
                     ("array2mfcc", array2mfcc),
                     ("final_model", self.final_model)
                    ]
        else:
            steps = [("final_model", self.final_model)]
        
        return steps
    
