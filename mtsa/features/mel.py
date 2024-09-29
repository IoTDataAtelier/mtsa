"""Base classes for all mfcc estimators."""

import numpy as np
import sys
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats as st
import librosa as lib
from functools import reduce

class Array2MelSpec(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 sampling_rate,
                 n_fft,
                 hop_length,
                 n_mels,
                 frames,
                 power
                 ):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.frames = frames
        self.power = power

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        
        def normalize_melspec(mel_spectrogram):

            # 03 convert melspectrogram to log mel energy
            log_mel_spectrogram = 20.0 / self.power * np.log10(mel_spectrogram + sys.float_info.epsilon)

            # 04 calculate total vector size
            vectorarray_size = len(log_mel_spectrogram[0, :]) - self.frames + 1

            # 05 skip too short clips
            if vectorarray_size < 1:
                return np.empty((0, dims), float)

            # 06 generate feature vectors by concatenating multi_frames
            dims = self.n_mels * self.frames
            vectorarray = np.zeros((vectorarray_size, dims), float)
            for t in range(self.frames):
                vectorarray[:, self.n_mels * t: self.n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

            return vectorarray
            
        def extract_melspec(y):
            params = {
                'y':y, 
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                "n_mels": self.n_mels,
                "power": self.power
            }
            if self.sampling_rate:
                params['sr']=self.sampling_rate
            
            mel_spectrogram = lib.feature.melspectrogram(**params)
            
            normalized_melspec = normalize_melspec(mel_spectrogram)

            return normalized_melspec
        Xt = np.array(
            reduce(
                lambda n1, n2: np.concatenate([n1,n2]), 
                map(extract_melspec, X))
        )
        return Xt
    
    
class Array2Mfcc(BaseEstimator, TransformerMixin):
    """
     Gets a numpy array containing audio signals and transforms it into a numpy array containing mfcc signals
     
    """
    
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        #self.n_mfcc = n_mfcc

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        def extract_mfcc(y):
            if self.sampling_rate: 
                return lib.feature.mfcc(y=y, sr=self.sampling_rate)
            else: 
                return lib.feature.mfcc(y=y)
        Xt = np.array(list(map(extract_mfcc, X)))
        return Xt

