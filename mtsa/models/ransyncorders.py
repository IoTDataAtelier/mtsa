from functools import reduce
from joblib import dump, load
import numpy as np
import os
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
from spectrum import Periodogram
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.constraints import NonNeg
from tensorflow.python.keras.layers import Dense, Layer
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, model_from_json
from tensorflow.python.keras.initializers import Constant
from typing import List, Optional
from sklearn.pipeline import Pipeline
from mtsa.utils import Wav2Array
from sklearn.base import BaseEstimator, OutlierMixin

class RANSynCodersBase():
    """ class for building, training, and testing rancoders models """
    def __init__(
            self,
            # Rancoders inputs:
            n_estimators: int = 10, #100
            max_features: int = 4, #3
            encoding_depth: int = 1, #2
            latent_dim: int = 4, #2 
            decoding_depth: int = 2,
            activation: str = 'relu', #linear
            output_activation: str = 'sigmoid', #linear
            delta: float = 0.05,  # quantile bound for regression
            # Syncrhonization inputs
            synchronize: bool = True, #False
            force_synchronization: bool = True,  # if synchronization is true but no significant frequencies found
            min_periods: int = 3,  # if synchronize and forced, this is the minimum bound on cycles to look for in train set
            freq_init: Optional[List[float]] = None,  # initial guess for the dominant angular frequency
            max_freqs: int = 5,  # the number of sinusoidal signals to fit = 1
            min_dist: int = 60,  # maximum distance for finding local maximums in the PSD
            trainable_freq: bool = False,  # whether to make the frequency a variable during layer weight training
            bias: bool = True,  # add intercept (vertical displacement)
    ):
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
        # set all variables to default to float32
        tf.keras.backend.set_floatx('float32')

    def build(self, input_shape, initial_stage: bool = False):
        x_in = Input(shape=(input_shape[-1],))  # created for either raw signal or synchronized signal
        if initial_stage:
            freq_out = freqcoder()(x_in)
            self.freqcoder = Model(inputs=x_in, outputs=freq_out)
            self.freqcoder.compile(optimizer='adam', loss=lambda y,f: quantile_loss(0.5, y,f))
        else:
            bounds_out = RANCoders(
                    n_estimators = self.n_estimators,
                    max_features = self.max_features,
                    encoding_depth = self.encoding_depth,
                    latent_dim = self.latent_dim,
                    decoding_depth = self.decoding_depth,
                    delta = self.delta,
                    activation = self.activation,
                    output_activation = self.output_activation,
                    name='rancoders'
                )(x_in)
            self.rancoders = Model(inputs=x_in, outputs=bounds_out)
            self.rancoders.compile(
                    optimizer='adam', 
                    loss=[lambda y,f: quantile_loss(1-self.delta, y,f), lambda y,f: quantile_loss(self.delta, y,f)]
            )  
            if self.synchronize:
                t_in = Input(shape=(input_shape[-1],))
                sin_out = sincoder(freq_init=self.freq_init, trainable_freq=self.trainable_freq)(t_in)
                self.sincoder = Model(inputs=t_in, outputs=sin_out)
                self.sincoder.compile(optimizer='adam', loss=lambda y,f: quantile_loss(0.5, y,f))

    def get_time_matrix(self, X):
        indices_linha = np.arange(len(X), dtype=np.float32).reshape(-1, 1)
        time_matrix = np.tile(indices_linha, (1, X.shape[1]))
        return time_matrix
    
    def normalization(self, x):
        reshape_data = x.reshape((x.shape[0]*x.shape[2], x.shape[1]))
        xscaler = MinMaxScaler()
        x_train_scaled = xscaler.fit_transform(reshape_data)
        return x_train_scaled
        
    def fit(
            self, 
            x_input: np.ndarray, 
            t: np.ndarray,
            epochs: int = 10, #100
            batch_size: int = 180, #360
            shuffle: bool = True, 
            freq_warmup: int = 5,  # number of warmup epochs to prefit the frequency = 10
            sin_warmup: int = 5,  # number of warmup epochs to prefit the sinusoidal representation = 10
            pos_amp: bool = True,  # whether to constraint amplitudes to be +ve only
    ):
        x = self.normalization(x_input)
        t = self.get_time_matrix(x)
        # Prepare the training batches.
        dataset = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), t.astype(np.float32)))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=x.shape[0]).batch(batch_size)
            
        # build and compile models (stage 1)
        if self.synchronize:
            self.build(x.shape, initial_stage=True)
            if self.freq_init:
                self.build(x.shape)
        else:
            self.build(x.shape)
        
        # pretraining step 1:
        if freq_warmup > 0 and self.synchronize and not self.freq_init:
            for epoch in range(freq_warmup):
                print("\nStart of frequency pre-train epoch %d" % (epoch,))
                for step, (x_batch, t_batch) in enumerate(dataset):
                    # Prefit the oscillation encoder
                    with tf.GradientTape() as tape:
                        # forward pass
                        z, x_pred = self.freqcoder(x_batch)
                        
                        # compute loss
                        x_loss = self.freqcoder.loss(x_batch, x_pred)  # median loss
                    
                    # retrieve gradients and update weights
                    grads = tape.gradient(x_loss, self.freqcoder.trainable_weights)
                    self.freqcoder.optimizer.apply_gradients(zip(grads, self.freqcoder.trainable_weights))
                print("pre-reconstruction_loss:", tf.reduce_mean(x_loss).numpy(), end='\r')
                
            # estimate dominant frequency
            z = self.freqcoder(x)[0].numpy().reshape(-1)  # must be done on full unshuffled series
            z = ((z - z.min()) / (z.max() - z.min())) * (1 - -1) + -1  #  scale between -1 & 1
            p = Periodogram(z, sampling=1)
            p()
            peak_idxs = find_peaks(p.psd, distance=self.min_dist, height=(0, np.inf))[0]
            peak_order = p.psd[peak_idxs].argsort()[-self.min_periods-self.max_freqs:][::-1]  # max PSDs found
            peak_idxs = peak_idxs[peak_order]
            if peak_idxs[0] < self.min_periods and not self.force_synchronization:
                self.synchronize = False
                print('no common oscillations found, switching off synchronization attempts')
            elif max(peak_idxs[:self.min_periods]) >= self.min_periods:
                idxs = peak_idxs[peak_idxs >= self.min_periods]
                peak_freqs = [p.frequencies()[idx] for idx in idxs[:min(len(idxs), self.max_freqs)]]
                self.freq_init = [2 * np.pi * f for f in peak_freqs]
                print('found common oscillations at period(s) = {}'.format([1 / f for f in peak_freqs]))
            else:
                self.synchronize = False
                print('no common oscillations found, switching off synchronization attempts')
            
            # build and compile models (stage 2)
            self.build(x.shape)
        
        # pretraining step 2:
        if sin_warmup > 0 and self.synchronize:
            for epoch in range(sin_warmup):
                print("\nStart of sine representation pre-train epoch %d" % (epoch,))
                for step, (x_batch, t_batch) in enumerate(dataset):
                    # Train the sine wave encoder
                    with tf.GradientTape() as tape:
                        # forward pass
                        s = self.sincoder(t_batch)
                        
                        # compute loss
                        s_loss = self.sincoder.loss(x_batch, s)  # median loss
                    
                    # retrieve gradients and update weights
                    grads = tape.gradient(s_loss, self.sincoder.trainable_weights)
                    self.sincoder.optimizer.apply_gradients(zip(grads, self.sincoder.trainable_weights))
                print("sine_loss:", tf.reduce_mean(s_loss).numpy(), end='\r')
            
            # invert params (all amplitudes should either be -ve or +ve). Here we make them +ve
            if pos_amp:
                a_adj = tf.where(
                    self.sincoder.layers[1].amp[:,0] < 0, 
                    self.sincoder.layers[1].amp[:,0] * -1, 
                    self.sincoder.layers[1].amp[:,0]
                )  # invert all -ve amplitudes
                wb_adj = tf.where(
                    self.sincoder.layers[1].amp[:,0] < 0, 
                    self.sincoder.layers[1].wb[:,0] + np.pi, 
                    self.sincoder.layers[1].wb[:,0]
                )  # shift inverted waves by half cycle
                wb_adj = tf.where(
                    wb_adj > 2*np.pi, self.sincoder.layers[1].wb[:,0] - np.pi, wb_adj
                )  # any cycle > freq must be reduced by half the cycle
                g_adj = tf.where(
                    self.sincoder.layers[1].amp[:,0] < 0, 
                    self.sincoder.layers[1].disp - a_adj, 
                    self.sincoder.layers[1].disp
                )  # adjust the vertical displacements after reversing amplitude signs
                K.set_value(self.sincoder.layers[1].amp[:,0], a_adj)
                K.set_value(self.sincoder.layers[1].wb[:,0], wb_adj)
                K.set_value(self.sincoder.layers[1].disp, g_adj)
                
        # train anomaly detector
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            if self.synchronize:
                for step, (x_batch, t_batch) in enumerate(dataset):
                    # Train the sine wave encoder
                    with tf.GradientTape() as tape:
                        # forward pass
                        s = self.sincoder(t_batch)
                        
                        # compute loss
                        s_loss = self.sincoder.loss(x_batch, s)  # median loss
                    
                    # retrieve gradients and update weights
                    grads = tape.gradient(s_loss, self.sincoder.trainable_weights)
                    self.sincoder.optimizer.apply_gradients(zip(grads, self.sincoder.trainable_weights))
                    
                    # synchronize batch
                    b = self.sincoder.layers[1].wb / self.sincoder.layers[1].freq  # phase shift(s)
                    b_sync = b - tf.expand_dims(b[:,0], axis=-1)
                    th_sync = tf.expand_dims(
                        tf.expand_dims(self.sincoder.layers[1].freq, axis=0), axis=0
                    ) * (tf.expand_dims(t_batch, axis=-1) + tf.expand_dims(b_sync, axis=0))  # synchronized angle
                    e = (
                        x_batch - s
                    ) * tf.sin(
                        self.sincoder.layers[1].freq[0] * ((np.pi / (2 * self.sincoder.layers[1].freq[0])) - b[:,0])
                    )  # noise
                    x_batch_sync = tf.reduce_sum(
                        tf.expand_dims(self.sincoder.layers[1].amp, axis=0) * tf.sin(th_sync), axis=-1
                    ) + self.sincoder.layers[1].disp + e
                    
                    # train the rancoders
                    with tf.GradientTape() as tape:
                        # forward pass
                        o_hi, o_lo = self.rancoders(x_batch_sync)
                        
                        # compute losses
                        o_hi_loss = self.rancoders.loss[0](
                            tf.tile(tf.expand_dims(x_batch_sync, axis=0), (self.n_estimators, 1, 1)), o_hi
                        )
                        o_lo_loss = self.rancoders.loss[1](
                            tf.tile(tf.expand_dims(x_batch_sync, axis=0), (self.n_estimators, 1, 1)), o_lo
                        )
                        o_loss = o_hi_loss + o_lo_loss

                    # retrieve gradients and update weights
                    grads = tape.gradient(o_loss, self.rancoders.trainable_weights)
                    self.rancoders.optimizer.apply_gradients(zip(grads, self.rancoders.trainable_weights))
                print(
                    "sine_loss:", tf.reduce_mean(s_loss).numpy(), 
                    "upper_bound_loss:", tf.reduce_mean(o_hi_loss).numpy(), 
                    "lower_bound_loss:", tf.reduce_mean(o_lo_loss).numpy(), 
                    end='\r'
                )
            else:
                for step, (x_batch, t_batch) in enumerate(dataset):
                   # train the rancoders
                    with tf.GradientTape() as tape:
                        # forward pass
                        o_hi, o_lo = self.rancoders(x_batch)
                        
                        # compute losses
                        o_hi_loss = self.rancoders.loss[0](
                            tf.tile(tf.expand_dims(x_batch, axis=0), (self.n_estimators, 1, 1)), o_hi
                        )
                        o_lo_loss = self.rancoders.loss[1](
                            tf.tile(tf.expand_dims(x_batch, axis=0), (self.n_estimators, 1, 1)), o_lo
                        )
                        o_loss = o_hi_loss + o_lo_loss

                    # retrieve gradients and update weights
                    grads = tape.gradient(o_loss, self.rancoders.trainable_weights)
                    self.rancoders.optimizer.apply_gradients(zip(grads, self.rancoders.trainable_weights))
                print(
                    "upper_bound_loss:", tf.reduce_mean(o_hi_loss).numpy(), 
                    "lower_bound_loss:", tf.reduce_mean(o_lo_loss).numpy(), 
                    end='\r'
                )
            
    def predict(self, x: np.ndarray, batch_size: int = 1000, desync: bool = False):
        t = self.get_time_matrix(x)
        # Prepare the training batches.
        dataset = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), t.astype(np.float32)))
        dataset = dataset.batch(batch_size)
        batches = int(np.ceil(x.shape[0] / batch_size))
        
        # loop through the batches of the dataset.
        if self.synchronize:
            s, x_sync, o_hi, o_lo = [None] * batches, [None] * batches, [None] * batches, [None] * batches
            for step, (x_batch, t_batch) in enumerate(dataset):
                s_i = self.sincoder(t_batch).numpy()
                b = self.sincoder.layers[1].wb / self.sincoder.layers[1].freq  # phase shift(s)
                b_sync = b - tf.expand_dims(b[:,0], axis=-1)
                th_sync = tf.expand_dims(
                    tf.expand_dims(self.sincoder.layers[1].freq, axis=0), axis=0
                ) * (tf.expand_dims(t_batch, axis=-1) + tf.expand_dims(b_sync, axis=0))  # synchronized angle
                e = (
                    x_batch - s_i
                ) * tf.sin(
                    self.sincoder.layers[1].freq[0] * ((np.pi / (2 * self.sincoder.layers[1].freq[0])) - b[:,0])
                )  # noise
                x_sync_i = (tf.reduce_sum(
                    tf.expand_dims(self.sincoder.layers[1].amp, axis=0) * tf.sin(th_sync), axis=-1
                ) + self.sincoder.layers[1].disp + e).numpy()  
                o_hi_i, o_lo_i = self.rancoders(x_sync_i)
                o_hi_i, o_lo_i = tf.transpose(o_hi_i, [1,0,2]).numpy(), tf.transpose(o_lo_i, [1,0,2]).numpy()
                if desync:
                    o_hi_i, o_lo_i = self.predict_desynchronize(x_batch, x_sync_i, o_hi_i, o_lo_i)
                s[step], x_sync[step], o_hi[step], o_lo[step]  = s_i, x_sync_i, o_hi_i, o_lo_i
            return (
                np.concatenate(s, axis=0), 
                np.concatenate(x_sync, axis=0), 
                np.concatenate(o_hi, axis=0), 
                np.concatenate(o_lo, axis=0)
            )
        else:
            o_hi, o_lo = [None] * batches, [None] * batches
            for step, (x_batch, t_batch) in enumerate(dataset):
                o_hi_i, o_lo_i = self.rancoders(x_batch)
                o_hi_i, o_lo_i = tf.transpose(o_hi_i, [1,0,2]).numpy(), tf.transpose(o_lo_i, [1,0,2]).numpy()
                o_hi[step], o_lo[step]  = o_hi_i, o_lo_i
            return np.concatenate(o_hi, axis=0), np.concatenate(o_lo, axis=0)
    
    def score_samples(self, X):
        x_scaled = self.normalization(X)
        sins, synched, upper, lower = self.predict(x_scaled)
        synched_tiles = np.tile(synched.reshape(synched.shape[0], 1, synched.shape[1]), (1, 10, 1))
        result = np.where((synched_tiles < lower) | (synched_tiles > upper), 1, 0)
        return np.mean(result)

    def normalization2(self, x):
        reshape_data = x.reshape((x.shape[0]*x.shape[2], x.shape[1]))
        xscaler = MinMaxScaler()
        x_train_scaled = xscaler.transform(reshape_data)
        return reshape_data

    def save(self, filepath: str = os.path.join(os.getcwd(), 'ransyncoders.z')):
        file = {'params': self.get_config()}
        if self.synchronize:
            file['freqcoder'] = {'model': self.freqcoder.to_json(), 'weights': self.freqcoder.get_weights()}
            file['sincoder'] = {'model': self.sincoder.to_json(), 'weights': self.sincoder.get_weights()}
        file['rancoders'] = {'model': self.rancoders.to_json(), 'weights': self.rancoders.get_weights()}
        dump(file, filepath, compress=True)
    
    @classmethod
    def load(cls, filepath: str = os.path.join(os.getcwd(), 'ransyncoders.z')):
        file = load(filepath)
        cls = cls()
        for param, val in file['params'].items():
            setattr(cls, param, val)
        if cls.synchronize:
            cls.freqcoder = model_from_json(file['freqcoder']['model'], custom_objects={'freqcoder': freqcoder})
            cls.freqcoder.set_weights(file['freqcoder']['weights'])
            cls.sincoder = model_from_json(file['sincoder']['model'], custom_objects={'sincoder': sincoder})
            cls.sincoder.set_weights(file['sincoder']['weights'])
        cls.rancoders = model_from_json(file['rancoders']['model'], custom_objects={'RANCoders': RANCoders})  
        cls.rancoders.set_weights(file['rancoders']['weights'])
        return cls
    
    def predict_desynchronize(self, x: np.ndarray, x_sync: np.ndarray, o_hi: np.ndarray, o_lo: np.ndarray):
        if self.synchronize:
            E = (o_hi + o_lo)/ 2  # expected values
            deviation = tf.expand_dims(x_sync, axis=1) - E  # input (synchronzied) deviation from expected
            deviation = self.desynchronize(deviation)  # desynchronize
            E = tf.expand_dims(x, axis=1) - deviation  # expected values in desynchronized form
            offset = (o_hi - o_lo) / 2  # this is the offet from the expected value
            offset = abs(self.desynchronize(offset))  # desynch
            o_hi, o_lo = E + offset, E - offset  # add bound displacement to expected values
            return o_hi.numpy(), o_lo.numpy()  
        else:
            raise ParameterError('synchronize', 'parameter not set correctly for this method')
    
    def desynchronize(self, e: np.ndarray):
        if self.synchronize:
            b = self.sincoder.layers[1].wb / self.sincoder.layers[1].freq  # phase shift(s)
            return e * tf.sin(
                self.sincoder.layers[1].freq[0] * ((np.pi / (2 * self.sincoder.layers[1].freq[0])) + b[:,0])
            ).numpy()
        else:
            raise ParameterError('synchronize', 'parameter not set correctly for this method')
        
        
    def get_config(self):
        config = {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "encoding_depth": self.encoding_depth,
            "latent_dim": self.encoding_depth,
            "decoding_depth": self.decoding_depth,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "delta": self.delta,
            "synchronize": self.synchronize,
            "force_synchronization": self.force_synchronization,
            "min_periods": self.min_periods,
            "freq_init": self.freq_init,
            "max_freqs": self.max_freqs,
            "min_dist": self.min_dist,
            "trainable_freq": self.trainable_freq,
            "bias": self.bias,
        }
        return config
           
# Loss function
def quantile_loss(q, y, f):
    e = (y - f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

class ParameterError(Exception):

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

FINAL_MODEL = RANSynCodersBase()

class RANSynCoders(BaseEstimator, OutlierMixin):
    def __init__(self, 
                 final_model = FINAL_MODEL, 
                 sampling_rate = None,
                 mono: bool = True,
                ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.mono = mono
        self.final_model = final_model
        self.model = self.build_model()

    #@property
    def name(self):
        return "RANSynCoder " + "+".join([f[0] for f in self.features])
        
    def fit(self, X, y=None):
        return self.model.fit(X, y)
    
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
        

    def build_model(self):
        wav2array = Wav2Array(sampling_rate=self.sampling_rate, mono=self.mono)
        
        model = Pipeline(
            steps=[
                ("wav2array", wav2array),
                ("final_model", self.final_model),
                ]
            )
        
        return model
# ==============================================================================================================================
# SINCODER
# ==============================================================================================================================
class freqcoder(Layer):
    """ 
    Encode multivariate to a latent space of size 1 for extracting common oscillations in the series (similar to finding PCA).
    """
    def __init__(self, **kwargs):
        super(freqcoder, self).__init__(**kwargs)
        self.kwargs = kwargs
        
    def build(self, input_shape):
        self.latent = Dense(1, activation='linear')
        self.decoder = Dense(input_shape[-1], activation='linear')
    
    def call(self, inputs):
        z = self.latent(inputs)
        x_pred = self.decoder(z)
        return z, x_pred
    
    def get_config(self):
        base_config = super(freqcoder, self).get_config()
        return dict(list(base_config.items()))
    
class sincoder(Layer):
    """ Fit m sinusoidal waves to an input t-matrix (matrix of m epochtimes) """
    def __init__(self, freq_init: Optional[List[float]] = None, max_freqs: int = 1, trainable_freq: bool = False, **kwargs):
        super(sincoder, self).__init__(**kwargs)
        self.freq_init = freq_init
        if freq_init:
            self.max_freqs = len(freq_init)
        else:
            self.max_freqs = max_freqs
        self.trainable_freq = trainable_freq
        self.kwargs = kwargs
        
    def build(self, input_shape):
        self.amp = self.add_weight(shape=(input_shape[-1], self.max_freqs), initializer="zeros", trainable=True)
        if self.freq_init and not self.trainable_freq:
            self.freq = [self.add_weight(initializer=Constant(f), trainable=False) for f in self.freq_init]
        elif self.freq_init:
            self.freq = [self.add_weight(initializer=Constant(f), constraint=NonNeg(), trainable=True) for f in self.freq_init]
        else:
            self.freq = [
                self.add_weight(initializer="zeros", constraint=NonNeg(), trainable=True) for f in range(self.max_freqs)
            ]
        self.wb = self.add_weight(
            shape=(input_shape[-1], self.max_freqs), initializer="zeros", trainable=True
        )  # angular frequency (w) x phase shift
        self.disp = self.add_weight(shape=input_shape[-1], initializer="zeros", trainable=True)
    
    def call(self, inputs):
        th = tf.expand_dims(
            tf.expand_dims(self.freq, axis=0), axis=0
        ) * tf.expand_dims(inputs, axis=-1) + tf.expand_dims(self.wb, axis=0)
        return tf.reduce_sum(tf.expand_dims(self.amp, axis=0) * tf.sin(th), axis=-1) + self.disp
    
    def get_config(self):
        base_config = super(sincoder, self).get_config()
        config = {"freq_init": self.freq_init, "max_freqs": self.max_freqs, "trainable_freq": self.trainable_freq}
        return dict(list(base_config.items()) + list(config.items()))

# ==============================================================================================================================
# RANCODER
# ==============================================================================================================================
class Encoder(Layer):
    def __init__(self, latent_dim: int, activation: str, depth: int = 2, **kwargs,):
        super(Encoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.activation = activation
        self.depth = depth
        self.kwargs = kwargs
        
    def build(self, input_shape):
        self.hidden = {
            'hidden_{}'.format(i): Dense(
                int(input_shape[-1] / (2**(i+1))), activation=self.activation,
            ) for i in range(self.depth)
        }
        self.latent = Dense(self.latent_dim, activation=self.activation)
        
    def call(self, inputs):
        x = self.hidden['hidden_0'](inputs)
        for i in range(1, self.depth):
            x = self.hidden['hidden_{}'.format(i)](x)
        return self.latent(x)
    
    def get_config(self):
        base_config = super(Encoder, self).get_config()
        config = {"latent_dim": self.latent_dim, "activation": self.activation,"depth": self.depth,}
        return dict(list(base_config.items()) + list(config.items()))
    
class Decoder(Layer):
    def __init__(self, output_dim: int, activation: str, output_activation: str,depth: int, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation
        self.depth = depth
        self.kwargs = kwargs
        
    def build(self, input_shape):
        self.hidden = {
            'hidden_{}'.format(i): Dense(
                int(self.output_dim/ (2**(self.depth-i))), activation=self.activation,
            ) for i in range(self.depth)
        }
        self.restored = Dense(self.output_dim, activation=self.output_activation)
        
    def call(self, inputs):
        x = self.hidden['hidden_0'](inputs)
        for i in range(1, self.depth):
            x = self.hidden['hidden_{}'.format(i)](x)
        return self.restored(x)
    
    def get_config(self):
        base_config = super(Decoder, self).get_config()
        config = {
            "output_dim": self.output_dim, 
            "activation": self.activation, 
            "output_activation": self.output_activation, 
            "depth": self.depth,
        }
        return dict(list(base_config.items()) + list(config.items()))
    
class RANCoders(Layer):
    def __init__(
            self, 
            n_estimators: int = 100,
            max_features: int = 3,
            encoding_depth: int = 2,
            latent_dim: int = 2, 
            decoding_depth: int = 2,
            delta: float = 0.05,
            activation: str = 'linear',
            output_activation: str = 'linear',
            **kwargs,
    ):
        super(RANCoders, self).__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.encoding_depth = encoding_depth
        self.latent_dim = latent_dim
        self.decoding_depth = decoding_depth
        self.delta = delta
        self.activation = activation
        self.output_activation = output_activation
        self.kwargs = kwargs
        
    def build(self, input_shape):
        assert(input_shape[-1] > self.max_features)
        self.encoders = {
            'encoder_{}'.format(i): Encoder(
                self.latent_dim, self.activation, depth=self.encoding_depth,
            ) for i in range(self.n_estimators)
        }
        self.decoders_upper = {
            'decoder_hi_{}'.format(i): Decoder(
                input_shape[-1], self.activation, self.output_activation, self.decoding_depth
            ) for i in range(self.n_estimators)
        }
        self.decoders_lower = {
            'decoder_lo_{}'.format(i): Decoder(
                input_shape[-1], self.activation, self.output_activation, self.decoding_depth
            ) for i in range(self.n_estimators)
        }
        self.randsamples = tf.Variable(
            np.concatenate(
                [
                    np.random.choice(
                        input_shape[-1], replace=False, size=(1, self.max_features),
                    ) for i in range(self.n_estimators)
                ]
            ), trainable=False
        )  # the feature selector (bootstrapping)
        
    def call(self, inputs):
        z = {
            'z_{}'.format(i): self.encoders['encoder_{}'.format(i)](
                tf.gather(inputs, self.randsamples[i], axis=-1)
            ) for i in range(self.n_estimators)
        }
        w_hi = {
            'w_{}'.format(i): self.decoders_upper['decoder_hi_{}'.format(i)](
                z['z_{}'.format(i)]
            ) for i in range(self.n_estimators)
        }
        w_lo = {
            'w_{}'.format(i): self.decoders_lower['decoder_lo_{}'.format(i)](
                z['z_{}'.format(i)]
            ) for i in range(self.n_estimators)
        }
        o_hi = tf.concat([tf.expand_dims(i, axis=0) for i in w_hi.values()], axis=0)  
        o_lo = tf.concat([tf.expand_dims(i, axis=0) for i in w_lo.values()], axis=0)
        return o_hi, o_lo
    
    def get_config(self):
        base_config = super(RANCoders, self).get_config()
        config = {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "encoding_depth": self.encoding_depth,
            "latent_dim": self.latent_dim,
            "decoding_depth": self.decoding_depth,
            "delta": self.delta,
            "activation": self.activation,
            "output_activation": self.output_activation,
        }
        return dict(list(base_config.items()) + list(config.items()))       

