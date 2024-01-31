import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import Model, model_from_json
from joblib import dump, load
from scipy.signal import find_peaks
from tensorflow.python.keras.layers import Input
from spectrum import Periodogram
from typing import List, Optional
from mtsa.models.ransyncoders_components.rancoders import RANCoders
from mtsa.models.ransyncoders_components.frequencycoder import FrequencyCoder
from mtsa.models.ransyncoders_components.sinusoidalcoder import SinusoidalCoder
from mtsa.models.ransyncoders_components.exceptions.parametererror import ParameterError

class RANSynCodersBase():
    def __init__(
            self,
            n_estimators: int,
            max_features: int,
            encoding_depth: int,
            latent_dim: int,
            decoding_depth: int,
            activation: str,
            output_activation: str,
            delta: float,
            synchronize: bool,
            force_synchronization: bool,
            min_periods: int,
            freq_init: Optional[List[float]],
            max_freqs: int,
            min_dist: int,
            trainable_freq: bool,
            bias: bool,
            sampling_rate: int,
            mono: bool,
            is_acoustic_data: bool,
            normal_classifier: int,
            abnormal_classifier: int,
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
        self.sampling_rate = sampling_rate
        self.mono = mono
        self.is_acoustic_data = is_acoustic_data
        self.normal_classifier = normal_classifier
        self.abnormal_classifier = abnormal_classifier
        self.min_max_scaler = MinMaxScaler()
        # set all variables to default to float32
        tf.keras.backend.set_floatx('float32')
    
    #Region: Public methods      
    def fit(self, 
            X: np.ndarray,
            y, 
            timestamps_matrix: np.ndarray,
            learning_rate: float,
            batch_size: int, #old default value = 360
            epochs: int, #old default value = 100 
            freq_warmup: int,  # number of warmup epochs to prefit the frequency = 10 (old default value)
            sin_warmup: int,  # number of warmup epochs to prefit the sinusoidal representation = 10 (old default value)
            pos_amp: bool,  # whether to constraint amplitudes to be +ve only
            shuffle: bool
        ):
        self.timestamps_matrix = timestamps_matrix
        self.batch_size = batch_size
        self.experiment_dataframe = self.__get_experiment_dataframe()

        if self.is_acoustic_data:
            self.__train_model_with_audio_data(X, learning_rate, epochs, freq_warmup, sin_warmup, pos_amp) 
        else:
            self.__train_model_with_default_data(X, timestamps_matrix, learning_rate, epochs, freq_warmup, sin_warmup, pos_amp, shuffle)
             
    def predict(self, X: np.ndarray, time_matrix: np.ndarray, desync: bool = False):
        if self.is_acoustic_data:
            return self.__predict_with_audio_data(X, desync)
        return self.__predict_with_default_data(X, time_matrix, desync)
                       
    def build(self, input_shape, learning_rate: float, initial_stage: bool = False):
        x_in = Input(shape=(input_shape[-1],))  # created for either raw signal or synchronized signal
        optmizer = adam_v2.Adam(learning_rate = learning_rate)
        if initial_stage:
            freq_out = FrequencyCoder()(x_in)
            self.freqcoder = Model(inputs=x_in, outputs=freq_out)
            self.freqcoder.compile(optimizer=optmizer, loss=lambda y,f: quantile_loss(0.5, y,f))
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
                    optimizer=optmizer, 
                    loss=[lambda y,f: quantile_loss(1-self.delta, y,f), lambda y,f: quantile_loss(self.delta, y,f)]
            )  
            K.set_value(self.rancoders.optimizer.learning_rate, 0.01)
            if self.synchronize:
                t_in = Input(shape=(input_shape[-1],))
                sin_out = SinusoidalCoder(freq_init=self.freq_init, trainable_freq=self.trainable_freq)(t_in)
                self.sincoder = Model(inputs=t_in, outputs=sin_out)
                self.sincoder.compile(optimizer=optmizer, loss=lambda y,f: quantile_loss(0.5, y,f))
                
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
    
    def score_samples(self, X):
        sins, synched, upper, lower = self.predict(X, self.timestamps_matrix)
        synched_tiles = np.tile(synched.reshape(synched.shape[0], 1, synched.shape[1]), (1, self.n_estimators, 1))
        result = np.where((synched_tiles < lower) | (synched_tiles > upper), self.abnormal_classifier, self.normal_classifier)
        
        if self.is_acoustic_data:
           return np.mean(result)
        return np.mean(np.mean(result, axis=1), axis=1) 
    
    def set_timestamps_matrix_to_predict(self, timestamps_matrix: np.ndarray):
        self.timestamps_matrix = timestamps_matrix
    
    def save(self, filepath: str = os.path.join(os.getcwd(), 'ransyncoders.z')):
        file = {'params': self.get_config()}
        if self.synchronize:
            file['frequencycoder'] = {'model': self.freqcoder.to_json(), 'weights': self.freqcoder.get_weights()}
            file['sinusoidalcoder'] = {'model': self.sincoder.to_json(), 'weights': self.sincoder.get_weights()}
        file['rancoders'] = {'model': self.rancoders.to_json(), 'weights': self.rancoders.get_weights()}
        dump(file, filepath, compress=True)
    
    @classmethod
    def load(cls, filepath: str = os.path.join(os.getcwd(), 'ransyncoders.z')):
        file = load(filepath)
        cls = cls()
        for param, val in file['params'].items():
            setattr(cls, param, val)
        if cls.synchronize:
            cls.freqcoder = model_from_json(file['frequencycoder']['model'], custom_objects={'FrequencyCoder': FrequencyCoder})
            cls.freqcoder.set_weights(file['frequencycoder']['weights'])
            cls.sincoder = model_from_json(file['sinusoidalcoder']['model'], custom_objects={'SinusoidalCoder': SinusoidalCoder})
            cls.sincoder.set_weights(file['sinusoidalcoder']['weights'])
        cls.rancoders = model_from_json(file['rancoders']['model'], custom_objects={'RANCoders': RANCoders})  
        cls.rancoders.set_weights(file['rancoders']['weights'])
        return cls
    
    def get_config(self):
        return self.experiment_dataframe
    #End region

    #Region: Private Methods
    def __train_model_with_default_data(self, X: np.ndarray, timestamps_matrix: np.ndarray, learning_rate, epochs, freq_warmup, sin_warmup, pos_amp, shuffle):
        
        if timestamps_matrix is None:
            timestamps_matrix = self.__get_time_matrix(X=X)
        
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), timestamps_matrix.astype(np.float32)))
            if shuffle:
                dataset = dataset.shuffle(buffer_size=X.shape[0]).batch(self.batch_size)
            else:
                dataset = dataset.batch(self.batch_size)
                
        self.__build_models(X, learning_rate)
                
        self.__train_ransyncoders_model(X, learning_rate, epochs, freq_warmup, sin_warmup, pos_amp, dataset)

    def __train_model_with_audio_data(self, X, learning_rate, epochs, freq_warmup, sin_warmup, pos_amp):
        
        datasets, X = self.__load_dataset(X)
        
        # build and compile models (stage 1)
        self.__build_models(X[0], learning_rate)
        
        for index, dataset in enumerate(datasets):
            self.__train_ransyncoders_model(X[index], learning_rate, epochs, freq_warmup, sin_warmup, pos_amp, dataset)

    def __train_ransyncoders_model(self, X, learning_rate, epochs, freq_warmup, sin_warmup, pos_amp, dataset):
        # pretraining step 1:
        self.__train_autoencoder(X, learning_rate, freq_warmup, dataset)
            
        # pretraining step 2:    
        self.__train_synchronization(sin_warmup, pos_amp, dataset)
                    
        # train anomaly detector   
        self.__train_anomaly_detector(learning_rate, epochs, dataset)
    
    def __train_anomaly_detector(self, learning_rate, epochs, dataset):
        with tf.device('/gpu:1'):
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

                        upper_bound_loss = tf.reduce_mean(o_hi_loss).numpy()
                        lower_bound_loss = tf.reduce_mean(o_lo_loss).numpy()
                    print(
                            "sine_loss:", tf.reduce_mean(s_loss).numpy(), 
                            "upper_bound_loss:", upper_bound_loss, 
                            "lower_bound_loss:", lower_bound_loss, 
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

                        upper_bound_loss = tf.reduce_mean(o_hi_loss).numpy()
                        lower_bound_loss = tf.reduce_mean(o_lo_loss).numpy()
                    print(
                            "upper_bound_loss:", upper_bound_loss, 
                            "lower_bound_loss:", lower_bound_loss, 
                            end='\r'
                        )

                self.experiment_dataframe.loc[len(self.experiment_dataframe)] = {"epoch_size": epochs,
                                                                                 "sampling_rate": self.sampling_rate,
                                                                                 "learning_rate": learning_rate,
                                                                                 "batch_size": self.batch_size,
                                                                                 "n_estimators": self.n_estimators,
                                                                                 "max_features": self.max_features,
                                                                                 "encoding_depth": self.encoding_depth,
                                                                                 "latent_dim": self.latent_dim,
                                                                                 "decoding_depth": self.decoding_depth,
                                                                                 "activation": self.decoding_depth,
                                                                                 "output_activation": self.decoding_depth,
                                                                                 "delta": self.delta,
                                                                                 "synchronize": self.synchronize,
                                                                                 "upper_bound_loss": upper_bound_loss,
                                                                                 "lower_bound_loss": lower_bound_loss,
                                                                                 "AUC_ROC": None,
                                                                                 "Confidence_interval_AUC_ROC": None
                                                                                }
    
    def __build_models(self, X, learning_rate):
        with tf.device('/cpu:0'):
            if self.synchronize:
                self.build(X.shape, initial_stage = True, learning_rate = learning_rate)
                if self.freq_init:
                    self.build(X.shape, learning_rate = learning_rate)
            else:
                self.build(X.shape, learning_rate = learning_rate)

    def __train_synchronization(self, sin_warmup, pos_amp, dataset):
        if sin_warmup > 0 and self.synchronize:
            with tf.device('/gpu:1'):
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

    def __train_autoencoder(self, X, learning_rate, freq_warmup, dataset):
        if freq_warmup > 0 and self.synchronize and not self.freq_init:
            with tf.device('/gpu:1'):
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
            with tf.device('/cpu:0'):
                z = self.freqcoder(X)[0].numpy().reshape(-1)  # must be done on full unshuffled series
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
            with tf.device('/cpu:0'):
                self.build(X.shape, learning_rate = learning_rate)
    
    def __normalize_data(self, X: np.ndarray, training_step: bool):
        if(training_step):
            return self.min_max_scaler.fit_transform(X)
        return self.min_max_scaler.transform(X)
        
    def __get_time_matrix(self, X):
        #each line represents an instant of time in each time series
        total_time_in_seconds, time_serie_time_instant = self.__calculate_time_serie_time_instatnt(X)
        time_matrix = np.arange(start=0, stop=total_time_in_seconds, step=time_serie_time_instant)
        return np.tile(time_matrix.reshape(-1, 1), (1, X.shape[1]))

    def __calculate_time_serie_time_instatnt(self, X):
        X_length = X.shape[0]
        #REMAKE
        dividend = self.sampling_rate
        if X_length < self.sampling_rate:
            dividend = X_length
            
        total_time_in_seconds = X_length/dividend
        time_serie_time_instant = total_time_in_seconds/X_length
        return total_time_in_seconds,time_serie_time_instant

    def __load_dataset(self, X: np.ndarray, training_step: bool = True):
        datasets = []
        x_2D = []
        
        for x in X:
            x = self.__transform_X(x)
            time_matrix = self.__get_time_matrix(x)
            x_normalized = self.__normalize_data(x, training_step)
            x_2D.append(x_normalized)
            with tf.device('/cpu:0'):
                dataset = tf.data.Dataset.from_tensor_slices((x_normalized.astype(np.float32), time_matrix.astype(np.float32)))
                datasets.append(dataset.batch(self.batch_size))  
                    
        return datasets, x_2D

    def __transform_X(self, x):
        if self.mono and (not self.is_acoustic_data):
            x = x.reshape(-1, 1)
        else:
            x = x.T
        return x
    
    def __get_experiment_dataframe(self):
        parameters_columns = ["epoch_size",
                              "sampling_rate",
                              "learning_rate",
                              "batch_size",
                              "n_estimators",
                              "max_features",
                              "encoding_depth",
                              "latent_dim",
                              "decoding_depth",
                              "activation",
                              "output_activation",
                              "delta",
                              "synchronize",
                              "upper_bound_loss",
                              "lower_bound_loss",
                              "AUC_ROC",
                              "Confidence_interval_AUC_ROC"
                            ]               
        return pd.DataFrame(columns=parameters_columns)
                
    def __predict_with_audio_data(self, X: np.ndarray, desync):
        
        datasets, X = self.__load_dataset(X, False)
        batches = int(np.ceil(X[0].shape[0] / self.batch_size))
        
        return self.__model_predict(datasets[0], batches, desync)
            
    def __predict_with_default_data(self, X: np.ndarray, timestamps_matrix: np.ndarray, desync):
        with tf.device('/gpu:1'):
            if timestamps_matrix is None:
                timestamps_matrix = self.__get_time_matrix(X=X)
                
            dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), timestamps_matrix.astype(np.float32)))
            dataset = dataset.batch(self.batch_size)
            batches = int(np.ceil(X.shape[0] / self.batch_size))
            return self.__model_predict(dataset, batches, desync)
           
    def __model_predict(self, dataset, batches, desync):
        with tf.device('/gpu:1'):
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
    #End region
    
# Loss function
def quantile_loss(q, y, f):
    e = (y - f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)