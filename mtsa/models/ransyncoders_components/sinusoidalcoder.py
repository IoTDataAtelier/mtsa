from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.layers import Layer
from typing import List, Optional
from tensorflow.python.keras.constraints import NonNeg
import tensorflow as tf

class SinusoidalCoder(Layer):
    """ Fit m sinusoidal waves to an input t-matrix (matrix of m epochtimes) """
    def __init__(self, freq_init: Optional[List[float]] = None, max_freqs: int = 1, trainable_freq: bool = False, **kwargs):
        super(SinusoidalCoder, self).__init__(**kwargs)
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
        base_config = super(SinusoidalCoder, self).get_config()
        config = {"freq_init": self.freq_init, "max_freqs": self.max_freqs, "trainable_freq": self.trainable_freq}
        return dict(list(base_config.items()) + list(config.items()))