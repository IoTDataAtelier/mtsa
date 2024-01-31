import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from mtsa.models.ransyncoders_components.decoder import Decoder
from mtsa.models.ransyncoders_components.encoder import Encoder

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
        assert(input_shape[-1] >= self.max_features)
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