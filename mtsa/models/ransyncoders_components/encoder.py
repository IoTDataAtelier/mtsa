from tensorflow.python.keras.layers import Dense, Layer

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