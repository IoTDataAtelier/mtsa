from tensorflow.python.keras.layers import Dense, Layer

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