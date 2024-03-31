from tensorflow.python.keras.layers import Dense, Layer

class FrequencyCoder(Layer):
    """ 
    Encode multivariate to a latent space of size 1 for extracting common oscillations in the series (similar to finding PCA).
    """
    def __init__(self, **kwargs):
        super(FrequencyCoder, self).__init__(**kwargs)
        self.kwargs = kwargs
        
    def build(self, input_shape):
        self.latent = Dense(1, activation='linear')
        self.decoder = Dense(input_shape[-1], activation='linear')
    
    def call(self, inputs):
        z = self.latent(inputs)
        x_pred = self.decoder(z)
        return z, x_pred
    
    def get_config(self):
        base_config = super(FrequencyCoder, self).get_config()
        return dict(list(base_config.items()))