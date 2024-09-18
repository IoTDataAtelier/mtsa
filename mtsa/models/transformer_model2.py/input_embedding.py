import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        #Multiplicação pela raiz de d_model para estabilização do gradiente (Page 5, 3.4)
        return self.embedding(x) * math.sqrt(self.d_model)