import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length=3000, dropout=0.1):
    super(PositionalEncoding, self).__init__()
    self.dropout_layer = nn.Dropout(p=dropout)

    pe = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    self.register_buffer('pe', pe.unsqueeze(0))
  
  def forward(self, x):
    return self.dropout_layer(x + self.pe[:, :x.size(1)])