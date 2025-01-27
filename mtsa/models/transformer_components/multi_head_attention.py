import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()

    # check if the d_model is divisible by num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    #initialize dimension
    self.d_model = d_model # Model's Dimension
    self.num_heads = num_heads # Number of attention heads
    self.d_k = d_model // num_heads # Dimension of each head's key, query and value

    self.W_q = nn.Linear(d_model, d_model) # query transformation
    self.W_k = nn.Linear(d_model, d_model) # key transformation
    self.W_v = nn.Linear(d_model, d_model) # value transformation
    self.W_o = nn.Linear(d_model, d_model) # output transformation

  def scaled_dot_product_attention(self, Q, K ,V, mask=None):
    # calculate attention scores
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

    # Apply mask if provided (useful for preventing attention to certain parts like padding)
    if mask is not None:
      attn_scores = attn_scores.masked_fill(mask==0, -1e9)
    
    # softmax is applied to obtain attention probabilities
    attn_scores = torch.softmax(attn_scores, dim=-1)

    # multiply by values to obtain final output
    output = torch.matmul(attn_scores, V)
    return output
  
  def split_heads(self, x):
    # Reshape the input to have num_heads for multi-head attention
    batch_size, seq_length, d_model = x.size()
    return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

  def combine_heads(self, x):
    # Combine the multiple heads back to original shape
    batch_size, _, seq_length, d_k = x.size()
    return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

  def forward(self, Q, K, V, mask=None):
    # Apply linear transformations and split heads
    Q = self.split_heads(self.W_q(Q))
    K = self.split_heads(self.W_k(K))
    V = self.split_heads(self.W_v(V))
    
    # Perform scaled dot-product attention
    attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
    
    # Combine heads and apply output transformation
    output = self.W_o(self.combine_heads(attn_output))
    return output
