import torch.nn as nn
from mtsa.models.transformer_components.multi_head_attention import MultiHeadAttention
from mtsa.models.transformer_components.position_wise_feedforward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super(EncoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x, mask):
    attn_output = self.self_attn(x, x, x, mask)
    x = self.norm1(x + self.dropout1(attn_output))
    ff_output = self.feed_forward(x)
    x = self.norm2(x + self.dropout2(ff_output))
    return x