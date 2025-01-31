import torch.nn as nn
from mtsa.models.transformer_components.multi_head_attention import MultiHeadAttention
from mtsa.models.transformer_components.position_wise_feedforward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super(DecoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.cross_attn = MultiHeadAttention(d_model, num_heads)
    self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

  def forward(self, x, enc_output, src_mask, tgt_mask):
    attn_output = self.self_attn(x, x, x, tgt_mask)
    x = self.norm1(x + self.dropout1(attn_output))
    attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
    x = self.norm2(x + self.dropout2(attn_output))
    ff_output = self.feed_forward(x)
    x = self.norm3(x + self.dropout3(ff_output))
    return x