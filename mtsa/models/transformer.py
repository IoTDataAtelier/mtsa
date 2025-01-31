import torch
import torch.nn as nn
from mtsa.models.transformer_components.positional_encoding import PositionalEncoding
from mtsa.models.transformer_components.encoder_layer import EncoderLayer
from mtsa.models.transformer_components.decoder_layer import DecoderLayer

class Transformer(nn.Module):
  def __init__(self, mfcc_dim, d_model, nhead, num_layers, d_ff, max_seq_length, dropout, device):
    super(Transformer, self).__init__()
    self.device = device
    self.src_linear_input = nn.Linear(mfcc_dim, d_model).to(device)
    self.tgt_linear_input = nn.Linear(mfcc_dim, d_model).to(device)
    self.model_output = nn.Linear(d_model, mfcc_dim).to(device)

    self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout).to(device)

    self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]).to(device)
    self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]).to(device)

    self.dropout1 = nn.Dropout(dropout).to(device)
    self.dropout2 = nn.Dropout(dropout).to(device)

  def generate_mask(self, tgt, device):
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
    tgt_mask = nopeak_mask
    return tgt_mask

  def forward(self, src, tgt):
    src = self.src_linear_input(src)
    tgt = self.tgt_linear_input(tgt)

    src = self.positional_encoding(src)
    tgt = self.positional_encoding(tgt)

    src = self.dropout1(src)
    tgt = self.dropout2(tgt)
    tgt_m = self.generate_mask(tgt, self.device)
    enc_output = src
    for enc_layer in self.encoder_layers:
      enc_output = enc_layer(enc_output, mask=None)

    dec_output = tgt
    for dec_layer in self.decoder_layers:
      dec_output = dec_layer(dec_output, enc_output, src_mask=None, tgt_mask=tgt_m)

    output = self.model_output(dec_output)
    return output