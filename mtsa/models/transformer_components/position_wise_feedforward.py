import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionWiseFeedForward, self).__init__()
    self.dropout_layer = nn.Dropout(p=dropout)
    self.fc_1 = nn.Linear(d_model, d_ff)
    self.fc_2 = nn.Linear(d_ff, d_model)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.fc_1(x)
    out = self.dropout_layer(self.relu(out))
    return self.fc_2(out)