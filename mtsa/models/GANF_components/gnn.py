import torch.nn as nn
import torch.nn.functional as F
import torch

class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """
    def __init__(self, input_size, hidden_size):
        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, adjacent_matrix):
        ## adjacent_matrix: K X K
        ## H: N X K  X L X D

        h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,adjacent_matrix))
        h_r = self.lin_r(h[:,:,:-1])
        h_n[:,:,1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h