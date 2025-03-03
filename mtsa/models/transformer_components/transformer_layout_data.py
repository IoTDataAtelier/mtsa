from torch.utils.data import Dataset
import torch
import numpy as np

class TransformerData(Dataset):
  def __init__(self, X, y=None, device="cuda"):

    self.device = device
    self.scalar_paths = torch.tensor(X, device=self.device) 

    if y is not None:
      self.labels = torch.tensor(y, dtype=torch.float32, device=self.device)
    else:
      self.labels = None 
    
  def __getitem__(self, idx):
    if self.labels is not None:
      return self.scalar_paths[idx], self.labels[idx]
    return self.scalar_paths[idx]

  def __len__(self):
    return len(self.scalar_paths)