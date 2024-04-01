
from sklearn.base import BaseEstimator, OutlierMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from functools import reduce
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from mtsa.models.GANF_model_components.NF import MAF, RealNVP
from functools import reduce
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_value_
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from mtsa.utils import Wav2Array
from sklearn.pipeline import Pipeline
import pandas as pd

from mtsa.utils import Wav2Array

class AudioData(Dataset):
    def __init__(self, df, window_size=12, stride_size=1):
        super(AudioData, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.time = self.preprocess(df)
    
    def preprocess(self, df):
        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time

        return df.values, start_idx[idx_mask], df.index[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1)

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

class GANFBaseModel(nn.Module):
    def __init__ (self, 
                  n_blocks = 6,
                  input_size = 1,
                  hidden_size = 32,
                  n_hidden = 1,
                  dropout = 0.1,
                  model="MAF",
                  batch_norm=False,
                  rho = 1.0,
                  rho_max = float(1e16),
                  max_iteraction = 1,
                  learning_rate = float(1e-3),
                  alpha = 0.0,
                  weight_decay = float(5e-4),
                  epochs = 5,
                  device = None
                  ):
        super().__init__()

        self.adjacent_matrix = None
        self.rho = rho         
        self.rho_max = rho_max     
        self.max_iteraction = max_iteraction    
        self.learning_rate = learning_rate          
        self.alpha = alpha       
        self.weight_decay= weight_decay
        self.epochs = epochs
        self.hidden_state = None

        if device != None: 
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device=self.device)

        #Reducing dimensionality 
        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True, dropout=dropout, device=self.device)

        # Graph Neural Networks model
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size) 
        self.gcn.to(self.device)

        #probability density estimation
        if model=="MAF":
            #Masked auto regressive flow
            self.nf = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh')
            self.nf.to(self.device)
        else:
            #real-valued non-volume preserving (real NVP)
            self.nf = RealNVP(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm)
            self.nf.to(self.device)

    @property
    def name(self):
        return "GANFBaseModel " + "+".join([f[0] for f in self.features])
    
    def create_dataLoader(self, X, batch_size = 128, window_size = 12):
        X = X.reshape((X.shape[0]*X.shape[2], X.shape[1]))
        X = torch.tensor(X)
        X = pd.DataFrame(X)
        X = X.iloc[:int(len(X))]
        X_dataLoader = DataLoader(AudioData(X, window_size=window_size), batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False) # Terei que fazer uma classe parecida com o traffic
        return X_dataLoader

        
    def fit(self, X, y=None, batch_size = 128):
        torch.cuda.empty_cache()
        gc.collect()
        X = self.create_dataLoader(X, batch_size)
        adjacent_matrix = self.__init_adjacent_matrix(X.dataset.data)
        dimension = X.dataset.data.shape[1]
        
        for _ in range(self.max_iteraction):
            learning_rate = self.learning_rate #* np.math.pow(0.1, epoch // 100)    
            optimizer = torch.optim.Adam([
                {'params': self.parameters(), 'weight_decay':self.weight_decay},
                {'params': [adjacent_matrix]}], lr=learning_rate, weight_decay=0.0)

            for i in range(self.epochs):
                loss_train = []
                self.train()
                print('epoch ' + str(i) + ' of ' + str(self.epochs))

                for x in X:
                    x = x.to(self.device)
                    optimizer.zero_grad()
                    A_hat = torch.divide(adjacent_matrix.T,adjacent_matrix.sum(dim=1).detach()).T
                    self.__treat_NaN(A_hat)
                    loss = -self.forward(x, A_hat)
                    h = torch.trace(torch.matrix_exp(A_hat*A_hat)) - dimension
                    total_loss = loss + 0.5 * self.rho * h * h + self.alpha * h
                    total_loss.backward()
                    clip_grad_value_(self.parameters(), 1)
                    optimizer.step()
                    loss_train.append(loss.item())
                    adjacent_matrix.data.copy_(torch.clamp(adjacent_matrix.data, min=0, max=1))
                    self.adjacent_matrix = adjacent_matrix

        self.adjacent_matrix = adjacent_matrix
        self.hidden_state = h
    
    def __treat_NaN(self, matrix):
        k = 0
        j = 0
        if matrix.isnan().any():
            for k in range(len(matrix)):
                for j in range(len(matrix[k])):
                    if matrix[k][j].isnan():
                        matrix[k][j] = 0

    def __init_adjacent_matrix(self, X):
        init = torch.zeros([X.shape[1], X.shape[1]])
        init = xavier_uniform_(init).abs()
        init = init.fill_diagonal_(0.0)
        return torch.tensor(init, requires_grad=True)
    
    def predict(self, X):
        return self.forward(X, self.adjacent_matrix)

    def score_samples(self, X):
        X_dataLoader = self.create_dataLoader(X, window_size=1)
        result = []
        for x in X_dataLoader:
            x = x.to(self.device)
            result.append(self.predict(X=x))
        return torch.tensor(result).mean()
    
    def forward(self, x, adjacent_matrix):
        return self.__test(x, adjacent_matrix).mean()

    def __test(self, x, adjacent_matrix):
        self.gcn.to(self.device)
        # x: N X K X L X D 
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        #x = torch.tensor(x.reshape(x.shape[0],1))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = h.to(self.device)
        adjacent_matrix = adjacent_matrix.to(self.device)
        h = self.gcn(h, adjacent_matrix)

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))
        x = x.reshape((-1,full_shape[3]))

        log_prob = self.nf.log_prob(x,h).reshape([full_shape[0],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)

        return log_prob
    
    def get_adjacent_matrix(self):
        return self.adjacent_matrix

FINAL_MODEL = GANFBaseModel()

class GANF(nn.Module, BaseEstimator, OutlierMixin):

    def __init__(self, 
                 final_model=FINAL_MODEL, 
                 sampling_rate=None,
                 random_state = None,
                 ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.final_model = final_model
        self.random_state = random_state
        self.model = self._build_model()

    @property
    def name(self):
        return "GANF " + "+".join([f[0] for f in self.features])
        
    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def transform(self, X, y=None):
        l = list()
        l.append(X)
        l.extend(self.model.steps[:-1])
        Xt = reduce(lambda x, y: y[1].transform(x), l)
        return Xt
    
    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        return np.array(
            list(
                map(
                    self.model.score_samples, 
                    [[x] for x in X])
                )
            )
    
    def get_adjacent_matrix(self):
        return self.final_model.get_adjacent_matrix()

    def _build_model(self):
        wav2array = Wav2Array(sampling_rate=self.sampling_rate, mono=False)
        
        model = Pipeline(
            steps=[
                ("wav2array", wav2array),
                ("final_model", self.final_model),
                ]
            )
        
        return model