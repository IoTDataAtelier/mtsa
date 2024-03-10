
from sklearn.base import BaseEstimator, OutlierMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import reduce
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.pipeline import (Pipeline) 
from mtsa.features.mel import (Array2Mfcc)
from mtsa.models.GANF_model.components.NF import MAF, RealNVP
from mtsa.utils import (Wav2Array,)
from functools import reduce
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_value_

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


class GANF(nn.Module, BaseEstimator, OutlierMixin):
    def __init__ (self, 
                  n_blocks,
                  input_size,
                  hidden_size,
                  n_hidden,
                  dropout = 0.1,
                  model="MAF",
                  batch_norm=True,
                  rho = 1.0,
                  rho_max = float(1e16),
                  max_iteraction = 20,
                  learning_rate = float(1e-3),
                  alpha = 0.0,
                  weight_decay = float(5e-4)
                  ):
        super(GANF, self).__init__()

        self.adjacent_matrix = torch.tensor()
        self.rho = rho         
        self.rho_max = rho_max     
        self.max_iteraction = max_iteraction    
        self.learning_rate = learning_rate          
        self.alpha = alpha       
        self.weight_decay= weight_decay
        
        #Reducing dimensionality 
        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True, dropout=dropout)

        # Graph Neural Networks model
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size) 

        #probability density estimation
        if model=="MAF":
            #Masked auto regressive flow
            self.nf = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh')
        else:
            #real-valued non-volume preserving (real NVP)
            self.nf = RealNVP(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm)

    @property
    def name(self):
        return "GANF " + "+".join([f[0] for f in self.features])
        
    def fit(self, X, y=None): #OK
        for _ in range(self.max_iteraction):

            while self.rho < self.rho_max:
                learning_rate = self.learning_rate #* np.math.pow(0.1, epoch // 100)
                optimizer = torch.optim.Adam([
                    {'params': self.parameters(), 'weight_decay':self.weight_decay},
                    {'params': [adjacent_matrix]}], learning_rate=learning_rate, weight_decay=0.0)

            for _ in range(self.epochs):
                init = torch.zeros([X.shape[0], X.shape[0]])
                init = xavier_uniform_(init).abs()
                init = init.fill_diagonal_(0.0)
                adjacent_matrix = torch.tensor(init, requires_grad=True)

                loss_train = []
                epoch += 1
                self.train()

                for x in X:
                    optimizer.zero_grad()
                    A_hat = torch.divide(adjacent_matrix.T,adjacent_matrix.sum(dim=1).detach()).T
                    loss = -self.forward(x, A_hat)
                    h = torch.trace(torch.matrix_exp(A_hat*A_hat)) - X.shape[0]
                    total_loss = loss + 0.5 * self.rho * h * h + self.alpha * h

                    total_loss.backward()
                    clip_grad_value_(self.parameters(), 1)
                    optimizer.step()
                    loss_train.append(loss.item())
                    self.adjacent_matrix.data.copy_(torch.clamp(adjacent_matrix.data, min=0, max=1))
        return

    def transform(self, X, y=None):
        l = list()
        l.append(X)
        l.extend(self.model.steps[:-1])
        Xt = reduce(lambda x, y: y[1].transform(x), l)
        return Xt
    
    def predict(self, X):
        return self.__forward(X, self.adjacent_matrix)

    def score_samples(self, X):
        return self.model.score_samples(X=X)

    def _build_model(self):
        wav2array = Wav2Array(sampling_rate=self.sampling_rate)
        array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)
        
        model = Pipeline(
            steps=[
                ("wav2array", wav2array),
                ("array2mfcc", array2mfcc),
                ("final_model", self),
                ]
            )
        
        return model
    
    def __forward(self, x, adjacent_matrix):

        return self.__test(x, adjacent_matrix).mean()

    def __test(self, x, adjacent_matrix):
        # x: N X K X L X D 
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))


        h = self.gcn(h, adjacent_matrix)

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))
        x = x.reshape((-1,full_shape[3]))

        log_prob = self.nf.log_prob(x,h).reshape([full_shape[0],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)

        return log_prob


