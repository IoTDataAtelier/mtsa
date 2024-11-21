import torch.nn as nn
import torch
import random
import numpy as np
import pandas as pd
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from mtsa.models.GANF_components.NF import MAF, RealNVP
from mtsa.models.GANF_components.ganfLayoutData import GANFData
from mtsa.models.GANF_components.gnn import GNN

class GANFBaseModel(nn.Module):
    def __init__ (self, 
                  n_blocks = 6,
                  input_size = 1,
                  hidden_size = 32,
                  n_hidden = 1,
                  dropout = 0.0,
                  model="MAF",
                  batch_norm=False,
                  rho_max = float(1e16),
                  max_iteraction = 2,
                  learning_rate = float(1e-3),
                  alpha = 0.0,
                  weight_decay = float(5e-4),
                  epochs = 20,
                  device = None,
                  index_CUDA_device = '0'
                  ):
        super().__init__()
        self.min = 0
        self.max = 0 
        self.adjacent_matrix = None
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
            self.device = torch.device(f'cuda:{str(index_CUDA_device)}' if torch.cuda.is_available() else "cpu")

        #Reducing dimensionality 
        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True, dropout=dropout, device=self.device)

        # Graph Neural Networks model
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size) 

        #probability density estimation
        if model=='MAF':
            #Masked auto regressive flow
            self.nf = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh')
        else:
            #real-valued non-volume preserving (real NVP)
            self.nf = RealNVP(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm)

        self.to(device=self.device)

    @property
    def name(self):
        return "GANFBaseModel " + "+".join([f[0] for f in self.features])
      
    def get_adjacent_matrix(self):
        return self.adjacent_matrix

    def fit(self, X, y=None, batch_size = None, epochs= None, max_iteraction= None, learning_rate = None, debug_dataframe= None, isWaveData = False, mono = None):
        torch.cuda.manual_seed(10)
        random.seed(10)
        np.random.seed(10)
        torch.manual_seed(10)
        torch.autograd.set_detect_anomaly(True)

        self.isMonoData = mono
        loss_best = 100
        self.isWaveData = isWaveData
        h_A_old = np.inf
        h_tol = float(1e-6)
        rho = 1.0

        if epochs is not None:
           self.epochs = epochs
        if max_iteraction is not None:
            self.max_iteraction =  max_iteraction
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if batch_size is None:
            batch_size = 32

        dimension = X.data.shape[1]
        self.adjacent_matrix = self.__init_adjacent_matrix(X.data)

        dataloaders = self.__create_dataLoader(torch.tensor(X), batch_size=batch_size, isMonoData = mono)
        adjacent_matrix = self.adjacent_matrix

        self.__fitCore(loss_best, h_A_old, h_tol, dimension, dataloaders, adjacent_matrix, rho)

    def predict(self, X):
        return self.__forward(X, self.adjacent_matrix)

    def score_samples(self, X):
        if self.isWaveData:
            return self.__score_wave_data(X=X)
        return self.__score_discrete_data(X=X)
    
    def __forward(self, x, A):
        # x: N X K X L X D 
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))
        x = x.reshape((-1,full_shape[3]))

        log_prob = self.nf.log_prob(x,h).reshape([full_shape[0],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)
        log_prob_x = log_prob.mean()
        return log_prob_x

    def __fitCore(self, loss_best, h_A_old, h_tol, dimension, dataloaders, adjacent_matrix, rho):
        previous_total_loss = None
        for j in range(self.max_iteraction):
            print('iteraction ' + str(j+1) + ' of ' + str(self.max_iteraction))

            while rho < self.rho_max:
                learning_rate = self.learning_rate #np.math.pow(0.1, epoch // 100)    
                optimizer = torch.optim.Adam([
                    {'params': self.parameters(), 'weight_decay':self.weight_decay},
                    {'params': [adjacent_matrix]}], lr=learning_rate, weight_decay=0.0)

                for epoch in range(self.epochs):
                    loss_train = []
                    epoch += 1
                    self.train()

                    for dataloader in dataloaders:
                        for x in dataloader:
                            x = x.to(self.device)
                            
                            optimizer.zero_grad()
                            loss = -self.__forward(x, adjacent_matrix)
                            h = torch.trace(torch.matrix_exp(adjacent_matrix*adjacent_matrix)) - dimension
                            total_loss = loss + 0.5 * rho * h * h + self.alpha * h

                            h.to(self.device)
                            
                            total_loss.backward()
                            clip_grad_value_(self.parameters(), 1)
                            optimizer.step()
                            loss_train.append(loss.item())
                            adjacent_matrix.data.copy_(torch.clamp(adjacent_matrix.data, min=0, max=1))
                            
                            if np.mean(loss_train) < loss_best:
                                loss_best = np.mean(loss_train)
                                self.adjacent_matrix = adjacent_matrix

                            previous_total_loss = total_loss

                    print('Epoch: {}, train -log_prob: {:.2f}, h: {}'.format(epoch, np.mean(loss_train), h.item()))
                    
                del optimizer
                torch.cuda.empty_cache()

                if h.item() > 0.5 * h_A_old:
                    rho *= 10
                else:
                    break

            h_A_old = h.item()
            self.alpha += rho*h.item()

            if h_A_old <= h_tol or rho >= self.rho_max:
                break
    
    def __score_discrete_data(self, X):
        X = np.array(X)
        X = X.reshape(X.shape[0], X.shape[1], 1, 1)
        X = torch.tensor(X).float()
        X = X.to(self.device)
        result = self.predict(X=X)
        return float(result)
    
    def __score_wave_data(self, X):
        dataloaders = self.__create_dataLoader(X, window_size=1, isMonoData = self.isMonoData)
        result = []
        for dataloader in dataloaders:
            for x in dataloader:
                x = x.to(self.device)
                result.append(self.predict(X=x))
        return torch.tensor(result).mean()
       
    def __create_dataLoader(self, X, batch_size = 32, window_size = 12, isMonoData = True):
        X = self.__create_dataframe(X, isMonoData)
        dataloaders = []
        for x in X:
            x_dataloader = DataLoader(GANFData(x, window_size=window_size), batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False)
            dataloaders.append(x_dataloader)
        return dataloaders
    
    def __create_dataframe(self, X, isMonoData):
        if self.isWaveData and not isMonoData:
            dataframes = []
            for x in X:
                x = x.T
                x_df = self.__array2Dataframe(x) 
                dataframes.append(x_df)
            return dataframes
        
        if self.isWaveData and isMonoData:
            X = X.T

        X_df =  self.__array2Dataframe(X)

        return [X_df]

    def __array2Dataframe(self, x):
        x_df = pd.DataFrame(x)
        x_df= x_df.reset_index()
        x_df = x_df.rename(columns={"index":"channel"})
        x_df["channel"] = pd.to_datetime(x_df["channel"], unit="s")
        x_df = x_df.set_index("channel")
        x_df = self.__normalize(x_df)
        return x_df

    def __normalize(self, X):
        mean = X.values.mean()
        std = X.values.std()
        X = (X - mean)/std
        return X

    def __init_adjacent_matrix(self, X):
        torch.cuda.manual_seed(10)
        random.seed(10)
        np.random.seed(10)
        torch.manual_seed(10)
        init = torch.zeros([X.shape[1], X.shape[1]])
        init = xavier_uniform_(init).abs()
        init = init.fill_diagonal_(0.0)
        return torch.tensor(init, requires_grad=True, device=self.device)
