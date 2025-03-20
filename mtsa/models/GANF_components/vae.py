import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        # encode
        self.fc1 = nn.Linear(33, 400)  # feature_size + class_size = 1 + 32
        self.fc21 = nn.Linear(400, 64)  # Dimensão reduzida para z_mu
        self.fc22 = nn.Linear(400, 64)  # Dimensão reduzida para z_var

        # decode
        self.fc3 = nn.Linear(64 + 32, 400)  # z (64) + c (32)
        self.fc4 = nn.Linear(400, 1)  # Reconstruir x com dimensão 1

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):  # Q(z|x, c)
        '''
        x: (bs, 1)  # feature_size
        c: (bs, 32)  # class_size
        '''
        inputs = torch.cat([x, c], dim=1)  # (bs, feature_size + class_size = 33)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):  # P(x|z, c)
        '''
        z: (bs, 64)  # latent_size
        c: (bs, 32)  # class_size
        '''
        inputs = torch.cat([z, c], dim=1)  # (bs, latent_size + class_size = 96)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
