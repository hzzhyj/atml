import torch
import torch.nn as nn
import torch.nn.functional as F
from atml.models.beta_vae import BetaVAEDSprites

class FactorVAEDSprites(BetaVAEDSprites):
    def __init__(self,n_latents=10):
        super(FactorVAEDSprites, self).__init__()
        #self.n_latents = n_latents
        #self.encode_conv = nn.Sequential(
        #    nn.Conv2d(1, 32, 4, 2, 1),
        #    nn.ReLU(),
        #    nn.Conv2d(32, 32, 4, 2, 1),
        #    nn.ReLU(),
        #    nn.Conv2d(32, 64, 4, 2, 1),
        #    nn.ReLU(),
        #    nn.Conv2d(64, 64, 4, 2, 1),
        #    nn.ReLU()
        #)
        #self.encode_fc1 = nn.Linear(1024, 128)
        #self.encode_fc2 = nn.Linear(128,2*n_latents)
        #
        #self.decode_fc1 = nn.Linear(10,128)
        #self.relu1 = nn.ReLU()
        #self.decode_fc2 = nn.Linear(128,1024)
        #self.relu2 = nn.ReLU()
        #
        #self.decode_conv = nn.Sequential(
        #    nn.ConvTranspose2d(64, 64, 4, 2, 1),
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(64, 32, 4, 2, 1),
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(32, 32, 4, 2, 1),
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(32, 1, 4, 2, 1)
        #)
        
        
    #def encode(self, x):
    #    x = self.encode_conv(x)
    #    x = x.view(x.size(0),-1)
    #    x = self.encode_fc1(x)
    #    x = self.encode_fc2(x)
    #    return x
    #
    #def decode(self, x):
    #    x = self.decode_fc1(x)
    #    x = self.relu1(x)
    #    x = self.decode_fc2(x)
    #    x = self.relu2(x)
    #    x = x.view(x.size(0),64,4,4)
    #    x = self.decode_conv(x)
    #    return x

    #def reparameterize(self, mu, logvar):
    #    std = torch.exp(torch.mul(0.5,logvar))
    #    epsilon =  torch.randn_like(std)
    #    return mu + torch.mul(epsilon,std)     

    def forward(self, x):
        x = x.view(-1, 64*64)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z).view(-1,64,64)
        return x_recon, mu, logvar, z.view(-1,self.n_latents)
    
    
class Discriminator(nn.Module):
    def __init__(self, n_latents=10, slope=0.01, nb_layers=6, hidden_dim=1000):
        super(Discriminator, self).__init__()
        self.n_latents = n_latents
        self.nb_layers = nb_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_latents, hidden_dim))
        self.layers.append(nn.LeakyReLU(slope))
        for i in range(1,nb_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU(slope))
        self.layers.append(nn.Linear(hidden_dim, 2))

    def forward(self, z):
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return z

