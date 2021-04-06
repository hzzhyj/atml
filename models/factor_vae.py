import torch
import torch.nn as nn
import torch.nn.functional as F
from beta_vae import BetaVAEDSprites, BetaVAECelebA

class FactorVAEDSprites(BetaVAEDSprites):
    def __init__(self,n_latents=10):
        super(FactorVAEDSprites, self).__init__(n_latents)    

    def forward(self, x):
        x = x.view(-1, 64*64)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z).view(-1,64,64)
        return recon, mu, logvar, z.view(-1,self.n_latents)
    
    
class FactorVAEcnn(BetaVAECelebA):
    def __init__(self,n_latents=10, n_channels=1):
        super(FactorVAEcnn, self).__init__(n_latents, n_channels)  
        
    def forward(self, x):   
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # should return  (likelihood mu, posterior mu, posterior log-variance)
        return recon, mu, logvar, z.view(-1,self.n_latents)



    
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

