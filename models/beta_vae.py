import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAEDSprites(nn.Module):
    def __init__(self, n_latents=10):
        super(BetaVAEDSprites, self).__init__()

        self.n_latents = n_latents
        self.encoder = nn.Sequential(
            nn.Linear(64*64, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Linear(1200, n_latents * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_latents, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def encode(self, x):
        # estimates mu, log-variance of approximate posterior q(z|x)
        x = x.view(-1, 64*64)
        parameters = self.encoder(x)
        mu = parameters[:, :self.n_latents]
        logvar = parameters[:, self.n_latents:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # reparameterize so the loss can propagate back to mu and logvar, parameters
        # for the posterior latent distribution
        # std = exp(1/2 * log(std^2))
        std = logvar.div(2).exp()
        samples = torch.normal(torch.zeros_like(mu), torch.ones_like(logvar))
        return samples * std + mu

    def decode(self, z):
        recon = self.decoder(z)
        # do not apply sigmoid since BCEWithLogits is more efficient (applies sigmoid)
        # and binary cross entropy in one step to avoid instability
        return recon  # estimates mu of likelihood p(x|z)

    def forward(self, x):

        # 1) mu,log-variance of the approximate posterior given x
        # 2) mu of the likelihood given a random sample from the posterior given x
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # should return  (likelihood mu, posterior mu, posterior log-variance)
        return recon, mu, logvar
    
    def get_latent_representation(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return z.view(-1, self.n_latents)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BetaVAECelebA(nn.Module):
    """
    Beta VAE model with convolutional layers
    Input image should be downsized to 64x64
    """

    def __init__(self, n_latents=32, n_channels=3):
        super(BetaVAECelebA, self).__init__()
        
        self.n_channels = n_channels
        self.n_latents = n_latents

        self.encoder = nn.Sequential(
            # after this convolution, we have a tensor of size [batch_size, 32, 31, 31]     
            nn.Conv2d(n_channels, 32, 4, 2), 
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2), # [batch_size, 32, 14, 14]      
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), #[batch_size, 64, 6, 6]         
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2), #[batch_size, 64, 2, 2]        
            nn.ReLU(),
            View((-1, 256)),                 
            nn.Linear(256, n_latents*2),            
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latents, 256),               
            View((-1, 64, 2, 2)),                    
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, output_padding=1), # padding required to compensate
            nn.ReLU(True),
            nn.ConvTranspose2d(32, n_channels, 4, 2),  
        )

    def encode(self, x):
        # estimates mu, log-variance of approximate posterior q(z|x)
        x = x.view(-1,self.n_channels,64,64)
        parameters = self.encoder(x)
        mu = parameters[:, :self.n_latents]
        logvar = parameters[:, self.n_latents:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # reparameterize so the loss can propagate back to mu and logvar, parameters
        # for the posterior latent distribution
        # std = exp(1/2 * log(std^2))
        std = logvar.div(2).exp()
        samples = torch.normal(torch.zeros_like(mu), torch.ones_like(logvar))
        return samples * std + mu

    def decode(self, z):
        recon = self.decoder(z)
        # do not apply sigmoid since BCEWithLogits is more efficient (applies sigmoid)
        # and binary cross entropy in one step to avoid instability
        return recon  # estimates mu of likelihood p(x|z)

    def forward(self, x):

        # 1) mu,log-variance of the approximate posterior given x
        # 2) mu of the likelihood given a random sample from the posterior given x
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # should return  (likelihood mu, posterior mu, posterior log-variance)
        return recon, mu, logvar

class Classifier(nn.Module):
    def __init__(self, n_latents=10, n_factors=5):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_latents,n_factors)         
            
    def forward(self, x):
        x = self.fc(x)
        x = F.log_softmax(x,dim=1)
        return x
    
    def reset_parameters(self):
        self.fc.reset_parameters()
             