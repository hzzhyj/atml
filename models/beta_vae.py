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
        x = x.view(-1, 64*64)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # should return  (likelihood mu, posterior mu, posterior log-variance)
        return recon, mu, logvar

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fclayers = nn.Sequential(
             nn.Linear(10,100),
             nn.ReLU(),
             nn.Linear(100,50),
             nn.ReLU(),
             nn.Linear(50,20),
             nn.ReLU(),
             nn.Linear(20,5)
         )
            
    def forward(self, x):
        x = self.fclayers(x)
        x = F.log_softmax(x,dim=1)
        return x
             