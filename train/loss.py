import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_beta_vae(recon, x, mu, logvar, beta, distribution='gaussian'):

    # The reconstruction term can be approximated with a single sample from the approximate posterior.
    # Use the analytical exppression for the KL term.
    # recon, mu and logvar are the outputs from model.forward(x).

    # make sure size matches since x can still be in a grid
    x = x.view(recon.size())

    reconstruction_loss = 0
    if distribution == 'gaussian':
        recon = F.sigmoid(recon)
        reconstruction_loss = F.mse_loss(recon, x)
    else if distribution == 'bernoulli':
        reconstruction_loss = F.binary_cross_entropy_with_logits(recon, x)
    else:
        raise Exception('distribution must be either gaussian or bernoulli')

    # KL term loss
    mu_sq = mu.pow(2)
    neg_kl_loss = -0.5 * torch.mean(1 + logvar - logvar.exp() - mu_sq)
    return reconstruction_loss + beta * neg_kl_loss
