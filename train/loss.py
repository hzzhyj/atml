import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_reconstruction_loss(recon, x, distribution, batch_size):
    reconstruction_loss = 0
    if distribution == 'gaussian':
        recon = F.sigmoid(recon)
        reconstruction_loss = F.mse_loss(recon, x, size_average=False).div(batch_size)
    elif distribution == 'bernoulli':
        reconstruction_loss = F.binary_cross_entropy_with_logits(recon, x, size_average=False).div(batch_size)
    else:
        raise Exception('distribution must be either gaussian or bernoulli')
    return reconstruction_loss

def compute_kl_div(mu, logvar):
    mu_sq = mu.pow(2)
    return -0.5 * torch.mean(1 + logvar - logvar.exp() - mu_sq)

def compute_tc_loss(dz):
    loss = (dz[:, :1] - dz[:, 1:]).mean()
    return loss

def loss_discriminator(dz, permuted):
    if permuted:
        return F.cross_entropy(dz, torch.ones(dz.size(0), dtype=torch.long), reduction='mean')
    else:
        return F.cross_entropy(dz, torch.zeros(dz.size(0), dtype=torch.long), reduction='mean') 



def loss_beta_vae(recon, x, mu, logvar, beta, distribution='gaussian'):

    # The reconstruction term can be approximated with a single sample from the approximate posterior.
    # Use the analytical exppression for the KL term.
    # recon, mu and logvar are the outputs from model.forward(x).

    # make sure size matches since x can still be in a grid
    x = x.view(recon.size())
    batch_size = x.size(0)

    # Reconstruction term
    reconstruction_loss = compute_reconstruction_loss(recon, x, distribution, batch_size)

    # KL term loss
    neg_kl_loss = compute_kl_div(mu, logvar)
    return reconstruction_loss + beta * neg_kl_loss, reconstruction_loss

def loss_control_vae(recon, x, mu, logvar, beta, distribution='gaussian'):
    x = x.view(recon.size())
    batch_size = x.size(0)

    # Reconstruction term
    reconstruction_loss = compute_reconstruction_loss(recon, x, distribution, batch_size)

    # KL term loss
    neg_kl_loss = compute_kl_div(mu, logvar)
    return reconstruction_loss, neg_kl_loss

def loss_factor_vae(recon, x, mu, logvar, dz, distribution = 'gaussian'):
    x = x.view(recon.size())
    batch_size = x.size(0)

    # Reconstruction term
    reconstruction_loss = compute_reconstruction_loss(recon, x, distribution, batch_size)

    # KL term loss
    neg_kl_loss = compute_kl_div(mu, logvar)
    tc_loss = compute_tc_loss(dz)
    return reconstruction_loss, neg_kl_loss, tc_loss
