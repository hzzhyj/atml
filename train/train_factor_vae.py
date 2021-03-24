import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def recon_loss(x_recon, x):
    loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction = "sum").div(x.size(0))
    return loss

def kld_loss(mu, logvar):
    loss = torch.mul(0.5,(-1 + torch.pow(mu,2) - logvar + torch.exp(logvar)).sum(1).mean())
    return loss
    
def tc_loss(dz):
    loss = (dz[:, :1] - dz[:, 1:]).mean()
    return loss

def discriminator_loss(dz, permuted):
    if permuted:
        return F.cross_entropy(dz, torch.ones(dz.size(0), dtype=torch.long), reduction='mean')
    else:
        return F.cross_entropy(dz, torch.zeros(dz.size(0), dtype=torch.long), reduction='mean') 

def random_permute(v):
    batch_size, latent_dim = v.size()
    new_v = torch.zeros_like(v)
    
    for j in range(latent_dim):
        permutation = torch.randperm(batch_size)
        for i in range(batch_size):
            new_v[i,j] = v[permutation[i],j]

    return new_v


def train_factor_VAE(model, discriminator, vae_optimizer, discriminator_optimizer, gamma, nb_epoch, train_loader, viz = False):
    model.train()
    vae_loss = []
    for epoch in range(nb_epoch):
        count=1
        train_vae_loss = 0
        train_recon_loss = 0
        train_kld_loss = 0
        train_tc_loss = 0
        train_d_loss = 0
        for data1, data2 in train_loader:
            batch_size = data1.size(0)
            x_recon1, mu1, logvar1, z1 = model(data1)

            recon_l = recon_loss(x_recon1, data1)
            train_recon_loss+= recon_l* batch_size

            kld_l = kld_loss(mu1, logvar1)
            train_kld_loss+= kld_l* batch_size

            dz = discriminator(z1)
            tc_l = tc_loss(dz)
            train_tc_loss+= tc_l* batch_size

            total_loss = recon_l + kld_l + torch.mul(gamma, tc_l)
            train_vae_loss += total_loss * batch_size

            vae_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            vae_optimizer.step()

            with torch.no_grad():
                x_recon2, mu2, logvar2, z2 = model(data2)
                newz = random_permute(z2)

            dnewz = discriminator(newz)
            dz = discriminator(z1.detach())
            discriminator_l = 0.5*(discriminator_loss(dnewz,True) + discriminator_loss(dz,False))
            train_d_loss += discriminator_l * batch_size

            discriminator_optimizer.zero_grad()
            discriminator_l.backward()
            discriminator_optimizer.step()

            count+=1
 
        vae_loss.append(train_vae_loss.item()/len(train_loader.dataset))
   
        if viz:
            print("Epoch "+str(epoch+1)+":")
            print("training vae loss of "+ str(train_vae_loss.item()/len(train_loader.dataset)))
            print("training recon loss of "+ str(train_recon_loss.item()/len(train_loader.dataset)))
            print("training kld loss of "+ str(train_kld_loss.item()/len(train_loader.dataset)))
            print("training TC loss of "+ str(train_tc_loss.item()/len(train_loader.dataset)))
            print("training discriminator loss of "+ str(train_d_loss.item()/len(train_loader.dataset)))
    if viz:
        plt.plot(np.arange(1,nb_epoch+1,1), vae_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train VAE loss evolution over epochs")
        plt.show()
    return 
        