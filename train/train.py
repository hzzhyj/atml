from loss import loss_beta_vae, loss_control_vae, loss_factor_vae, loss_discriminator
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np


def train_beta_vae(model, epochs, train_loader, optimizer, beta, distribution, device=None):
    model.train()

    train_loss = []
    for epoch in range(epochs):
        epoch_loss = []
        recon_losses = []
        for batch_idx, data in enumerate(train_loader):

            data = data.float()
            if device != None:
                data = data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            recon, mu, logvar = model(data)
            loss, recon_loss = loss_beta_vae(recon, data, mu, logvar, beta, distribution)
            loss.backward()
            optimizer.step()
            # print statistics
            epoch_loss.append(loss.item())
            recon_losses.append(recon_loss.item())
        epoch_loss = np.mean(epoch_loss)
        recon_losses = np.mean(recon_losses)
        train_loss.append((epoch_loss, recon_losses,))
        print("Epoch " + str(epoch) + " finished, loss: " + str((epoch_loss, recon_losses,)))
    return train_loss


def test_beta_vae(model, test_loader, beta, distribution, device=None):
    model.eval()

    test_loss = []
    recon_losses = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float()
            if device != None:
                data = data.to(device)
            # report the average loss over the test dataset
            recon, mu, logvar = model(data)
            loss, recon_loss = loss_beta_vae(recon, data, mu, logvar, beta, distribution)
            test_loss.append(loss.item())
            recon_losses.append(recon_loss.item())
    test_loss = np.mean(test_loss)
    recon_losses = np.mean(recon_losses)
    print("Test total loss: " + str(test_loss) + 'Test recon loss: ' + str(recon_losses))

def train_control_vae(model, epochs, train_loader, optimizer, distribution, device=None):
    model.train()

    train_loss = []
    recon_losses_list = []
    kl_divs_list = []
    iter_idx = 0
    for epoch in range(epochs):
        epoch_loss = []
        recon_losses = []
        kl_divs = []
        for batch_idx, data in enumerate(train_loader):
            iter_idx += 1
            data = data.float()
            if device != None:
                data = data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            recon, mu, logvar = model(data)
            recon_loss, kl_div = loss_control_vae(recon, data, mu, logvar, distribution)
            # get beta
            beta = model.update_beta(iter_idx, kl_div.clone().item())
            loss = recon_loss + beta * kl_div
            loss.backward()
            optimizer.step()
            # print statistics
            epoch_loss.append(loss.item())
            recon_losses.append(recon_loss.item())
            kl_divs.append(kl_div.item())
        epoch_loss = np.mean(epoch_loss)
        recon_loss = np.mean(recon_losses)
        kl_divs = np.mean(kl_divs)
        train_loss.append(epoch_loss)
        recon_losses_list.append(recon_loss)
        kl_divs_list.append(kl_divs)
        print("Epoch " + str(epoch) + " finished, loss: " + str(epoch_loss) + ", recon loss: " + str(recon_loss) + ", kl div: " + str(kl_divs))
    return train_loss, recon_losses_list, kl_divs_list


def test_control_vae(model, test_loader, distribution, device=None):
    model.eval()

    test_recon_loss = []
    test_kl_div = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float()
            if device != None:
                data = data.to(device)
            # report the average loss over the test dataset
            recon, mu, logvar = model(data)
            recon_loss, kl_div = loss_control_vae(recon, data, mu, logvar, distribution)
            test_recon_loss.append(recon_loss.item())
            test_kl_div.append(test_kl_div.item())
    test_recon_loss = np.mean(test_recon_loss)
    test_kl_div = np.mean(test_kl_div)
    print("Test recon loss: " + str(test_loss) + ", kl div: " + str(test_kl_div))


def random_permute(v):
    batch_size, latent_dim = v.size()
    new_v = torch.zeros_like(v)

    for j in range(latent_dim):
        permutation = torch.randperm(batch_size)
        for i in range(batch_size):
            new_v[i,j] = v[permutation[i],j]

    return new_v

def train_factor_vae(model, discriminator, epochs, train_loader, vae_optimizer, discriminator_optimizer, gamma, distribution, device=None):
    model.train()
    train_losses_list = []
    recon_losses_list = []
    kl_divs_list = []
    tc_losses_list = []
    discriminator_losses_list = []
    for epoch in range(epochs):
        epoch_losses = []
        recon_losses = []
        kl_divs = []
        tc_losses = []
        discriminator_losses = []
        for data1, data2 in train_loader:

            data1 = data1.float()
            data2 = data2.float()
            if device != None:
                data1 = data1.to(device)
                data2 = data2.to(device)

            x_recon1, mu1, logvar1, z1 = model(data1)
            if device != None:
                z1 = z1.to(device)
            dz = discriminator(z1)

            recon_loss, kl_div, tc_loss  = loss_factor_vae(x_recon1, data1, mu1, logvar1, dz, distribution)
            recon_losses.append(recon_loss.item())
            kl_divs.append(kl_div.item())
            tc_losses.append(tc_loss.item())
            epoch_loss = recon_loss + kl_div + torch.mul(gamma, tc_loss)
            epoch_losses.append(epoch_loss.item())

            vae_optimizer.zero_grad()
            epoch_loss.backward(retain_graph=True)
            vae_optimizer.step()

            with torch.no_grad():
                x_recon2, mu2, logvar2, z2 = model(data2)
                newz = random_permute(z2)

            if device != None:
                newz = newz.to(device)

            dnewz = discriminator(newz)
            dz = discriminator(z1.detach())
            
            ones = torch.ones(dz.size(0), dtype=torch.long)
            zeros = torch.zeros(dz.size(0), dtype=torch.long)

            if device != None :
                ones = ones.to(device)
                zeros = zeros.to(device)

            d_loss = 0.5*(loss_discriminator(dnewz,ones) + loss_discriminator(dz,zeros))
            discriminator_losses.append(d_loss.item())

            discriminator_optimizer.zero_grad()
            d_loss.backward()
            discriminator_optimizer.step()

        epoch_loss = np.mean(epoch_losses)
        recon_loss = np.mean(recon_losses)
        kl_div = np.mean(kl_divs)
        tc_loss = np.mean(tc_losses)
        discriminator_loss = np.mean(discriminator_losses)

        train_losses_list.append(epoch_loss)
        recon_losses_list.append(recon_loss)
        kl_divs_list.append(kl_div)
        tc_losses_list.append(tc_loss)
        discriminator_losses_list.append(discriminator_loss)


        print("Epoch " + str(epoch) + " finished, loss: " + str(epoch_loss) + ", recon loss: " + str(recon_loss) + ", kl div: " + str(kl_div)+ ", TC loss: "+str(tc_loss)+", discriminator loss: "+str(discriminator_loss))
    return train_losses_list, recon_losses_list, kl_divs_list, tc_losses_list, discriminator_losses_list

def test_factor_vae(model, discriminator, test_loader, gamma, distribution, device=None):
    model.eval()

    test_losses = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.float()
            if device != None:
                data = data.to(device)
            # report the average loss over the test dataset
            recon, mu, logvar, z = model(data)
            dz = discriminator(z)
            recon_loss, kl_div, tc_loss = loss_factor_vae(recon, data, mu, logvar, dz, distribution)
            loss =recon_loss + kl_div + torch.mul(gamma, tc_loss)
            test_losses.append(loss.item())
    test_loss = np.mean(test_losses)
    print("Test loss: " + str(test_loss))
    
    
def train_classifier_metric(model, epochs, train_loader, optimizer, device = None):
    losses=[]
    accuracies = []
    model.train()
    loss_nll = nn.NLLLoss()
    for epoch in range(epochs):
        
        epoch_losses = []
        total_correct = 0
        for z_diff, factor in train_loader:
            
            if device!= None:
                z_diff = z_diff.to(device)
                factor = factor.to(device)
            pred = model(z_diff)
            loss = loss_nll(pred, factor)
            epoch_losses.append(loss.item())
            total_correct+=pred.argmax(dim=1).eq(factor).sum().item()
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)
        accuracies.append(total_correct/len(train_loader.dataset))
    return losses, accuracies

def test_classifier_metric(model, test_loader, device = None):
    model.eval()
    loss_nll = nn.NLLLoss()
    losses= []
    accuracy = 0
    with torch.no_grad():
        for z_diff, factor in test_loader:
            if device != None:
                z_diff = z_diff.to(device)
                factor = factor.to(device)
            pred = model(z_diff)
            loss = loss_nll(pred, factor)
            accuracy+=pred.argmax(dim=1).eq(factor).sum().item()

            # report the average loss over the test dataset
            losses.append(loss.item())
    test_loss = np.mean(losses)
    #print("Test loss: " + str(test_loss)+ ", test accuracy: "+str(accuracy/len(test_loader.dataset)))
    return accuracy/len(test_loader.dataset)
    