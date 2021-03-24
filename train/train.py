from loss import loss_beta_vae, loss_control_vae
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
        for batch_idx, data in enumerate(train_loader):

            data = data.float()
            if device != None:
                data = data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            recon, mu, logvar = model(data)
            loss = loss_beta_vae(recon, data, mu, logvar, beta, distribution)
            loss.backward()
            optimizer.step()
            # print statistics
            epoch_loss.append(loss.item())
        epoch_loss = np.mean(epoch_loss)
        train_loss.append(epoch_loss)
        print("Epoch " + str(epoch) + " finished, loss: " + str(epoch_loss))
    return train_loss


def test_beta_vae(model, test_loader, beta, distribution, device=None):
    model.eval()

    test_loss = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float()
            if device != None:
                data = data.to(device)
            # report the average loss over the test dataset
            recon, mu, logvar = model(data)
            loss = loss_beta_vae(recon, data, mu, logvar, beta, distribution)
            test_loss.append(loss.item())
    test_loss = np.mean(test_loss)
    print("Test loss: " + str(test_loss))

def train_control_vae(model, epochs, train_loader, optimizer, distribution, device=None):
    model.train()

    train_loss = []
    recon_losses_list = []
    kl_divs_list = []
    for epoch in range(epochs):
        epoch_loss = []
        recon_losses = []
        kl_divs = []
        for batch_idx, data in enumerate(train_loader):

            data = data.float()
            if device != None:
                data = data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            recon, mu, logvar = model(data)
            recon_loss, kl_div = loss_control_vae(recon, data, mu, logvar, distribution)
            # get beta
            beta = model.update_beta(epoch + 1, kl_div.clone().item())
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
    return train_loss, recon_loss_list, kl_div_list


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
