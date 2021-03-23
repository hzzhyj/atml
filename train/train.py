from loss import loss_beta_vae
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
