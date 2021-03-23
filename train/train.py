import torch
import torch.nn.functional as F
from torch import optim


def train_beta_vae(model, epochs, train_loader, loss_function, optimizer, beta, distribution):
    model.train()

    train_loss = []
    for epoch in range(epochs):
        epoch_loss = []
        for batch_idx, (data, _) in enumerate(train_loader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar, beta, distribution)
            loss.backward()
            optimizer.step()
            # print statistics
            epoch_loss.append(loss.item())
        epoch_loss = torch.mean(epoch_loss)
        train_loss.append(epoch_loss)
        print("Epoch " + str(epoch) + " finished, loss: " + str(epoch_loss))
    return train_loss
