import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import loss_beta_vae


def test(test_loader, loss_function, beta, distribution):
    model.eval()

    test_loss = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):

            # TODO: write some code here
            # report the average loss over the test dataset
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar, beta, distribution)
            test_loss.append(loss.item())
    test_loss = torch.mean(test_loss)
    print("Test loss: " + str(test_loss))
