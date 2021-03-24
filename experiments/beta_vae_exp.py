import numpy as np
import torch
import os
import sys
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path+"/models")
sys.path.append(module_path+"/train")
sys.path.append(module_path+"/datasets")

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using device: ' + str(device))

from datasets import train_test_random_split, load_dsprites
from train import train_beta_vae, test_beta_vae
from loss import loss_beta_vae
from beta_vae import BetaVAEDSprites

# load the dataset
dataset = load_dsprites("../datasets/dsprites.npz")
dataset = torch.from_numpy(dataset)

#n_imgs = 50000
#indices = torch.randperm(dataset.size(0))[:n_imgs]
#dataset = dataset[indices]

data_train, data_test = train_test_random_split(dataset, 0.8)

batch_size = 64
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)

# define the model
model = BetaVAEDSprites()
model.to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2)

# train the model
epochs = 10
beta = 4
distribution = 'bernoulli'
print('Training started')
train_beta_vae(model, epochs, train_loader, optimizer, beta, distribution, device)
print('Training finished')
torch.save(model, 'betavae_beta4_e10_alldata.dat')


train_beta_vae(model, epochs, train_loader, optimizer, beta, distribution, device)
torch.save(model, 'betavae_beta4_e20_alldata.dat')

train_beta_vae(model, epochs, train_loader, optimizer, beta, distribution, device)
torch.save(model, 'betavae_beta4_e30_alldata.dat')

train_beta_vae(model, epochs, train_loader, optimizer, beta, distribution, device)
torch.save(model, 'betavae_beta4_e40_alldata.dat')

test_beta_vae(model, test_loader, beta, distribution, device)