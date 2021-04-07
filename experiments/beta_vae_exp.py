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

from datasets import train_test_random_split, load_dsprites, CustomDSpritesDataset, AddUniformNoise
from train import train_beta_vae, test_beta_vae
from loss import loss_beta_vae
from beta_vae import BetaVAEDSprites
from utils import save_checkpoint, load_checkpoint
from torchvision import transforms

torch.manual_seed(2)
np.random.seed(2)

# load the dataset
# dataset = load_dsprites("../datasets/dsprites.npz")
# dataset = torch.from_numpy(dataset)

transform = None
# transform = transforms.Compose([
#     AddUniformNoise(-.1, .1)
# ])
dataset = CustomDSpritesDataset(load_dsprites("../datasets/dsprites.npz", False), transform=transform)

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
beta = 1
distribution = 'bernoulli'
print('Training started')

loss_list = []

print('beta and dist')
print(beta)
print(distribution)


print('Training started')
for i in range(5):
    losses = train_beta_vae(model, epochs, train_loader, optimizer, beta, distribution, device)
    loss_list.append(losses)
    save_checkpoint(model, optimizer, 'betavae_beta1_e' + str(i+1) + '0_alldata_n.pth.tar', 
        loss_list, (i+1)*10)
    print(str(i+1) + '0 Epochs')

print('Training finished')

test_beta_vae(model, test_loader, beta, distribution, device)