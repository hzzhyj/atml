from loss import loss_beta_vae, loss_control_vae, loss_factor_vae, loss_discriminator
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm.notebook import tqdm as ntqdm
from tqdm.notebook import tqdm


def train_beta_vae(model, epochs, train_loader, optimizer, beta, distribution, 
            dataset, transform=None, transform_needs_latents=False, device=None):
    model.train()

    train_loss = []
    for epoch in tqdm(range(epochs), leave=False):
        epoch_loss = []
        recon_losses = []
        for batch_idx, data_idx in enumerate(tqdm(train_loader, leave=False)):

            data = dataset[data_idx]

            if transform is not None:
                if transform_needs_latents:
                    lt_mx = dataset.latent_matrix[data_idx]
                    latent_values = dataset.retrieve_latent_values(lt_mx)
                    data = transform(data, latent_values)
                else:
                    data = transform(data)
            
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


def test_beta_vae(model, test_loader, beta, distribution, 
            dataset, transform=None, transform_needs_latents=False, device=None):
    model.eval()

    test_loss = []
    recon_losses = []

    with torch.no_grad():
        for i, data_idx in enumerate(tqdm(test_loader, leave=False)):

            data = dataset[data_idx]

            if transform is not None:
                if transform_needs_latents:
                    lt_mx = dataset.latent_matrix[data_idx]
                    latent_values = dataset.retrieve_latent_values(lt_mx)
                    data = transform(data, latent_values)
                else:
                    data = transform(data)

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
    #print("Test total loss: " + str(test_loss) + 'Test recon loss: ' + str(recon_losses))
    return recon_losses

def train_control_vae(model, epochs, train_loader, optimizer, distribution,
                    dataset, transform=None, transform_needs_latents=False, device=None):
    model.train()

    train_loss = []
    recon_losses_list = []
    kl_divs_list = []
    iter_idx = 0

    for epoch in ntqdm(range(epochs), leave=False):
        epoch_loss = []
        recon_losses = []
        kl_divs = []

        for batch_idx, data_idx in enumerate(ntqdm(train_loader, leave=False)):

            iter_idx += 1

            data = dataset[data_idx]

            if transform is not None:
                if transform_needs_latents:
                    lt_mx = dataset.latent_matrix[data_idx]
                    latent_values = dataset.retrieve_latent_values(lt_mx)
                    data = transform(data, latent_values)
                else:
                    data = transform(data)

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


def test_control_vae(model, test_loader, distribution, 
        dataset, transform=None, transform_needs_latents=False, device=None):
    model.eval()

    test_recon_loss = []
    test_kl_div = []

    with torch.no_grad():
        for i, data_idx in enumerate(ntqdm(test_loader, leave=False)):

            data = dataset[data_idx]

            if transform is not None:
                if transform_needs_latents:
                    lt_mx = dataset.latent_matrix[data_idx]
                    latent_values = dataset.retrieve_latent_values(lt_mx)
                    data = transform(data, latent_values)
                else:
                    data = transform(data)

            data = data.float()
            
            if device != None:
                data = data.to(device)
            # report the average loss over the test dataset
            recon, mu, logvar = model(data)
            recon_loss, kl_div = loss_control_vae(recon, data, mu, logvar, distribution)
            test_recon_loss.append(recon_loss.item())
            test_kl_div.append(kl_div.item())
    test_recon_loss = np.mean(test_recon_loss)
    test_kl_div = np.mean(test_kl_div)
    #print("Test recon loss: " + str(test_recon_loss) + ", kl div: " + str(test_kl_div))
    return test_recon_loss


def random_permute(v):
    batch_size, latent_dim = v.size()
    new_v = torch.zeros_like(v)

    for j in range(latent_dim):
        permutation = torch.randperm(batch_size)
        for i in range(batch_size):
            new_v[i,j] = v[permutation[i],j]

    return new_v

def train_factor_vae(model, discriminator, epochs, train_loader, vae_optimizer, discriminator_optimizer, gamma, distribution, 
                dataset, transform=None, transform_needs_latents=False, device=None):
    model.train()
    train_losses_list = []
    recon_losses_list = []
    kl_divs_list = []
    tc_losses_list = []
    discriminator_losses_list = []

    for epoch in ntqdm(range(epochs), leave=False):
        epoch_losses = []
        recon_losses = []
        kl_divs = []
        tc_losses = []
        discriminator_losses = []

        for data1_idx in ntqdm(train_loader, leave=False):
            
            data2_idx = dataset.shuffled_indices[data1_idx]
            
            data1 = dataset[data1_idx]
            data2 = dataset[data2_idx]

            if transform is not None:
                if transform_needs_latents:
                    lt_mx = dataset.latent_matrix[data1_idx]
                    latent_values = dataset.retrieve_latent_values(lt_mx)
                    data1 = transform(data1, latent_values)

                    lt_mx = dataset.latent_matrix[data2_idx]
                    latent_values = dataset.retrieve_latent_values(lt_mx)
                    data2 = transform(data2, latent_values)

                else:
                    data1 = transform(data1)
                    data2 = transform(data2)

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

def test_factor_vae(model, discriminator, test_loader, gamma, distribution, 
        dataset, transform=None, transform_needs_latents=False, device=None):
    model.eval()

    test_losses = []
    recon_losses = []
    with torch.no_grad():
        
        for data_idx in ntqdm(test_loader, leave=False):

            data = dataset[data_idx]

            if transform is not None:
                if transform_needs_latents:
                    lt_mx = dataset.latent_matrix[data_idx]
                    latent_values = dataset.retrieve_latent_values(lt_mx)
                    data = transform(data, latent_values)
                else:
                    data = transform(data)

            data = data.float()

            if device != None:
                data = data.to(device)
            # report the average loss over the test dataset
            recon, mu, logvar, z = model(data)
            dz = discriminator(z)
            recon_loss, kl_div, tc_loss = loss_factor_vae(recon, data, mu, logvar, dz, distribution)
            loss =recon_loss + kl_div + torch.mul(gamma, tc_loss)
            test_losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            
    test_loss = np.mean(test_losses)
    test_recon_loss = np.mean(recon_losses)
    #print("Test loss: " + str(test_loss))
    #print("Test recon loss: " + str(test_recon_loss))
    return test_recon_loss

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
    