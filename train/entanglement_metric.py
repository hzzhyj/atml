import numpy as np
import torch
from torch.utils.data import DataLoader
from train import train_classifier_metric, test_classifier_metric
from datasets import CustomClassifierDataset, train_test_random_split


def compute_latent_variance(model, dataset, size = 10000, device = None):
    
    imgs_sampled = dataset.simulate_images(size)
    latents = []
    loader = DataLoader(imgs_sampled, batch_size=64)

    for data in loader:
        data = data.float()

        if device != None:
            data = data.to(device)
        data = data.view(-1,64*64)
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        z = list(z.detach().cpu().numpy())
        latents= latents+[list(l) for l in z]

    global_vars = np.var(latents, axis = 0) 
    return global_vars

def entanglement_metric_factor_vae(model, dataset, nb_samples, sample_size, device = None):
    model.eval()

    loss = 0
    factors_nb = dataset.num_factors()
    classification = np.zeros((factors_nb,10))
    
    with torch.no_grad():
        
        global_vars = compute_latent_variance(model, dataset, device = device)
    
        for i in range(nb_samples):

            k = np.random.randint(factors_nb-1)+1
            imgs_sampled = dataset.simulate_images(sample_size, fixed_factor=k)
            loader = DataLoader(imgs_sampled, batch_size=64)
            latents_rep = []
            for data in loader:

                data = data.float()

                if device != None:
                    data = data.to(device)
                
                data = data.view(-1,64*64)
                mu, logvar = model.encode(data)
                z = model.reparameterize(mu, logvar)
                z = list(z.detach().cpu().numpy())
                latents_rep= latents_rep+[list(l) for l in z]
            latents_var = np.var(latents_rep, axis = 0) 
            latents_var_normalized = np.divide(latents_var, global_vars)
            #for i in range(10):
            #    if global_vars[i] >=1:
            #        latents_var_normalized[i] = np.inf

            idx = np.argmin(latents_var_normalized)
            classification[k,idx]+=1
        
    print(classification)
    for i in range(factors_nb):
        loss = loss + np.sum(classification[i])- np.max(classification[i])
    return loss/nb_samples

def create_beta_vae_classifier_dataset(model, dataset, nb_samples, sample_size, device=None):
    model.eval()
    factors_nb = dataset.num_factors()
    z_diffs = torch.zeros((nb_samples,10))
    factors = torch.zeros(nb_samples, dtype=torch.long)
    with torch.no_grad():
            
        for i in range(nb_samples):

            k = torch.randint(low=0, high=factors_nb-1, size=(1,), dtype=torch.long)
            factors[i] = k
            z_diff = torch.zeros((sample_size,10))
            for j in range(sample_size):
                imgs_sampled = dataset.simulate_images(2, fixed_factor=k+1)
                data1 = imgs_sampled[0].float().view(-1,64*64)
                data2 = imgs_sampled[1].float().view(-1,64*64)

                if device != None:
                    data1 = data1.to(device)
                    data2 = data2.to(device)

                mu1, _ = model.encode(data1)
                mu2, _ = model.encode(data2)

                z_diff[j] = torch.abs(mu1.squeeze()-mu2.squeeze())
                    
            z_diff = torch.mean(z_diff,0)
            z_diffs[i] = z_diff  
        
    data = CustomClassifierDataset(factors, z_diffs)
    return data

def entanglement_metric_beta_vae(model, classifier, optimizer, epochs, dataset, nb_samples, sample_size, device=None):
    
    data = create_beta_vae_classifier_dataset(model, dataset, nb_samples, sample_size, device=device)
    data_train, data_test = train_test_random_split(data, 0.8)
    batch_size = 10
    train_loader = DataLoader(data_train, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size,shuffle=False) 
    
    losses, accuracies = train_classifier_metric(classifier, epochs, train_loader, optimizer, device=device)
    test_classifier_metric(classifier, test_loader, device=device)
    return losses, accuracies


    

