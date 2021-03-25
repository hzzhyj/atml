import numpy as np
import torch
from torch.utils.data import DataLoader

def compute_latent_variance(model, dataset, size = 10000, device = None):
    
    imgs_sampled = dataset.simulate_images(size)
    latents = []
    loader = DataLoader(imgs_sampled, batch_size=64)

    for data in loader:
        data = data.float()

        if device != None:
            data = data.to(device)

        _, _, _, z = model(data)
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

                _, _, _, z = model(data)
                z = list(z.detach().cpu().numpy())
                latents_rep= latents_rep+[list(l) for l in z]
            latents_var = np.var(latents_rep, axis = 0) 
            latents_var_normalized = np.divide(latents_var, global_vars)
            for i in range(10):
                if global_vars[i] >=1:
                    latents_var_normalized[i] = np.inf

            idx = np.argmin(latents_var_normalized)
            classification[k,idx]+=1
        
    print(classification)
    for i in range(factors_nb):
        loss = loss + np.sum(classification[i])- np.max(classification[i])
    return loss/nb_samples