import numpy as np
import torch
from torch.utils.data import DataLoader
from train import train_classifier_metric, test_classifier_metric
from datasets import CustomClassifierDataset, train_test_random_split
from datasets import CustomClassifierDataset, train_test_random_split, CustomDSpritesDataset
from sklearn import metrics

def compute_latent_variance(model, dataset, size=10000, device=None):
    imgs_sampled = dataset.simulate_images(size)
    latents = []
    loader = DataLoader(imgs_sampled, batch_size=64)

    for data in loader:
        data = data.float()

        if device != None:
            data = data.to(device)
        mu, _ = model.encode(data)
        z = list(mu.detach().cpu().numpy())
        latents= latents+[list(l) for l in z]

    global_vars = np.var(latents, axis = 0)
    return global_vars

def entanglement_metric_factor_vae(model, dataset, n_samples, sample_size, n_latents=10, random_seeds=1, device=None, seed=None):
    model.eval()
    losses=[]
    classifications=[]
    if seed!=None:
        np.random.seed(seed)
    for i in range(random_seeds):
        loss = 0
        factors_nb = dataset.num_factors()
        classification = np.zeros((n_latents,factors_nb))

        with torch.no_grad():

            global_vars = compute_latent_variance(model, dataset, device = device)

            for i in range(n_samples):

                k = np.random.randint(factors_nb-1)+1
                imgs_sampled = dataset.simulate_images(sample_size, fixed_factor=k)
                loader = DataLoader(imgs_sampled, batch_size=64)
                latents_rep = []
                for data in loader:

                    data = data.float()

                    if device != None:
                        data = data.to(device)

                    mu, _ = model.encode(data)
                    z = list(mu.detach().cpu().numpy())
                    latents_rep= latents_rep+[list(l) for l in z]
                latents_var = np.var(latents_rep, axis = 0)
                latents_var_normalized = np.divide(latents_var, global_vars)

                idx = np.argmin(latents_var_normalized)
                classification[idx,k]+=1

        classifications.append([classification])
        for i in range(n_latents):
            loss = loss + np.sum(classification[i])- np.max(classification[i])
        losses.append(loss/n_samples)
    print("accuracies : "+str([(1-x) for x in losses]))
    print(np.mean(classifications,0))
    print (1-np.mean(losses))
    return ([(1-x) for x in losses])

def create_beta_vae_classifier_dataset(model, dataset, n_samples, sample_size, n_latents=10, device=None):
    model.eval()
    factors_nb = dataset.num_factors()
    z_diffs = torch.zeros((n_samples,n_latents))
    factors = torch.zeros(n_samples, dtype=torch.long)
    with torch.no_grad():

        for i in range(n_samples):

            k = torch.randint(low=0, high=factors_nb-1, size=(1,), dtype=torch.long)
            factors[i] = k
            z_diff = torch.zeros((sample_size,n_latents))
            for j in range(sample_size):
                imgs_sampled = dataset.simulate_images(2, fixed_factor=k+1)
                data1 = imgs_sampled[0].float()
                data2 = imgs_sampled[1].float()

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

def create_beta_vae_classifier_dataset_fast(model, dataset, n_samples, sample_size, n_latents=10, device=None):
    model.eval()
    factors_nb = dataset.num_factors()
    z_diffs = torch.zeros((n_samples,n_latents))
    factors = torch.zeros(n_samples, dtype=torch.long)
    with torch.no_grad():

        for i in range(n_samples):

            k = torch.randint(low=0, high=factors_nb-1, size=(1,), dtype=torch.long)
            factors[i] = k

            # Sample images first - 2 * sample_size
            imgs_sampled = dataset.simulate_images(2 * sample_size, fixed_factor=k+1).float()
            # Get the first half
            data1 = imgs_sampled[:sample_size]
            data2 = imgs_sampled[sample_size:]
            if device != None:
                data1 = data1.to(device)
                data2 = data2.to(device)

            mu1, _ = model.encode(data1)
            mu2, _ = model.encode(data2)

            z_diffs[i] =  torch.mean(torch.abs(mu1 - mu2), 0)

    data = CustomClassifierDataset(factors, z_diffs)
    return data

def entanglement_metric_beta_vae(model, classifier, optimizer, epochs, dataset, n_samples, sample_size, n_latents=10, random_seeds=1, device=None, seed=None):
    test_accuracies=[]
    train_losses=[]
    train_accuracies=[]
    if seed!=None:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
    for i in range(random_seeds):
        classifier.reset_parameters()
        data = create_beta_vae_classifier_dataset(model, dataset, n_samples, sample_size, n_latents=n_latents, device=device)
        #data = create_beta_vae_classifier_dataset_fast(model, dataset, n_samples, sample_size, n_latents=n_latents, device=device)
        data_train, data_test = train_test_random_split(data, 0.8,seed=seed)
        batch_size = 10
        train_loader = DataLoader(data_train, batch_size=batch_size,shuffle=True)
        test_loader = DataLoader(data_test, batch_size=batch_size,shuffle=False)

        losses, accuracies = train_classifier_metric(classifier, epochs, train_loader, optimizer, device=device)
        train_accuracies+=[accuracies]
        train_losses+=[losses]
        test_accuracies.append(test_classifier_metric(classifier, test_loader, device=device))
    return train_losses, train_accuracies, test_accuracies


###################################
#  Functions for calculating MIG  #
###################################

def discretize_matrix(matrix, num_bins):
    '''
    The matrix would get discretized along its rows
    '''
    discretized_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        discretized_matrix[i: ] = np.digitize(matrix[i: ], 
                                              bins=np.histogram_bin_edges(matrix[i: ], num_bins)[:-1])
        
    return discretized_matrix

def get_factor_and_z_matrices(model, dataset:CustomDSpritesDataset, num_samples:int, batch_size:int, device=None):

    model = model.to(device)

    factor_matrix = None
    z_matrix = None
    
    i = 0
    while i < num_samples:
        ns = min(num_samples - i, batch_size)

        sampled_factors = dataset.sample_latent(ns)
        sampled_indices = dataset.latent_to_index(sampled_factors)
        sampled_x = dataset[sampled_indices].type(torch.float).to(device)
        sampled_z = model.get_latent_representation(sampled_x).cpu().detach().numpy()

        if factor_matrix is None:
            factor_matrix = sampled_factors[:]
            z_matrix = sampled_z[:]
        else:
            factor_matrix = np.vstack((factor_matrix, sampled_factors))
            z_matrix = np.vstack((z_matrix, sampled_z))

        i += ns
  
    factor_matrix = factor_matrix.transpose()
    z_matrix = z_matrix.transpose()

    return factor_matrix, z_matrix

def compute_mig(model, dataset, num_samples=100000, batch_size=1024, random_seeds=1, device=None, seed= None):
    mig_scores=[]
    if seed!=None:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
    for i in range(random_seeds):
        factor_matrix, z_matrix = get_factor_and_z_matrices(model, dataset, num_samples, batch_size, device)
        z_matrix = discretize_matrix(z_matrix, num_bins=20)

        factor_matrix = factor_matrix.astype('uint8')
        z_matrix = z_matrix.astype('uint8') 

        mutual_info_matrix = np.zeros((z_matrix.shape[0], factor_matrix.shape[0])) # z_dim * num of factors
        for i in range(z_matrix.shape[0]):
            for j in range(factor_matrix.shape[0]):
                mutual_info_matrix[i, j] = metrics.mutual_info_score(z_matrix[i, :], factor_matrix[j, :])

        sorted_mi_matrix = np.sort(mutual_info_matrix, axis=0)[::-1]

        factor_entropies = np.zeros(factor_matrix.shape[0])
        for i in range(len(factor_entropies)):
            factor_entropies[i] = metrics.mutual_info_score(factor_matrix[i, :], factor_matrix[i, :])

        factor_entropies[factor_entropies == 0] = np.nan
        mig = np.nanmean(np.divide((sorted_mi_matrix[0, :] - sorted_mi_matrix[1, :]), factor_entropies))
        mig_scores.append(mig)
    return mig_scores


