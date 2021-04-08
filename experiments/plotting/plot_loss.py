import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path+"/models")
sys.path.append(module_path+"/train")
sys.path.append(module_path+"/experiments")


def get_cvae_filename(cmax, losstype, file_list):
    for fn in file_list:
        if("Cmax" + str(cmax) in fn and losstype in fn):
            return fn
    return None

def get_fvae_filename(gamma, losstype, file_list):
    for fn in file_list:
        if("gamma" + str(gamma) in fn and losstype in fn):
            return fn
    return None

def get_bvae_filename(beta, file_list):
    for fn in file_list:
        if("beta" + str(beta) in fn and "n_losses" in fn):
            return fn
    return None

loss_file_path = 'experiments/trained_models'
loss_filenames = [f for f in os.listdir(loss_file_path) if f[-4:] =='.npy' and not "alldata" in f]


cvae_cmax_list = [8, 10, 12]
fvae_gamma_list = [5, 40]
bvae_beta_list = [1, 4]
# Test get filebane functions
print(get_cvae_filename(cvae_cmax_list[0], "recon", loss_filenames))
print(get_fvae_filename(fvae_gamma_list[0], "recon", loss_filenames))
print(get_bvae_filename(bvae_beta_list[0], loss_filenames))

num_epoch = 50
epoch_steps = [i for i in range(1, num_epoch + 1)]

# Separate losses for beta vae
beta_recon_losses = {}
beta_kl_div = {}
for beta in bvae_beta_list:
    filename = get_bvae_filename(beta, loss_filenames)
    data = np.load(loss_file_path + "/" + filename)
    beta_recon_losses[beta] = data[:, 1]
    kl_div_data = (data[:, 0] - data[:, 1]) / float(beta)
    beta_kl_div[beta] = kl_div_data

# Plot recon loss
# c = np.arange(1, len(cvae_cmax_list) + 4)
# norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
# cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
# cmap.set_array([])
# i = 0
# for cmax in cvae_cmax_list:
#     filename =  get_cvae_filename(cmax, "recon", loss_filenames)
#     data = np.load(loss_file_path + "/" + filename)
#     assert data.shape[0] == num_epoch
#     plt.plot(epoch_steps, data, c=cmap.to_rgba(c.shape[0] - i - 1), label = "ControlVAE_Cmax" + str(cmax))
#     i += 1
# c = np.arange(1, len(cvae_cmax_list) + 3)
# norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
# cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
# cmap.set_array([])
# i = 0
# for gamma in fvae_gamma_list:
#     filename =  get_fvae_filename(gamma, "recon", loss_filenames)
#     data = np.load(loss_file_path + "/" + filename)
#     assert data.shape[0] == num_epoch
#     plt.plot(epoch_steps, data, c=cmap.to_rgba(c.shape[0] - i - 1), label = "FactorVAE_Gamma" + str(gamma))
#     i += 1
#
# c = np.arange(1, len(cvae_cmax_list) + 3)
# norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
# cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Oranges)
# cmap.set_array([])
# i = 0
# for beta in bvae_beta_list:
#     data = beta_recon_losses[beta]
#     assert data.shape[0] == num_epoch
#     plt.plot(epoch_steps, data, c=cmap.to_rgba(c.shape[0] - i - 1), label = "BetaVAE_Beta" + str(beta))
#     i += 1
#
# # Plot settings
# plt.legend()
# plt.xlabel("Epoch")
# plt.ylabel("Reconstruction Loss")
# plt.title("Reconstruction loss over time")
# plt.show()


# Plot Kl Divergence
c = np.arange(1, len(cvae_cmax_list) + 4)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap.set_array([])
i = 0
for cmax in cvae_cmax_list:
    filename =  get_cvae_filename(cmax, "kl_divs", loss_filenames)
    data = np.load(loss_file_path + "/" + filename)
    assert data.shape[0] == num_epoch
    plt.plot(epoch_steps, data, c=cmap.to_rgba(c.shape[0] - i - 1), label = "ControlVAE_Cmax" + str(cmax))
    i += 1
c = np.arange(1, len(cvae_cmax_list) + 3)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
cmap.set_array([])
i = 0
for gamma in fvae_gamma_list:
    filename =  get_fvae_filename(gamma, "kl_divs", loss_filenames)
    data = np.load(loss_file_path + "/" + filename)
    assert data.shape[0] == num_epoch
    plt.plot(epoch_steps, data, c=cmap.to_rgba(c.shape[0] - i - 1), label = "FactorVAE_Gamma" + str(gamma))
    i += 1

c = np.arange(1, len(cvae_cmax_list) + 3)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Oranges)
cmap.set_array([])
i = 0
for beta in bvae_beta_list:
    data = beta_kl_div[beta]
    assert data.shape[0] == num_epoch
    plt.plot(epoch_steps, data, c=cmap.to_rgba(c.shape[0] - i - 1), label = "BetaVAE_Beta" + str(beta))
    i += 1

# Plot settings
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("KL Divergence")
plt.title("KL Divergence over time")
plt.show()
