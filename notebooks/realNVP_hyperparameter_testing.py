# Import required packages
import torch
import numpy as np
import normflows as nf
from torch.utils.data import TensorDataset, DataLoader, random_split

from matplotlib import pyplot as plt

from tqdm import tqdm

import joblib
import sklearn
import datetime

import sys
sys.path.append('../src')
import ice

from scipy.io import loadmat

max_epoch = 50000

Em = loadmat("../data/NGrIS/ensemble_data_pixelated/enthalpy_avg_grid.mat")['enthalpy_avg_grid']
pmp = loadmat("../data/NGrIS/ensemble_data_pixelated/pmp_grid.mat")['pmp_grid']
radar_mask = loadmat("../data/NGrIS/ensemble_data_pixelated/radar_mask.mat")['radar_mask']
radar_mask = (radar_mask == 1)

Em = np.array(Em)
pmp = pmp[:, :, np.newaxis]
pmp = np.tile(pmp, (1, 1, Em.shape[2]))

# convert Em to attenu rate
Tm = ice.enthalpy_to_temperature(Em, Tpmp = pmp, istorch = False)

# convert Tm to attenu rate
attenu_fromEm = ice.temperature_to_atten_rate_ice(Tm)

non_convergence_idx = np.arange(attenu_fromEm.shape[2])[np.all(np.isnan(attenu_fromEm), axis = (0, 1))]
attenu_fromEm = np.delete(attenu_fromEm, non_convergence_idx, axis = 2)
# attenu_fromEm = np.moveaxis(attenu_fromEm, 2, 0) # sample, image_dimension, image_dimension

batch_size = [32, 64]
lr = [5e-5, 1e-5]
hidden_layer_neurons = [64, 128]
num_layers = [32, 64]
std_relaxation = [0, 1, 2]

import itertools
for batch_size, lr, hidden_layer_neurons, num_layers, std_relaxation in itertools.product(batch_size, lr, hidden_layer_neurons, num_layers, std_relaxation):
    print(f"{batch_size}_{lr}_{hidden_layer_neurons}_{num_layers}_{std_relaxation}")
    attenu_fromEm_mean = np.nanmean(attenu_fromEm, axis = 2)[:, :, np.newaxis]
    attenu_fromEm_std = np.nanstd(attenu_fromEm, axis = 2)[:, :, np.newaxis]
    attenu_fromEm_std = attenu_fromEm_std + std_relaxation
    attenu_fromEm_mean = np.tile(attenu_fromEm_mean, (1, 1, attenu_fromEm.shape[2]))
    attenu_fromEm_std = np.tile(attenu_fromEm_std, (1, 1, attenu_fromEm.shape[2]))
    attenu_fromEm_standard = (attenu_fromEm - attenu_fromEm_mean) / attenu_fromEm_std
    attenu_fromEm_standard[np.tile(radar_mask[:, :, np.newaxis], (1, 1, attenu_fromEm_standard.shape[2])) != 1] = 0
    attenu_fromEm_standard = np.moveaxis(attenu_fromEm_standard, -1, 0)
    attenu_fromEm_standard = attenu_fromEm_standard[:, np.newaxis, :, :]
    attenu_fromEm_standard[np.isnan(attenu_fromEm_standard)] = 0
    attenu_fromEm_standard = torch.tensor(attenu_fromEm_standard).float()
    np.savez(f'./saved_data/attenu_fromEm_standard_{batch_size}_{lr}_{hidden_layer_neurons}_{num_layers}_{std_relaxation}.npz', attenu_fromEm_standard = attenu_fromEm_standard)

    attenu_pca = sklearn.decomposition.PCA(n_components = 32)
    attenu_pca.fit(attenu_fromEm_standard[:, 0, :, :].reshape((-1, attenu_fromEm_standard.shape[2] ** 2)))
    joblib.dump(attenu_pca, f'./saved_data/attenu_pca_{batch_size}_{lr}_{hidden_layer_neurons}_{num_layers}_{std_relaxation}.pkl')
    attenu_latent = attenu_pca.transform(attenu_fromEm_standard[:, 0, :, :].reshape((-1, attenu_fromEm_standard.shape[2] ** 2)))
    attenu_latent = torch.tensor(attenu_latent).float()

    # train val test split
    attenu_latent_dataset = TensorDataset(attenu_latent, attenu_latent)
    #attenu_latent_dataset = TensorDataset(attenu_latent_standard, attenu_latent_standard)
    train_size = (int(0.7 * len(attenu_latent_dataset)) // batch_size) * batch_size
    val_size = int((len(attenu_latent_dataset) - train_size)/2)
    test_size = int(len(attenu_latent_dataset) - train_size - val_size)
    generator = torch.Generator().manual_seed(42)
    attenu_latent_train_dataset, attenu_latent_val_dataset, attenu_latent_test_dataset = random_split(attenu_latent_dataset, [train_size, val_size, test_size], generator=generator)
    attenu_latent_train_dataloader = DataLoader(attenu_latent_train_dataset, batch_size = batch_size, shuffle = 1)
    attenu_latent_val_dataloader = DataLoader(attenu_latent_val_dataset, batch_size = batch_size, shuffle = 1)
    attenu_latent_test_dataloader = DataLoader(attenu_latent_test_dataset, batch_size = batch_size, shuffle = 1)

    # Set up model_nf
    in_features = int(attenu_latent.shape[1]/2)
    # Define list of flows
    flows = []
    for i in range(num_layers):
        param_map = nf.nets.MLP([in_features, hidden_layer_neurons, hidden_layer_neurons, in_features * 2], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(in_features * 2, mode='swap'))

    # base distribution is a Gaussian with num of dimension same as training data
    base = nf.distributions.base.DiagGaussian(in_features * 2)
    # Construct flow model_nf
    model_nf = nf.NormalizingFlow(base, flows)

    # using linear regression to prevent overfitting
    window_size = 250

    reg = sklearn.linear_model.LinearRegression()
    window = np.arange(window_size).reshape(-1, 1)

    # Train model_nf
    early_stopping_threshold = 1e-7

    epoch_loss = np.array([])
    epoch_loss_val = np.array([])

    #optimizer = torch.optim.Adam(model_nf.parameters(), lr=1e-5, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model_nf.parameters(), lr=lr)
    model_nf.train()

    try:
        for epoch in range(max_epoch):
            optimizer.zero_grad()

            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)
            start.record()

            running_loss = 0.0
            running_loss_val = 0.0

            for train_batch, _ in attenu_latent_train_dataloader:
                # Compute loss
                loss = model_nf.forward_kld(train_batch)
                
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * train_batch.size(0)

            for val_batch, _ in attenu_latent_val_dataloader:
                # Compute loss
                val_loss = model_nf.forward_kld(val_batch)

                # Do backprop and optimizer step
                if ~(torch.isnan(val_loss) | torch.isinf(val_loss)):
                    val_loss.backward()
                    optimizer.step()
                    running_loss_val += val_loss.item() * val_batch.size(0)
            
            #print(running_loss)

            epoch_loss = np.append(epoch_loss, running_loss / len(attenu_latent_train_dataloader.dataset))
            epoch_loss_val = np.append(epoch_loss_val, running_loss_val / len(attenu_latent_val_dataloader.dataset))

            if epoch >= window.shape[0]:
                #print(epoch_loss[-window_size:])
                #print(epoch_loss_val[-window_size:])
                reg.fit(window, epoch_loss[-window_size:])
                epoch_loss_slope = reg.coef_
                
                reg.fit(window, epoch_loss_val[-window_size:])
                epoch_loss_val_slope = reg.coef_
                
                if epoch%100 == 0:
                    print(f"Epoch loss slope: {epoch_loss_slope}")
                    print(f"Epoch validation loss slope: {epoch_loss_val_slope}")

                if (epoch_loss_slope > -early_stopping_threshold) or (epoch_loss_val_slope > -early_stopping_threshold):
                    print("Early stopping")
                    break

            end.record()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            if epoch%100 == 0:
                print(f"Epoch [{epoch + 1}/{max_epoch}], loss: {epoch_loss[epoch]:.4f}, validation loss: {epoch_loss_val[epoch]:.4f}, time elapsed = {start.elapsed_time(end)/1000:.2f} s")
            
        torch.save(model_nf.state_dict(), f'./saved_data/realNVP_{batch_size}_{lr}_{hidden_layer_neurons}_{num_layers}_{std_relaxation}.pth')
        np.savez(f'./saved_data/realNVP_loss_{batch_size}_{lr}_{hidden_layer_neurons}_{num_layers}_{std_relaxation}.npz',epoch_loss = epoch_loss, epoch_loss_val = epoch_loss_val) 

    except KeyboardInterrupt:
        torch.save(model_nf.state_dict(), f'/saved_data/realNVP_{batch_size}_{lr}_{hidden_layer_neurons}_{num_layers}_{std_relaxation}.pth')
        np.savez(f'/saved_data/realNVP_loss_{batch_size}_{lr}_{hidden_layer_neurons}_{num_layers}_{std_relaxation}.npz',epoch_loss = epoch_loss, epoch_loss_val = epoch_loss_val)
