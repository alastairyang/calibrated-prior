import torch
import numpy as np

class custom_potential:
    def __init__(self, model_nf, model_pca, attenu_obs, verbose = False):
        self.model_nf = model_nf
        self.model_pca = model_pca
        self.attenu_obs = torch.tensor(attenu_obs)
        self.verbose = verbose

        #self.gaussian_obs_image = torch.distributions.Normal(torch.tensor(attenu_obs), torch.full(attenu_obs.shape, 0.1))

    def log_posterior(self, params):
        x = params['x']

        log_p_prior = self.model_nf.log_prob(x)

        z = self.model_nf.inverse(x)
        z_image = torch.matmul(z, torch.tensor(self.model_pca.components_).float()) + torch.tensor(self.model_pca.mean_)
        #gaussian_z_image = torch.distributions.Normal(z_image, torch.full(attenu_obs.shape, 0.1))

        #print(gaussian_z_image.log_prob(self.attenu_obs)[0][0])

        mean_se = - torch.mean((self.attenu_obs.detach().clone() - z_image) ** 2) * 1e3

        if self.verbose:
            print(f"prior log prob: {log_p_prior[0]:.2f}, log likelihood: {mean_se:.2f}")
    
        return -(log_p_prior + mean_se)