import torch
import numpy as np

class custom_potential:
    def __init__(self, model_nf, model_pca, attenu_obs, attenu_obs_std, radar_mask, verbose = False):
        self.model_nf = model_nf
        self.model_pca = model_pca
        self.attenu_obs = torch.tensor(attenu_obs.flatten())
        self.attenu_obs_std = torch.tensor(attenu_obs_std.flatten())
        self.radar_mask = radar_mask.flatten()
        self.verbose = verbose
        self.n = self.attenu_obs_std[self.radar_mask].shape[0]
        self.gaussian_negative_log_likelihood = torch.nn.GaussianNLLLoss(reduction = 'sum')
        self.uniform_std = 0.1
        print(f"Likelihood Gaussian rank = {self.n}")

        #self.gaussian_obs_image = torch.distributions.Normal(torch.tensor(attenu_obs), torch.full(attenu_obs.shape, 0.1))

    def log_posterior(self, params):
        x_full = params['x']

        # handle batch input
        # if x_full.shape[0] > 1:
        log_p_posterior_sum = 0
        for i in range(x_full.shape[0]):
            x = x_full[i:i+1, :]
            z = x
            # print(x.shape)


    # log_likelihood = torch.empty(1)
    # log_likelihood.requires_grad_()
            log_p_prior_sample = self.model_nf.log_prob(x)
            #print(log_p_prior_sample)

            #z = self.model_nf.forward(x)

            z_image = torch.matmul(z, torch.tensor(self.model_pca.components_).float()) + torch.tensor(self.model_pca.mean_)
            #gaussian_z_image = torch.distributions.Normal(z_image, torch.full(attenu_obs.shape, 0.1))
            # print(z_image.shape)
            z_image = z_image.flatten()


    # z_image = x.flatten()
    #std = 2
    #n = z_image.shape[0]
    # print(z_image[self.radar_mask].shape)
    # print(self.attenu_obs_std[self.radar_mask].shape)
    # print(self.attenu_obs[self.radar_mask].shape)
    # n = self.attenu_obs_std[self.radar_mask].shape[0]
    # print(n)
    # print(((self.attenu_obs[self.radar_mask] - z_image[self.radar_mask]) ** 2))
    # print(1/(self.attenu_obs_std[self.radar_mask] ** 2))
    # print(((self.attenu_obs[self.radar_mask] - z_image[self.radar_mask]) ** 2))


    # analytical gaussian
    # log_likelihood = -0.5 * torch.sum((1/(self.attenu_obs_std[self.radar_mask] ** 2)) * ((self.attenu_obs[self.radar_mask] - z_image[self.radar_mask]) ** 2)) - np.log(np.sqrt(((2 * np.pi) ** n) * (torch.prod(self.attenu_obs_std ** 2))))
    # log_likelihood = -0.5 * torch.sum((1/(self.attenu_obs_std[self.radar_mask] ** 2)) * ((self.attenu_obs[self.radar_mask] - z_image[self.radar_mask]) ** 2)) #- np.log(np.sqrt(((2 * np.pi) ** n) * (torch.prod(self.attenu_obs_std ** 2))))
    # log_likelihood = log_likelihood / self.n
    #likelihood = -0.5 * ((1/(std ** 2)) * torch.sum((self.attenu_obs.detach().clone() - z_image) ** 2)) - np.log(np.sqrt(((2 * np.pi) ** n) * ((std ** 2) ** n)))
    #print(gaussian_z_image.log_prob(self.attenu_obs)[0][0])

    # MSE
    # log_likelihood = - torch.mean((self.attenu_obs.detach().clone()[self.radar_mask.flatten()] - z_image) ** 2) * 1e3

    # torch implementation
    # input = z_image[self.radar_mask.flatten()]
    # target = self.attenu_obs[self.radar_mask.flatten()]
    # var = torch.ones(target.shape[0]) * (self.uniform_std ** 2)
    # # var = self.attenu_obs_std[self.radar_mask.flatten()] ** 2
    # log_likelihood = - self.gaussian_negative_log_likelihood(input, target, var)

    # analytical gaussian assuming identical std across all pixels
            #log_likelihood_sample = -0.5 * torch.sum(((1/(self.uniform_std ** 2)) * ((self.attenu_obs[self.radar_mask] - z_image[self.radar_mask]) ** 2))) - (self.n / 2) * np.log(np.sqrt(((2 * np.pi * self.uniform_std))))
            
            # analytical gaussian
            log_likelihood_sample = -0.5 * torch.sum((1/(self.attenu_obs_std[self.radar_mask] ** 2)) * ((self.attenu_obs[self.radar_mask] - z_image[self.radar_mask]) ** 2)) #- np.log(np.sqrt(((2 * np.pi) ** n) * (torch.prod(self.attenu_obs_std ** 2))))

            #print(f"SE: {torch.sum((self.attenu_obs.detach().clone()[self.radar_mask.flatten()] - z_image[self.radar_mask.flatten()]) ** 2)}, {x[0][10]}, prior log prob: {log_p_prior_sample[0]:.5f}, log likelihood: {log_likelihood_sample:.5f}")



            log_p_posterior_sum += log_p_prior_sample + log_likelihood_sample
                # log_p_posterior_sum += log_likelihood_sample
                # print(log_likelihood_sample)
                # print(i)
                # print(log_likelihood_sum)
        # attenu_obs_latent = torch.tensor(self.model_pca.transform(self.attenu_obs.reshape(1, -1)))
        # n = attenu_obs_latent.shape[0]
        # # log_likelihood = -0.5 * torch.sum(((1/(self.uniform_std ** 2)) * ((attenu_obs_latent - z) ** 2))) - (self.n / 2) * np.log(np.sqrt(((2 * np.pi * self.uniform_std))))
        # log_likelihood = -0.5 * torch.sum((1/(self.attenu_obs_std ** 2)) * ((attenu_obs_latent - z) ** 2)) - np.log(np.sqrt(((2 * np.pi) ** n) * (torch.prod(self.attenu_obs_std ** 2))))
        
        np.savez('z_image.npz', z_image = z_image.detach().numpy())

        # if self.verbose:
        #     print(f"SE: {torch.sum((self.attenu_obs.detach().clone()[self.radar_mask.flatten()] - z_image[self.radar_mask.flatten()]) ** 2)}, {x[0][10]}, prior log prob: {log_p_prior[0]:.5f}, log likelihood: {log_likelihood:.5f}")
        # print(log_p_prior.dtype)
        # print(log_likelihood.dtype)
        #return -(log_p_prior + log_likelihood)
        return - log_p_posterior_sum.to(torch.float32)
        # return log_likelihood