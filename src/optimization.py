from tabnanny import verbose
import numpy as np
from gmr import MVN
from src.amortization import to_log_probability_density
# import pytorch for AD
import torch

def log_prior(Eb, gmm):
    """   
    Compute the log prior probability of the basal enthalpy field under the GMM model.

    Parameters:
    -------
    Eb: array
        basal enthalpy field (n_features,)
    gmm: GaussianMixture
        trained GMM model representing the prior distribution over basal enthalpy fields
    """

    return to_log_probability_density(gmm, Eb)

def log_prior_gradient(Eb, gmm):
    """   
    Compute the gradient of the log prior probability with respect to Eb.
    
    For GMM: p(x) = sum_k pi_k * N(x | mu_k, Sigma_k)
    Gradient: nabla log p(x) = [sum_k r_k * nabla log N_k(x)] 
    where r_k is the responsibility (posterior weight) of component k
    
    Parameters:
    -----------
    Eb: array, shape (n_features,)
        Input vector (e.g., basal enthalpy field)
    gmm: GMM object
        Trained GMM model
    
    Returns:
    --------
    grad: array, shape (n_features,)
        Gradient of log p(Eb) with respect to Eb
    """

    n_features = Eb.shape[0]
    n_components = gmm.n_components
    
    # Step 1: Compute probability of Eb under each component
    component_log_probs = np.zeros(n_components)
    component_grads = np.zeros((n_components, n_features))
    
    for k in range(n_components):
        mvn = MVN(mean=gmm.means[k], 
                  covariance=gmm.covariances[k],
                  random_state=gmm.random_state)
        
        # Get normalization factor and exponent
        norm_factor, exponent = mvn.to_norm_factor_and_exponents(Eb)
        
        # Log probability of component k (including prior weight)
        component_log_probs[k] = np.log(gmm.priors[k]) + np.log(norm_factor) + exponent[0]
        
        # Gradient of log N(x | mu_k, Sigma_k) = -Sigma_k^{-1} (x - mu_k)
        cov_inv = np.linalg.inv(gmm.covariances[k])
        # print('shape of cov_inv:', cov_inv.shape)
        # print('shape of Eb.T.reshape(-1,1):', Eb.T.reshape(-1,1).shape)
        # print('shape of gmm.means[k].reshape(-1,1):', gmm.means[k].reshape(-1,1).shape)
        component_grads[k,:] = (-cov_inv @ (Eb.T.reshape(-1,1) - gmm.means[k].reshape(-1,1))).flatten()
    
    # Compute responsibilities (posterior weights) using log-sum-exp trick
    max_log_prob = np.max(component_log_probs)
    log_probs_stable = component_log_probs - max_log_prob
    
    # Responsibilities: r_k = p(k|x) = pi_k * N(x|mu_k,Sigma_k) / p(x)
    responsibilities = np.exp(log_probs_stable)
    responsibilities /= np.sum(responsibilities)
    
    # Weighted sum of gradients
    grad = np.sum(responsibilities[:, np.newaxis] * component_grads, axis=0)
    
    return grad

def log_prior_hessian(Eb, gmm):
    """
    Analytical Hessian of log GMM prior.
    
    H[log p(x)] = sum_k r_k(-Sigma_k^{-1} + g_k g_k^T) - (sum_k r_k g_k)(sum_k r_k g_k)^T
    
    Parameters:
    -----------
    Eb : np.ndarray, shape (n_features,)
    gmm : trained GMM object
    
    Returns:
    --------
    hessian : np.ndarray, shape (n_features, n_features)
    """
    n_features    = Eb.shape[0]
    n_components  = gmm.n_components

    component_log_probs = np.zeros(n_components)
    component_grads     = np.zeros((n_components, n_features))
    cov_invs            = []

    for k in range(n_components):
        mvn = MVN(mean=gmm.means[k],
                  covariance=gmm.covariances[k],
                  random_state=gmm.random_state)

        norm_factor, exponent = mvn.to_norm_factor_and_exponents(Eb)
        component_log_probs[k] = (np.log(gmm.priors[k])
                                  + np.log(norm_factor)
                                  + exponent[0])

        cov_inv = np.linalg.inv(gmm.covariances[k])
        cov_invs.append(cov_inv)
        diff = (Eb - gmm.means[k]).reshape(-1, 1)          
        component_grads[k] = (-cov_inv @ diff).flatten()   

    # Responsibilities via log-sum-exp (same as your gradient function)
    max_log_prob    = np.max(component_log_probs)
    responsibilities = np.exp(component_log_probs - max_log_prob)
    responsibilities /= responsibilities.sum()             

    # Hessian assembly
    weighted_hess = np.zeros((n_features, n_features))
    for k in range(n_components):
        g_k = component_grads[k].reshape(-1, 1)           
        weighted_hess += responsibilities[k] * (
            -cov_invs[k] + g_k @ g_k.T                    
        )

    # Subtract outer product of the total gradient
    mean_grad = (responsibilities[:, np.newaxis] * component_grads).sum(axis=0)
    weighted_hess -= np.outer(mean_grad, mean_grad)         

    return weighted_hess

def log_posterior_gradient():
    print("I am here")
    return

def log_posterior():
    pass
