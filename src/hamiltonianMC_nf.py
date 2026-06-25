import torch
import numpy as np

class custom_energy(torch.autograd.Function):
    """
    Bridges the pure PyTorch/NumPy hybrid log-posterior directly into Pyro.
    """
    @staticmethod
    def forward(ctx, x, model_nf):
        # x in target distribution space

        print('forward')
        ctx.save_for_backward(x)
        ctx.model_nf = model_nf
        
        with torch.no_grad():
            lp = model_nf.log_prob(x)
            if isinstance(lp, torch.Tensor):
                lp = lp.item()
                
        return torch.tensor(-lp, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):

        print('backward')
        x, = ctx.saved_tensors
        x_np = x.detach().cpu().numpy()
        
        # CRITICAL FIX: Re-enable gradient tracking so the internal AD graph can build!
        with torch.enable_grad():
            grad_tensor = - torch.autograd.grad(model_nf.log_prob(x), x, retain_graph= True)
        
        grad_potential = grad_tensor.detach().numpy()
        
        return grad_tensor * grad_output
    
class whitened_potential:
    """ 
    Apply a coordinate transformation to Hamiltonian Monte Carlo (HMC) to 
    make posterior near MAP more isotropic. 
    """
    def __init__(self, X_opt_tensor, L, V, gmm, beta,beta_w,
                       Tpmp, Eb_mean, Eb_std, dw, df, Eb_epsilon):
        self.X_opt_tensor = X_opt_tensor
        self.L = L
        self.V = V
        self.gmm = gmm
        self.beta = beta
        self.beta_w = beta_w
        self.Tpmp = Tpmp
        self.Eb_mean = Eb_mean
        self.Eb_std = Eb_std
        self.dw = dw
        self.df = df
        self.Eb_epsilon = Eb_epsilon

    def __call__(self, params_dict):
        u = params_dict["u"]
        x = self.X_opt_tensor + torch.matmul(self.L, u)

        return custom_energy.apply(
            x, self.V, self.gmm, self.beta, self.beta_w, self.Tpmp,
            self.Eb_mean, self.Eb_std, self.dw, self.df, self.Eb_epsilon
        )
    
class regular_potential:
    """
    Direct potential in x-space (no whitening/Hessian preconditioning).
    Used for exploratory runs where the posterior landscape is unknown.
    """
    def __init__(self, V, gmm, beta, beta_w, Tpmp, Eb_mean, Eb_std, dw, df, Eb_epsilon):
        self.V = V
        self.gmm = gmm
        self.beta = beta
        self.beta_w = beta_w
        self.Tpmp = Tpmp
        self.Eb_mean = Eb_mean
        self.Eb_std = Eb_std
        self.dw = dw
        self.df = df
        self.Eb_epsilon = Eb_epsilon

    def __call__(self, params_dict):
        x = params_dict["x"]
        return custom_energy.apply(
            x, self.V, self.gmm, self.beta, self.beta_w, self.Tpmp,
            self.Eb_mean, self.Eb_std, self.dw, self.df, self.Eb_epsilon
        )
