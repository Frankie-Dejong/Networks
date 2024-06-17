import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import math

def get_timestep_embeddings(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(0, half_dim, dtype=torch.float32) * -emb).to(timesteps.device)
    emb = timesteps.to(emb.dtype)[:, None] * emb[None, :]
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], dim=1)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float32) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float32)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float32)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float32)
    elif beta_schedule == 'cosine':
        betas = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class GaussianDiffusion:
    def __init__(self, betas, device='cuda', pred='noise', loss_type='mse'):
        self.device = device
        self.pred = pred
        self.loss_type = loss_type
        if pred not in ['noise', 'x0']:
            raise NotImplementedError()
        if loss_type not in ['mse', 'kl']:
            raise NotImplementedError()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        assert alphas_cumprod_prev.shape == (timesteps,)
        
        self.betas = torch.from_numpy(betas).to(device)
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).to(device)
        self.alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev).to(device)
        
        self.sqrt_alphas_cumprod = torch.from_numpy(np.sqrt(alphas_cumprod)).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(np.sqrt(1. - alphas_cumprod)).to(device)
        self.log_one_minus_alphas_cumprod = torch.from_numpy(np.log(1. - alphas_cumprod)).to(device)
        self.sqrt_recip_alphas_cumprod = torch.from_numpy(np.sqrt(1. / alphas_cumprod)).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.from_numpy(np.sqrt(1. / alphas_cumprod - 1)).to(device)
        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = torch.from_numpy(posterior_variance).to(device)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.from_numpy(np.log(np.maximum(posterior_variance, 1e-20))).to(device)
        self.posterior_mean_coef1 = torch.from_numpy(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(device)
        self.posterior_mean_coef2 = torch.from_numpy((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(device)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = a[t]
        assert out.shape[0] == bs
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))
    
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1. - self.sqrt_alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        return (self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
                self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_losses(self, denoise_fn, x_start, t, noise=None):
        """
        Training loss calculation
        """
        B, H, W, C = x_start.shape
        assert t.shape == (B,)

        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = denoise_fn(x_noisy, t)
        assert x_noisy.shape == x_start.shape == noise.shape
        if self.pred == 'noise':
            if self.loss_type == 'mse':
                losses = F.mse_loss(x_recon, noise)
            else:
                raise NotImplementedError()
        else:
            losses = 10 * nn.functional.mse_loss(x_recon, x_start)
        return losses
    
    def p_mean_variance(self, denoise_fn, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_fn(x, t))
        if clip_denoised:
            x_recon = torch.clip(x_recon, -1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        assert model_mean.shape == x_recon.shape == x.shape
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, denoise_fn, x, t, clip_denoised=True):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x, dtype=torch.float32).to(self.device)
        assert noise.shape == x.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - ((t == 0).to(x.dtype)), [x.shape[0]] + [1] * (len(x.shape) - 1))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    def p_sample_loop(self, denoise_fn, shape, img=None):
        """
        Generate samples
        """
        if img is None:
            img = torch.randn(*shape, dtype=torch.float32).to(self.device)
        else:
            img = self.q_sample(img, torch.tensor([self.num_timesteps - 1]*shape[0]).reshape(shape[0],).to(self.device))
        if self.pred == 'x0':
            for i in tqdm(range(self.num_timesteps - 1, -1, -1)):
                img = denoise_fn(img, torch.tensor([i]*shape[0]).reshape(shape[0],).to(self.device))
                if i > 1:
                    img = self.q_sample(img, torch.tensor([i-1]*shape[0]).reshape(shape[0],).to(self.device))
        else:
            for i in tqdm(range(self.num_timesteps - 1, -1, -1)):
                img = self.p_sample(denoise_fn=denoise_fn, x=img, t=torch.tensor([i]*shape[0]).reshape(shape[0],).to(self.device))
        return img
    
    def p_sample_loop_with_trajectory(self, denoise_fn, shape, step):
        i = self.num_timesteps - 1
        images = []
        img = torch.randn(*shape, dtype=torch.float32).to(self.device)
        if self.pred == 'x0':
            for i in tqdm(range(self.num_timesteps - 1, -1, -1)):
                img = denoise_fn(img, torch.tensor([i]*shape[0]).reshape(shape[0],).to(self.device))
                if i > 1:
                    img = self.q_sample(img, torch.tensor([i-1]*shape[0]).reshape(shape[0],).to(self.device))
                if i % step == 0:
                    images.append(img)
        else:
            for i in tqdm(range(self.num_timesteps - 1, -1, -1)):
                img = self.p_sample(denoise_fn=denoise_fn, x=img, t=torch.tensor([i]*shape[0]).reshape(shape[0],).to(self.device))
                if i % step == 0:
                    images.append(img)
        return images