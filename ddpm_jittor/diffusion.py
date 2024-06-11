import jittor as jit
from jittor import nn
import numpy as np
from tqdm import tqdm

def get_timestep_embeddings(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = jit.log(10000) / (half_dim - 1)
    emb = jit.exp(jit.misc.arange(0, half_dim, dtype=jit.float32) * -emb)
    emb = jit.type_as(timesteps, emb)[:, None] * emb[None, :]
    emb = jit.concat([jit.sin(emb), jit.cos(emb)], dim=1)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class GaussianDiffusion:
    def __init__(self, betas):
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        assert alphas_cumprod_prev.shape == (timesteps,)
        
        self.betas = jit.Var(betas)
        self.alphas_cumprod = jit.float32(alphas_cumprod)
        self.alphas_cumprod_prev = jit.float32(alphas_cumprod_prev)
        
        self.sqrt_alphas_cumprod = jit.float32(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = jit.float32(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = jit.float32(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = jit.float32(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = jit.float32(np.sqrt(1. / alphas_cumprod - 1))
        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = jit.float32(posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = jit.float32(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = jit.float32(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = jit.float32((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = a[t]
        assert out.shape == [bs]
        return jit.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))
    
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1. - self.sqrt_alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = jit.randn_like(x_start)
        return (self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
                self._extract(1. - self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
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
        assert t.shape == [B]

        if noise is None:
            noise = jit.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = denoise_fn(x_noisy, t)
        assert x_noisy.shape == x_start.shape
        assert x_recon.shape[:3] == [B, H, W] and len(x_recon.shape) == 4

        losses = nn.mse_loss(x_recon, noise)
        return losses
    
    def p_mean_variance(self, denoise_fn, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_fn(x, t))
        if clip_denoised:
            x_recon = jit.safe_clip(x_recon, -1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        assert model_mean.shape == x_recon.shape == x.shape
        assert posterior_variance.shape == posterior_log_variance.shape == [x.shape[0], 1, 1, 1]
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, denoise_fn, x, t, clip_denoised=True):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, x=x, t=t, clip_denoised=clip_denoised)
        noise = jit.randn_like(x)
        assert noise.shape == x.shape
        # no noise when t == 0
        nonzero_mask = jit.reshape(1 - jit.type_as(jit.equal(t, 0), x), [x.shape[0]] + [1] * (len(x.shape) - 1))
        return model_mean + nonzero_mask * jit.exp(0.5 * model_log_variance) * noise
    
    def p_sample_loop(self, denoise_fn, shape):
        """
        Generate samples
        """
        # i = jit.int32(self.num_timesteps - 1)
        img = jit.randn(*shape)
        for i in tqdm(range(self.num_timesteps - 1, -1, -1)):
            img = self.p_sample(denoise_fn=denoise_fn, x=img, t=jit.array([i]*shape[0]).reshape(shape[0],))
        assert img.shape == shape
        return img
    
    def p_sample_loop_with_trajectory(self, denoise_fn, shape, step):
        i = jit.int32(self.num_timesteps - 1)
        images = []
        img = jit.randn(*shape)
        while i >= 0:
            img = self.p_sample(denoise_fn=denoise_fn, x=img, t=jit.array([i]*shape[0]).reshape(shape[0],))
            if i % step == 0:
                images.append(img.numpy())
            i = i - 1
        assert img.shape == shape
        return images