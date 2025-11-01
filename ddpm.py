import torch
import numpy as np

"""
DDPM Sampler

Implements the Denoising Diffusion Probabilistic Model (DDPM) sampling algorithm.
This is the core of the diffusion process that generates images by iteratively removing noise.

How DDPM works:
1. Training: Gradually add noise to images over T steps, learn to predict the noise
2. Sampling: Start with pure noise, iteratively remove predicted noise over T steps

The diffusion process uses a fixed noise schedule (betas) that determines how much noise
to add at each step. The sampler uses this schedule in reverse to generate images.

Key concepts:
- Forward process: q(x_t | x_{t-1}) - adds noise
- Reverse process: p(x_{t-1} | x_t) - removes noise (what we learn)
- Noise schedule: controls how much noise is added at each timestep
"""
class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        # Noise schedule parameters from Stable Diffusion config
        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        
        # Beta schedule: linearly interpolated in sqrt space then squared
        # This creates a schedule that starts slow and accelerates
        # Linear in sqrt space ensures smoother transitions
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        
        # Alpha: amount of signal to keep (1 - beta = amount of noise)
        self.alphas = 1.0 - self.betas
        
        # Cumulative product of alphas: alpha_bar_t = product of alphas up to step t
        # This tells us how much signal remains after t steps of noise addition
        # Used in the closed-form formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # Random number generator for sampling noise
        self.generator = generator

        self.num_train_timesteps = num_training_steps
        # Timesteps for inference: reversed order (start from high noise, go to clean)
        # We reverse because sampling starts from noise and removes it
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        # During inference, we don't need all 1000 training steps
        # We can use fewer steps (typically 20-50) by sampling a subset
        # This speeds up generation significantly
        self.num_inference_steps = num_inference_steps
        
        # Compute step ratio: how many training steps per inference step
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        
        # Sample evenly spaced timesteps from training schedule
        # Round to nearest integer and reverse (start from high noise)
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        Compute the variance of the posterior distribution p(x_{t-1} | x_t).
        
        This variance determines how much stochasticity to add during sampling.
        The formula comes from the DDPM paper (formula 7).
        """
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # Compute predicted variance βt
        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # This is the variance of the posterior q(x_{t-1} | x_t, x_0)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # Clamp to prevent numerical issues
        # We always take operations that use variance (like sqrt), so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
    def set_strength(self, strength=1):
        """
        Set how much noise to add to the input image (for img2img).
        
        This controls the balance between the input image and generated content:
        - strength = 1.0: Start from pure noise (full generation, ignore input)
        - strength = 0.5: Start from moderately noisy input (hybrid generation)
        - strength = 0.1: Start from slightly noisy input (mostly preserves input)
        
        More noise (strength ~ 1) means that the output will be further from the input image.
        Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # Compute which timestep to start from
        # Higher strength -> skip more early steps -> start from higher noise
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        # Use only timesteps from start_step onwards
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """
        Perform one denoising step: predict x_{t-1} from x_t
        
        This implements the reverse diffusion process:
        1. Predict the original image x_0 from current noisy latents x_t
        2. Predict the previous timestep x_{t-1} using the predicted x_0
        3. Add some noise back (stochastic sampling) for t > 0
        
        This follows the DDPM sampling algorithm from the paper.
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. Compute alpha and beta values for current and previous timesteps
        # These control how we combine predictions and how much noise to add
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t  # Amount of noise at step t
        beta_prod_t_prev = 1 - alpha_prod_t_prev  # Amount of noise at step t-1
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev  # Signal ratio between steps
        current_beta_t = 1 - current_alpha_t  # Noise ratio between steps

        # 2. Predict the original (clean) image x_0 from noisy latents x_t
        # Formula: x_0 = (x_t - sqrt(1 - alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)
        # This is formula (15) from the DDPM paper
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 3. Compute coefficients for combining predicted x_0 and current x_t
        # These coefficients determine how much to trust the prediction vs current state
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 4. Predict previous sample x_{t-1}
        # This is the mean of the distribution p(x_{t-1} | x_t)
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 5. Add stochastic noise (for t > 0)
        # At the last step (t=0), we don't add noise (deterministic)
        # For earlier steps, we sample from the predicted distribution
        variance = 0
        if t > 0:
            device = model_output.device
            # Sample random noise
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            # This is the variance of the posterior distribution
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # Sample from N(mu, sigma) where mu = pred_prev_sample and sigma = variance
        # X ~ N(mu, sigma) can be obtained by X = mu + sigma * N(0, 1)
        # The variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Add noise to original samples according to the forward diffusion process.
        
        This implements the forward process: q(x_t | x_0)
        Used for training (to create noisy examples) and img2img (to add noise to input image).
        
        Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        where epsilon ~ N(0, I) is random noise
        """
        # Move schedule tensors to correct device and dtype
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # Compute coefficient for original signal
        # sqrt(alpha_bar_t) determines how much of the original signal to keep
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # Expand dimensions to match sample shape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # Compute coefficient for noise
        # sqrt(1 - alpha_bar_t) determines how much noise to add
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # Expand dimensions to match sample shape for broadcasting
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from q(x_t | x_0) as in equation (4) of https://arxiv.org/pdf/2006.11239.pdf
        # This is the closed-form formula for the forward diffusion process
        # Because N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # here mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

        

    