import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

"""
Stable Diffusion Generation Pipeline

This module orchestrates the complete image generation process:
1. Encode text prompt using CLIP
2. Initialize noise or encode input image (for img2img)
3. Iteratively denoise using the UNET diffusion model
4. Decode final latents back to image space using VAE decoder

The pipeline supports:
- Text-to-image generation (pure noise -> image)
- Image-to-image generation (input image -> modified image)
- Classifier-free guidance (CFG) for better prompt adherence
"""

# Standard image dimensions for Stable Diffusion v1
WIDTH = 512
HEIGHT = 512
# Latent space dimensions (8x smaller due to VAE compression)
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    """
    Generate an image from a text prompt using Stable Diffusion.
    
    Args:
        prompt: Text description of desired image
        uncond_prompt: Unconditional prompt for classifier-free guidance (usually empty)
        input_image: Optional PIL image for img2img mode
        strength: How much to modify input image (0-1, only for img2img)
        do_cfg: Whether to use classifier-free guidance (improves prompt adherence)
        cfg_scale: Strength of guidance (higher = more adherence to prompt)
        sampler_name: Which sampler to use (currently only "ddpm")
        n_inference_steps: Number of denoising steps (more = better quality, slower)
        models: Dictionary of loaded models (clip, encoder, decoder, diffusion)
        seed: Random seed for reproducibility
        device: Device to run inference on (e.g., "cuda")
        idle_device: Device to move models to when not in use (saves VRAM)
        tokenizer: CLIP tokenizer for converting text to tokens
    """
    with torch.no_grad():  # Disable gradients for inference
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        # Helper to move models to idle device when not in use (saves memory)
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        # Seed control allows reproducible generation
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()  # Random seed
        else:
            generator.manual_seed(seed)  # Fixed seed for reproducibility

        # Step 1: Encode text prompt using CLIP
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Classifier-Free Guidance (CFG): improves prompt adherence
            # Run both conditional (with prompt) and unconditional (without prompt) passes
            # Then combine: output = uncond + cfg_scale * (cond - uncond)
            # This pushes the output toward the conditional prediction
            
            # Encode conditional prompt (the actual text description)
            # Convert into a list of length Seq_Len=77 (CLIP's max sequence length)
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # Encode tokens to embeddings
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            
            # Encode unconditional prompt (usually empty or generic text)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # Encode unconditional tokens
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            
            # Concatenate for batch processing (batch size becomes 2x)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Without CFG: just encode the prompt
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # Encode to embeddings
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        # Move CLIP to idle device to free VRAM (not needed during diffusion)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # Step 2: Initialize latents (starting point for diffusion)
        if input_image:
            # Image-to-image mode: start from input image instead of pure noise
            encoder = models["encoder"]
            encoder.to(device)

            # Preprocess input image: resize, convert to tensor, normalize
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # Convert PIL image to numpy array
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # Convert to PyTorch tensor
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # Normalize from [0, 255] to [-1, 1] (VAE expects this range)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # Add batch dimension
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # Convert from HWC to CHW format (PyTorch convention)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Encode image to latent space
            # Sample noise for VAE reparameterization trick
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # Encode image to latents
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # Strength controls how much noise to add (higher = more modification)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            # Add noise at the starting timestep
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            # Move encoder to idle device
            to_idle(encoder)
        else:
            # Text-to-image mode: start from pure noise
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # Step 3: Iterative denoising (the diffusion process)
        diffusion = models["diffusion"]
        diffusion.to(device)

        # Iterate through timesteps (from high noise to low noise)
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # Create time embedding for this timestep
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # Prepare model input
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # For CFG: duplicate input for conditional and unconditional passes
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # Predict noise using UNET
            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # Apply classifier-free guidance
                # Split conditional and unconditional predictions
                output_cond, output_uncond = model_output.chunk(2)
                # Guidance formula: push toward conditional, away from unconditional
                # Higher cfg_scale = stronger adherence to prompt
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Denoise: remove predicted noise to get cleaner latents
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        # Move diffusion model to idle device
        to_idle(diffusion)

        # Step 4: Decode latents back to image space
        decoder = models["decoder"]
        decoder.to(device)
        # Decode from latent space to pixel space
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        # Post-process: denormalize and convert to uint8
        # Rescale from [-1, 1] to [0, 255]
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # Convert from CHW to HWC format
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        # Convert to numpy array for output
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]  # Return first (and only) image in batch
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    """
    Create sinusoidal position embeddings for timestep.
    
    This is similar to positional embeddings in transformers, but for time.
    Uses sin/cos functions to encode timestep information in a way that
    preserves relative time relationships.
    
    Why sinusoidal?
    - Smooth and continuous (important for timestep interpolation)
    - Encodes relative positions well (timestep 10 vs 20 is same distance as 990 vs 1000)
    - Allows model to learn periodic patterns in noise schedules
    """
    # Generate frequency bases for different dimensions
    # Shape: (160,)
    # 10000 is a base frequency, higher dimensions use higher frequencies
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Multiply timestep by frequencies (outer product)
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Concatenate sin and cos for each frequency (standard positional encoding)
    # Shape: (1, 160 * 2) = (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
