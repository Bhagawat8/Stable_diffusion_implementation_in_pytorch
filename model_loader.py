"""
Model Loader

This module loads pre-trained Stable Diffusion weights and initializes the models.
It uses the model_converter to translate weights from the original checkpoint format
to our custom architecture.

The original Stable Diffusion uses different layer naming conventions and sometimes
different architectures. The converter maps those weights to match our implementation.
"""

from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    """
    Load all Stable Diffusion models from a checkpoint file.
    
    The checkpoint contains weights in the original Stable Diffusion format.
    The converter translates them to match our custom architecture.
    
    Args:
        ckpt_path: Path to the .ckpt or .safetensors file
        device: Device to load models on (e.g., "cuda" or "cpu")
    
    Returns:
        Dictionary containing all loaded models:
        - 'clip': Text encoder
        - 'encoder': VAE encoder (image -> latent)
        - 'decoder': VAE decoder (latent -> image)
        - 'diffusion': UNET diffusion model (denoising)
    """
    # Convert weights from original format to our architecture
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    # Initialize and load VAE encoder
    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    # Initialize and load VAE decoder
    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    # Initialize and load diffusion model (UNET)
    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    # Initialize and load CLIP text encoder
    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }