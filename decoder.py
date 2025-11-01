import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

"""
VAE Decoder

The VAE (Variational Autoencoder) decoder converts latent representations back into images.
It works in the latent space (8x smaller than image space) for efficiency during diffusion.

The decoder consists of:
1. Residual blocks: process features at different scales
2. Attention blocks: capture long-range spatial dependencies
3. Upsampling layers: increase spatial resolution

Why operate in latent space?
- Images are 512x512, latents are 64x64 (8x smaller)
- Processing in latent space is ~64x faster
- The VAE encoder/decoder learns a compressed but lossy representation
"""
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # GroupNorm: normalizes across channel groups
        # 32 groups: divides channels into 32 groups for normalization
        # Better than BatchNorm for variable batch sizes and small batches
        self.groupnorm = nn.GroupNorm(32, channels)
        # Self-attention with 1 head: captures spatial relationships
        # Single head is sufficient since we're processing spatial features, not sequences
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        # x: (Batch_Size, Features, Height, Width)
        # Save input for residual connection
        residue = x 

        # Normalize features
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        
        # Reshape spatial dimensions into sequence for attention
        # Treat each spatial position as a token in the sequence
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # Transpose to sequence format: (seq_len, features)
        # Each pixel becomes a feature vector, sequence length is Height * Width
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        # Perform self-attention WITHOUT mask (unlike CLIP which uses causal mask)
        # Spatial attention: each position can attend to all other positions
        # This helps capture long-range dependencies in the image (e.g., symmetry, global structure)
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)
        
        # Transpose back to feature-first format
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # Reshape back to spatial format
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))
        
        # Residual connection: allows information to bypass attention
        # Helps with gradient flow and training stability
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width) 
        x += residue

        # (Batch_Size, Features, Height, Width)
        return x 

"""
VAE Residual Block

Standard residual block for feature processing. Uses GroupNorm and SiLU activation.
Residual connections help with gradient flow in deep networks.

Why SiLU (Swish) activation?
- Smooth and differentiable everywhere (unlike ReLU)
- Non-zero gradients for negative values (helps with training)
- Often performs better than ReLU in practice
"""
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # First normalization and convolution
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        # 3x3 conv with padding=1 preserves spatial dimensions
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Second normalization and convolution
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection projection
        # If channels match, use identity (no computation needed)
        # Otherwise, use 1x1 conv to match channel dimensions
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            # 1x1 conv only changes channels, not spatial dimensions
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        # x: (Batch_Size, In_Channels, Height, Width)
        # Save input for residual connection
        residue = x

        # Normalize then activate: pre-norm style
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = self.groupnorm_1(x)
        
        # SiLU activation: x * sigmoid(x)
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = F.silu(x)
        
        # First convolution: may change channel dimension
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_1(x)
        
        # Normalize again
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.groupnorm_2(x)
        
        # Activate again
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = F.silu(x)
        
        # Second convolution: refines features
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_2(x)
        
        # Residual connection: add original (possibly projected) input
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return x + self.residual_layer(residue)

"""
VAE Decoder Architecture

The decoder progressively upsamples from latent space (64x64) to image space (512x512).
Architecture pattern:
1. Initial projection from 4 latent channels
2. Processing blocks at 64x64 (Height/8, Width/8)
3. Upsample to 128x128 (Height/4, Width/4) and process
4. Upsample to 256x256 (Height/2, Width/2) and process
5. Upsample to 512x512 (Height, Width) and process
6. Final projection to 3 RGB channels

The constant 0.18215 is a scaling factor from the encoder that must be removed.
"""
class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # Initial projection: prepare latent for processing
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(512, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 256, Height, Width)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            
            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(256, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.GroupNorm(32, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.SiLU(), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        
        # Remove the scaling added by the Encoder
        # The encoder scales by 0.18215 to stabilize training
        # We must undo this scaling to restore proper pixel values
        # This constant is from the original Stable Diffusion implementation
        x /= 0.18215

        # Process through all decoder layers sequentially
        # Each layer transforms features and/or upsamples spatial dimensions
        for module in self:
            x = module(x)

        # Output: RGB image in latent space representation
        # (Batch_Size, 3, Height, Width)
        return x