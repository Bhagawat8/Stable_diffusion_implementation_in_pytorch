# Stable Diffusion Inference Guide

This my PyTorch implementation of Stable Diffusion works for inference (image generation). We'll look into **what** each component does, **why** it's designed that way, and **how** they work together to generate images from text prompts.

## Table of Contents

1. [Overview: What is Stable Diffusion?](#overview)
2. [Architecture Overview](#architecture-overview)
3. [Inference Pipeline: Step-by-Step](#inference-pipeline)
4. [Component Deep Dive](#component-deep-dive)
5. [Loading Pretrained Weights](#loading-pretrained-weights)
6. [Code Navigation Guide](#code-navigation-guide)

---

## Overview: What is Stable Diffusion?

**What**: Stable Diffusion is a text-to-image generation model that creates images from text descriptions.

**Why**: Unlike traditional GANs, diffusion models are more stable to train and produce higher quality, more diverse images.

**How**: It works in a compressed "latent space" (8x smaller than pixel space) for efficiency. The process:
1. Encodes text prompt into embeddings (CLIP)
2. Starts with random noise
3. Iteratively removes noise guided by the text prompt (UNET)
4. Decodes the clean latents back to an image (VAE)

The key insight: **Instead of generating pixels directly, we generate in latent space and decode, making it 64x faster.**

---

## Architecture Overview

Stable Diffusion consists of 4 main components:

```
Text Prompt → [CLIP] → Text Embeddings
                              ↓
Random Noise → [UNET + DDPM] → Clean Latents → [VAE Decoder] → Image
                              ↑
                      Text Embeddings (conditioning)
```

### Components:

1. **CLIP (Text Encoder)**: Converts text prompt into semantic embeddings
   - **Why**: We need a way to represent text in a format the image model understands
   - **What**: Transformer encoder that outputs 77×768 dimensional embeddings
   - **File**: `sd/clip.py`

2. **VAE Encoder** (for img2img): Compresses images to latent space
   - **Why**: Diffusion in latent space is much faster than pixel space
   - **What**: Encodes 512×512×3 images → 64×64×4 latents
   - **File**: `sd/encoder.py`

3. **UNET (Diffusion Model)**: Predicts noise to remove at each step
   - **Why**: This is the "brain" that learns to generate images
   - **What**: U-shaped network with attention layers, conditioned on text and timestep
   - **File**: `sd/diffusion.py`

4. **VAE Decoder**: Converts latents back to images
   - **Why**: We need to convert latent representations to actual pixels
   - **What**: Decodes 64×64×4 latents → 512×512×3 images
   - **File**: `sd/decoder.py`

5. **DDPM Sampler**: Implements the denoising algorithm
   - **Why**: Defines how to iteratively remove noise over multiple steps
   - **What**: Manages timesteps, noise schedules, and denoising formulas
   - **File**: `sd/ddpm.py`

---

## Inference Pipeline: Step-by-Step

The complete inference process is orchestrated in `sd/pipeline.py`. Let's walk through what happens when you generate an image:

### Step 1: Encode Text Prompt

**What happens**: Your text prompt ("a cat on a mat") is converted into numerical embeddings.

**Why**: Neural networks work with numbers, not text. CLIP embeddings capture semantic meaning.

**How it works**:

```python
# In pipeline.py, generate() function
cond_tokens = tokenizer.batch_encode_plus([prompt], max_length=77).input_ids
cond_context = clip(cond_tokens)  # Shape: (1, 77, 768)
```

**Code location**: 
- Tokenization: `pipeline.py` lines 91-93
- CLIP encoding: `pipeline.py` line 98
- CLIP implementation: `sd/clip.py`

**Details**:
- Text is tokenized into subword tokens (max 77 tokens for CLIP)
- CLIP processes tokens through 12 transformer layers
- Output is a 77×768 tensor where each of the 77 positions has a 768-dim embedding
- **Why 77?** CLIP was trained with this sequence length
- **Why 768?** Standard embedding dimension for CLIP-Base model

### Step 2: Initialize Latents

**What happens**: We create the starting point for diffusion - either pure noise (text2img) or encoded image (img2img).

**Why**: Diffusion needs to start from noise and iteratively denoise. For img2img, we encode the input first.

**How it works**:

**Text-to-Image**:
```python
# Pure random noise in latent space
latents = torch.randn((1, 4, 64, 64))  # Shape: (batch, channels, height, width)
```

**Image-to-Image**:
```python
# Encode input image to latent space
latents = encoder(input_image_tensor, encoder_noise)  # Shape: (1, 4, 64, 64)
# Add noise based on strength parameter
latents = sampler.add_noise(latents, sampler.timesteps[0])
```

**Code location**:
- Text2img: `pipeline.py` line 180
- Img2img: `pipeline.py` lines 166, 173
- VAE encoder: `sd/encoder.py`

**Details**:
- **Why latent space (64×64×4) instead of pixel space (512×512×3)?**
  - 8×8 = 64× smaller spatial dimensions
  - 64× fewer pixels to process = 64× faster!
  - The VAE learns a compressed representation that preserves visual information
- **Why 4 channels?** The VAE encoder learns to compress RGB (3 channels) into a more efficient 4-channel representation
- **Strength parameter**: Controls how much noise to add (0.0 = preserve image, 1.0 = full generation)

### Step 3: Iterative Denoising (The Core Process)

**What happens**: The UNET repeatedly predicts and removes noise, guided by the text prompt.

**Why**: We can't generate an image in one step. Instead, we gradually refine from noise to image, with text guidance at each step.

**How it works**:

```python
for timestep in sampler.timesteps:  # e.g., [999, 980, 960, ..., 20, 0]
    # Create time embedding
    time_emb = get_time_embedding(timestep)  # Shape: (1, 320)
    
    # Predict noise using UNET
    predicted_noise = diffusion(latents, text_embeddings, time_emb)
    
    # Remove predicted noise (one step closer to clean image)
    latents = sampler.step(timestep, latents, predicted_noise)
```

**Code location**:
- Main loop: `pipeline.py` lines 187-217
- UNET forward: `sd/diffusion.py` `Diffusion.forward()`
- DDPM step: `sd/ddpm.py` `DDPMSampler.step()`

**Details**:
- **Why iterative?** 
  - Single-step generation is too difficult for the model
  - Gradual refinement produces better quality
  - Typical: 20-50 steps (can use fewer for speed, more for quality)

- **Why time embedding?**
  - Different noise levels require different denoising strategies
  - Early steps (high noise): focus on overall structure
  - Late steps (low noise): focus on fine details
  - Time embedding tells UNET which phase it's in

- **How UNET uses text (Cross-Attention)**:
  - Spatial features (queries) attend to text tokens (keys/values)
  - Each spatial position learns which text tokens are relevant
  - This is how "cat" in the prompt creates a cat in the image
  - See `sd/diffusion.py` `UNET_AttentionBlock` for implementation

### Step 4: Classifier-Free Guidance (CFG)

**What happens**: We generate two predictions - one with the prompt, one without - and push toward the prompt version.

**Why**: Improves adherence to the prompt. Without CFG, the model might ignore the text.

**How it works**:

```python
# Generate both conditional and unconditional predictions
model_input = latents.repeat(2, 1, 1, 1)  # Duplicate for batch
model_output = diffusion(model_input, context, time_emb)  # Shape: (2, 4, 64, 64)

# Split predictions
output_cond, output_uncond = model_output.chunk(2)

# Guidance formula: push toward conditional, away from unconditional
final_output = uncond + cfg_scale * (cond - uncond)
```

**Code location**: `pipeline.py` lines 197-213

**Details**:
- **Why this works?**
  - `cond - uncond` is the "direction" of the prompt
  - `cfg_scale` controls how strongly to follow that direction
  - Typical `cfg_scale = 7.5`: good balance of prompt adherence vs. quality
  - Higher values (15+) = very prompt-focused, but may reduce quality
  - Lower values (1-5) = more creative, but may ignore prompt

- **Unconditional prompt**: Usually empty string `""` - represents "no prompt"

### Step 5: Decode Latents to Image

**What happens**: Clean latents are converted back to a full-resolution image.

**Why**: We worked in latent space for speed, but need pixels for the final output.

**How it works**:

```python
# Decode from latent space (64×64×4) to pixel space (512×512×3)
image = decoder(latents)  # Shape: (1, 3, 512, 512)

# Post-process: normalize from [-1, 1] to [0, 255]
image = rescale(image, (-1, 1), (0, 255))
```

**Code location**: `pipeline.py` lines 222-238, `sd/decoder.py`

**Details**:
- **Why decoder needed?**
  - VAE learns a compressed representation during training
  - Decoder reconstructs pixels from this compressed representation
  - The 0.18215 scaling factor (from encoder) must be removed first
  - See `sd/decoder.py` line 246

- **Architecture**:
  - Progressive upsampling: 64×64 → 128×128 → 256×256 → 512×512
  - Uses residual blocks and attention for quality
  - Final layer: 128 channels → 3 RGB channels

---

## Component Deep Dive

### CLIP Text Encoder (`sd/clip.py`)

**What it does**: Converts text tokens into semantic embeddings that guide image generation.

**Why this design**:
- CLIP was trained on image-text pairs, so it understands visual-semantic relationships
- Transformer architecture captures long-range dependencies in text
- Position embeddings encode token order (critical: "cat on mat" ≠ "mat on cat")

**How it works**:

```python
# Architecture (12 layers):
tokens → Embedding → [CLIPLayer × 12] → Layernorm → embeddings
```

**Key components**:
- **Token Embedding** (`CLIPEmbedding`): Maps token IDs to vectors
- **Position Embedding**: Learnable position encodings
- **Transformer Layers** (`CLIPLayer`): Self-attention + feedforward
  - Self-attention: tokens attend to each other ("cat" understands "mat")
  - Feedforward: Non-linear transformations
  - Pre-norm architecture: normalization before attention/FFN

**Code references**:
- Token embedding: `sd/clip.py` lines 19-42
- Transformer layer: `sd/clip.py` lines 61-126
- Main CLIP model: `sd/clip.py` lines 141-178

### UNET Diffusion Model (`sd/diffusion.py`)

**What it does**: Predicts noise to remove at each diffusion timestep, conditioned on text and time.

**Why this design**:
- U-shaped architecture preserves fine details via skip connections
- Cross-attention injects text information at multiple scales
- Time conditioning adapts denoising strategy to noise level

**How it works**:

```python
# Forward pass:
latent (4,64,64) → UNET → predicted_noise (4,64,64)
         ↑                    ↑
    text_emb + time_emb
```

**Architecture**:
```
Encoder Path (Downsampling):
- Input: 64×64×4
- Downsample to 32×32, 16×16, 8×8, 4×4
- Each level: ResidualBlock + AttentionBlock
- Skip connections saved

Bottleneck:
- Process at 4×4 resolution (lowest)
- Captures global structure

Decoder Path (Upsampling):
- Upsample from 4×4 → 8×8 → 16×16 → 32×32 → 64×64
- Each level: Concatenate skip connection + Upsample
- ResidualBlocks and AttentionBlocks refine features
```

**Key components**:

1. **Time Embedding** (`TimeEmbedding`):
   - Converts timestep → 320-dim → 1280-dim embedding
   - Tells model how much noise is present
   - See `sd/diffusion.py` lines 35-58

2. **Residual Blocks** (`UNET_ResidualBlock`):
   - Process features with time conditioning
   - Time embedding modulates feature scale
   - See `sd/diffusion.py` lines 72-141

3. **Attention Blocks** (`UNET_AttentionBlock`):
   - **Self-attention**: Spatial relationships (pixel A relates to pixel B)
   - **Cross-attention**: Text conditioning (pixel attends to relevant text tokens)
   - **Feedforward**: GeGLU activation for non-linearity
   - See `sd/diffusion.py` lines 143-273

4. **Skip Connections**:
   - Encoder features concatenated to decoder at same resolution
   - Preserves fine details lost in downsampling
   - See `sd/diffusion.py` lines 298-301

**Code references**:
- Main UNET: `sd/diffusion.py` lines 196-303
- Diffusion wrapper: `sd/diffusion.py` lines 427-460

### DDPM Sampler (`sd/ddpm.py`)

**What it does**: Implements the mathematical formulas for the diffusion process.

**Why this design**:
- Defines noise schedule (how much noise at each step)
- Implements denoising formula from DDPM paper
- Manages timestep selection for inference

**How it works**:

**Noise Schedule**:
```python
# Beta schedule: linearly interpolated in sqrt space
betas = linspace(0.00085^0.5, 0.0120^0.5, 1000)^2
alphas = 1 - betas
alphas_cumprod = cumulative_product(alphas)
```

**Why this schedule?**
- Starts with small noise (preserves signal early)
- Gradually increases noise (more destruction later)
- Linear in sqrt space = smooth transitions

**Denoising Step**:
```python
# From ddpm.py, step() function:

# 1. Predict clean image x_0 from noisy x_t
x_0 = (x_t - sqrt(1-alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)

# 2. Predict previous step x_{t-1} from x_0 and x_t
x_{t-1} = coeff_1 * x_0 + coeff_2 * x_t + variance

# 3. Add stochastic noise (for t > 0)
if t > 0:
    x_{t-1} += sqrt(variance) * random_noise
```

**Why stochastic noise?**
- Adds randomness for diversity (same prompt → different images)
- At final step (t=0), no noise added (deterministic)

**Code references**:
- Initialization: `sd/ddpm.py` lines 24-49
- Denoising step: `sd/ddpm.py` lines 106-154
- Noise addition: `sd/ddpm.py` lines 156-196

### VAE Components (`sd/encoder.py`, `sd/decoder.py`)

**What they do**: Compress images to latent space and decompress back.

**Why this design**:
- Latent space is 64× smaller = 64× faster diffusion
- VAE learns efficient compression preserving visual information
- Variational approach adds stochasticity (multiple latents per image)

**Encoder** (image → latent):
```
Image (512×512×3) 
→ Downsample: 256×256, 128×128, 64×64
→ Residual blocks + Attention
→ Output: mean & log-variance (8 channels)
→ Sample: mean + sqrt(variance) * noise (4 channels)
→ Scale by 0.18215
```

**Decoder** (latent → image):
```
Latent (64×64×4)
→ Remove scaling (÷ 0.18215)
→ Upsample: 64×64 → 128×128 → 256×256 → 512×512
→ Residual blocks + Attention
→ Output: Image (512×512×3)
```

**Why variational (mean + variance)?**
- Allows multiple latents per image (diversity)
- Prevents posterior collapse
- Reparameterization trick makes it differentiable

**Code references**:
- Encoder: `sd/encoder.py` lines 22-137
- Decoder: `sd/decoder.py` lines 155-255

---

## Loading Pretrained Weights

**What**: Convert and load weights from official Stable Diffusion checkpoints.

**Why**: We implemented the architecture from scratch, but need to use pre-trained weights (training SD from scratch requires massive compute).

**How it works**:

### 1. Download Checkpoint

Download the checkpoint file (e.g., `v1-5-pruned-emaonly.ckpt`) from HuggingFace:
- Standard v1.5: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- Fine-tuned models also work (InkPunk, etc.)

Save to `data/` folder (or your preferred location).

### 2. Weight Conversion

**Problem**: Original checkpoint uses different layer names than our implementation.

**Solution**: `model_converter.py` maps thousands of weight tensors:

```python
# Original format:
"model.diffusion_model.input_blocks.0.0.weight"

# Our format:
"diffusion.unet.encoders.0.0.weight"
```

**Why needed?**
- Original SD uses different architecture naming
- Our implementation organizes layers differently
- Converter translates between formats

**Code location**: `sd/model_converter.py`
- Large file (~1000+ lines) with weight mappings
- Maps encoder, decoder, diffusion, and CLIP weights

### 3. Loading Process

```python
from model_loader import preload_models_from_standard_weights

models = preload_models_from_standard_weights(
    ckpt_path="data/v1-5-pruned-emaonly.ckpt",
    device="cuda"
)

# Returns:
# {
#     'clip': CLIP model,
#     'encoder': VAE encoder,
#     'decoder': VAE decoder,
#     'diffusion': UNET model
# }
```

**What happens internally**:
1. Load checkpoint with `torch.load()`
2. Convert weight names via `model_converter.py`
3. Initialize empty models with our architecture
4. Load converted weights into models
5. Models are ready for inference!

**Code location**: `sd/model_loader.py` lines 19-61

---

## Code Navigation Guide

Use this map to navigate the codebase:

### Main Entry Point
- **`sd/pipeline.py`**: `generate()` function - orchestrates entire process

### Text Processing
- **`sd/clip.py`**: CLIP text encoder
  - `CLIPEmbedding`: Token + position embeddings
  - `CLIPLayer`: Transformer layer (attention + feedforward)
  - `CLIP`: Main model (12 layers)

### Image Compression/Decompression
- **`sd/encoder.py`**: VAE encoder (image → latent)
- **`sd/decoder.py`**: VAE decoder (latent → image)
  - `VAE_ResidualBlock`: Feature processing
  - `VAE_AttentionBlock`: Spatial attention

### Diffusion Process
- **`sd/diffusion.py`**: UNET model that predicts noise
  - `TimeEmbedding`: Timestep encoding
  - `UNET_ResidualBlock`: Feature processing with time conditioning
  - `UNET_AttentionBlock`: Self-attention + cross-attention (text guidance)
  - `UNET`: Main U-shaped architecture
  - `Diffusion`: Wrapper combining UNET + time embedding

- **`sd/ddpm.py`**: DDPM sampling algorithm
  - `DDPMSampler`: Manages timesteps, noise schedule, denoising

### Attention Mechanisms
- **`sd/attention.py`**: Core attention implementations
  - `SelfAttention`: Tokens attend to tokens (used in CLIP, UNET)
  - `CrossAttention`: Image features attend to text (used in UNET)

### Model Loading
- **`sd/model_converter.py`**: Weight format conversion
- **`sd/model_loader.py`**: High-level loading function

---

## Common Questions

### Q: Why latent space instead of pixel space?

**A**: Processing 64×64×4 latents is 64× faster than 512×512×3 pixels. The VAE learns efficient compression, so we get speed without major quality loss.

### Q: Why multiple denoising steps?

**A**: Single-step generation is too difficult. Gradual refinement (20-50 steps) produces better quality. You can use fewer steps for speed (e.g., 20) or more for quality (e.g., 50).

### Q: What is classifier-free guidance?

**A**: Technique to improve prompt adherence. We generate with and without the prompt, then push toward the prompt version. Higher `cfg_scale` = stronger prompt adherence.

### Q: Why time embeddings?

**A**: Different noise levels require different denoising strategies. Early steps need coarse structure; late steps need fine details. Time embedding tells the model which phase it's in.

### Q: How does text guide generation?

**A**: Cross-attention in UNET. Each spatial position (pixel location) attends to relevant text tokens. The model learns which words affect which parts of the image.

### Q: Can I use fine-tuned models?

**A**: Yes! Any Stable Diffusion v1.5 checkpoint works. Download the `.ckpt` file and load it the same way. Fine-tuned models change the style/domain (e.g., anime, illustrations) but use the same architecture.

---

## Summary

**What** Stable Diffusion does:
- Generates images from text prompts
- Works in compressed latent space for efficiency
- Uses iterative denoising (diffusion process)

**Why** this design:
- Latent space = 64× faster than pixels
- Iterative denoising = better quality than one-shot
- Text conditioning via cross-attention = precise control

**How** it works:
1. CLIP encodes text → embeddings
2. Start from noise (or encoded image for img2img)
3. UNET iteratively removes noise, guided by text embeddings
4. VAE decoder converts clean latents → final image

The beauty of this implementation: **You understand every line of code.** Unlike using a black-box library, you can see exactly how each component contributes to the final result.

---

For training information, see the original Stable Diffusion paper and training code. This implementation focuses on inference with pretrained weights.
