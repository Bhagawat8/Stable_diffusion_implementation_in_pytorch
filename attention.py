import torch
from torch import nn
from torch.nn import functional as F
import math

"""
Multi-Head Self-Attention Mechanism

This module implements the self-attention mechanism used in transformers.
Self-attention allows each position in a sequence to attend to all positions 
in the same sequence, enabling the model to capture long-range dependencies 
and contextual relationships.

Why self-attention?
- It allows parallel computation (unlike RNNs which are sequential)
- It captures relationships between all pairs of positions in one pass
- It enables the model to understand which parts of the input are relevant to each other
"""
class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        # Why combine? It's more efficient to compute all three projections in a single matrix multiplication
        # Q, K, V projections are concatenated: [Q|K|V] = X * W_in_proj
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix (output projection)
        # Projects the concatenated attention heads back to the original embedding dimension
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        # Dimension per head: we split the embedding dimension across heads
        # This allows the model to attend to different representation subspaces simultaneously
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: # (Batch_Size, Seq_Len, Dim)

        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape 
        
        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape 

        # Prepare shape for multi-head attention: split embedding dimension across heads
        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # Compute Q, K, V in one pass for efficiency
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # Reshape and transpose for multi-head attention computation
        # Each head will process a different subspace of the embeddings
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute attention scores: Q @ K^T
        # Higher scores indicate stronger relationships between positions
        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Causal mask ensures positions can only attend to previous positions
            # This is used in autoregressive models (like language models) to prevent 
            # information leakage from future positions during training
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf so softmax will make these weights 0
            weight.masked_fill_(mask, -torch.inf) 
        
        # Scale by sqrt(d_k) to prevent softmax saturation
        # This scaling is crucial: without it, dot products can become very large,
        # causing softmax to be too peaked and gradients to vanish
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head) 

        # Apply softmax to get attention probabilities (weights sum to 1)
        # This determines how much each position should attend to every other position
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1) 

        # Apply attention weights to values
        # Weighted sum of values based on attention scores
        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # Concatenate heads back together
        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2) 

        # Reshape to original sequence format
        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape) 

        # Final output projection
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output) 
        
        # (Batch_Size, Seq_Len, Dim)
        return output

"""
Cross-Attention Mechanism

Unlike self-attention, cross-attention allows the query (Q) to attend to a different
sequence (keys and values). In Stable Diffusion, this is used to condition the 
latent image features on the text prompt embeddings.

Why cross-attention?
- Enables the image features (queries) to attend to text features (keys/values)
- This is how the text prompt guides the image generation process
- The model learns to focus on relevant parts of the text prompt at different spatial locations
"""
class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Query comes from the latent features (image)
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        # Key and Value come from the context (text embeddings)
        # Note: they can have different dimensions, so we project them to match
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # Output projection
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x (latent): image features that will query the text context
        # (Batch_Size, Seq_Len_Q, Dim_Q) - spatial positions in latent space
        # y (context): text embeddings from CLIP that provide conditioning information
        # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768) - 77 is max token length

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Prepare shape for multi-head attention
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # Project inputs to query, key, and value
        # Q comes from image features - "what am I looking for?"
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # K and V come from text context - "what information do I have?"
        # Project to same dimension as Q for compatibility
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # Reshape for multi-head attention computation
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2) 
        
        # Compute attention scores: how relevant is each text token to each spatial position?
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # Scale by sqrt(d_k) to prevent softmax saturation
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)
        
        # Normalize to get attention probabilities
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)
        
        # Apply attention weights to values
        # Each spatial position now contains a weighted combination of text information
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v
        
        # Concatenate heads
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        
        # Reshape to original format
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # Final projection
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # Output: image features enriched with relevant text information
        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output