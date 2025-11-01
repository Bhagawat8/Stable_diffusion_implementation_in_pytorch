import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

"""
CLIP Text Encoder

The CLIP (Contrastive Language-Image Pre-training) text encoder converts text prompts
into embeddings that guide the image generation. It uses a transformer architecture
similar to GPT, with self-attention layers to understand the semantic meaning of
the text prompt.

Why CLIP for text encoding?
- CLIP was trained on image-text pairs, so it understands visual-semantic relationships
- The embeddings capture semantic meaning that aligns well with image features
- This enables the model to generate images that match text descriptions
"""
class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        # Token embedding: maps each token ID to a dense vector
        # n_vocab: vocabulary size (49408 for CLIP tokenizer)
        # n_embd: embedding dimension (768)
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # A learnable weight matrix encodes the position information for each token
        # Position embeddings are crucial: they tell the model where each token appears
        # Without position info, "cat on mat" and "mat on cat" would be identical
        # n_token: maximum sequence length (77 for Stable Diffusion)
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        # Lookup token embeddings: convert token IDs to dense vectors
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) 
        x = self.token_embedding(tokens)
        # Add position embeddings: inform the model about token positions
        # This is element-wise addition, broadcasting position embeddings to all batches
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding
        
        return x

"""
CLIP Transformer Layer

Each layer applies:
1. Self-attention: allows tokens to attend to each other
2. Feed-forward network: applies non-linear transformation

The pre-norm architecture (normalization before attention/FFN) helps with:
- Training stability
- Gradient flow
- Preventing activation explosion

Why 4x expansion in feedforward?
- Common pattern in transformers: expands then contracts
- Provides capacity for complex transformations
- Non-linearity comes from activation function (QuickGELU)
"""
class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        # Pre-attention norm: normalizes before self-attention
        # This is pre-norm architecture (vs post-norm used in original Transformer)
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # Self attention: allows each token to attend to all tokens
        # This captures relationships between words in the prompt
        self.attention = SelfAttention(n_head, n_embd)
        # Pre-FNN norm: normalizes before feedforward network
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # Feedforward network: expands then contracts
        # Expansion provides capacity for complex transformations
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # (Batch_Size, Seq_Len, Dim)
        # Save input for residual connection
        residue = x
        
        ### SELF ATTENTION ###
        # Pre-norm: normalize before attention
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)
        
        # Self-attention with causal masking
        # Causal mask ensures tokens only attend to previous tokens (like in GPT)
        # This is important for training consistency, though during inference all tokens are available
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)
        
        # Residual connection: allows gradients to flow directly through
        # Helps with training deep networks by preventing vanishing gradients
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        ### FEEDFORWARD LAYER ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension
        # This provides capacity for non-linear transformations

        residue = x
        # Pre-norm: normalize before feedforward
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)
        
        # Expand dimension for transformation capacity
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear_1(x)
        
        # QuickGELU activation: faster approximation of GELU
        # GELU (Gaussian Error Linear Unit) is smoother than ReLU
        # Formula: x * sigmoid(1.702 * x), where 1.702 is a learned constant
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        
        # Contract back to original dimension
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)
        
        # Residual connection
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x

"""
CLIP Text Encoder Model

Stacks multiple CLIP layers (12 layers) to progressively build up semantic understanding
of the text prompt. The architecture follows the transformer encoder pattern.

Architecture details:
- 49408: CLIP vocabulary size
- 768: embedding dimension
- 77: maximum sequence length (tokens)
- 12: number of transformer layers
- 12: number of attention heads per layer
"""
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding layer: converts token IDs to embeddings
        # 49408: vocabulary size, 768: embedding dim, 77: max sequence length
        self.embedding = CLIPEmbedding(49408, 768, 77)

        # Stack 12 transformer layers
        # Each layer adds more semantic understanding
        # Deeper networks can capture more complex relationships
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        # Final layer normalization for stable outputs
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # Ensure tokens are long integers (required for embedding lookup)
        tokens = tokens.type(torch.long)
        
        # Convert tokens to embeddings (token + position)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder
        # Each layer refines the understanding of the text
        # Early layers: capture word-level and local phrase meanings
        # Later layers: capture sentence-level and global semantic meaning
        for layer in self.layers: 
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        # Final normalization for output stability
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)
        
        # Output: text embeddings ready to condition image generation
        return output