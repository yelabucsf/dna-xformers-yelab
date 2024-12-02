#!/usr/bin/env python

from typing import cast, Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from .mhsa import MultiHeadSelfAttention


class ClassicEncoderLayer(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 causal: bool = False):
        """
        A standard transformer encoder layer without rotary embeddings or performer attention.
        Uses pre-layer normalization architecture for better training stability.

        Args:
            dim (int): The input and output dimension of the layer
            num_heads (int): Number of attention heads
            ffn_dim (int): Dimension of the feed-forward network
            dropout (float, optional): Dropout rate. Defaults to 0.1
            causal (bool, optional): Whether to use causal attention. Defaults to False

        The layer consists of:
        1. Multi-head self-attention with pre-norm
        2. Residual connection and dropout
        3. Feed-forward network with pre-norm
        4. Residual connection and dropout
        """
        super().__init__()
        self.num_heads = num_heads
        # Classic attention without rotary embeddings or performer
        self.self_attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            performer=False,
            causal=causal,
            rotary_pos_emb=False
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_pad_mask: Optional[Tensor] = None,
                activation_checkpoint: bool = False) -> Tensor:
        """
        Forward pass of the encoder layer.

        Args:
            x (Tensor): Input tensor
            attn_mask (Optional[Tensor], optional): Attention mask. Defaults to None
            key_pad_mask (Optional[Tensor], optional): Key padding mask. Defaults to None
            activation_checkpoint (bool, optional): Whether to use activation checkpointing. Defaults to False

        Returns:
            Tensor: Output tensor of the same shape as input
        """
        res = x
        x = self.norm1(x)
        if activation_checkpoint:
            x = cast(
                Tensor,
                checkpoint(
                    self.self_attn,
                    use_reentrant=False,
                    x=x,
                    attn_mask=attn_mask,
                ),
            )
        else:
            x = self.self_attn(x, attn_mask=attn_mask, key_pad_mask=key_pad_mask)
        x = self.dropout(x)
        x = res + x

        res = x
        x = self.norm2(x)
        if activation_checkpoint:
            x = cast(
                Tensor,
                checkpoint(self.ffn, use_reentrant=False, input=x),
            )
        else:
            x = self.ffn(x)
        x = res + x
        return x


class ClassicEncoder(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 causal: bool = False):
        """
        A standard transformer encoder without rotary embeddings or performer attention.
        Uses pre-layer normalization architecture for better training stability.

        Args:
            dim (int): Model dimension
            num_heads (int): Number of attention heads
            ffn_dim (int): Feed-forward network dimension
            num_layers (int): Number of encoder layers
            dropout (float, optional): Dropout rate. Defaults to 0.1
            causal (bool, optional): Whether to use causal attention. Defaults to False

        The encoder consists of:
        1. A stack of ClassicEncoderLayers
        2. Final layer normalization
        """
        super().__init__()
        self.layers = nn.ModuleList([
            ClassicEncoderLayer(dim, num_heads, ffn_dim, dropout, causal)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_pad_mask: Optional[Tensor] = None,
                activation_checkpoint: bool = False) -> Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (Tensor): Input tensor
            attn_mask (Optional[Tensor], optional): Attention mask. Defaults to None
            key_pad_mask (Optional[Tensor], optional): Key padding mask. Defaults to None
            activation_checkpoint (bool, optional): Whether to use activation checkpointing. Defaults to False

        Returns:
            Tensor: Output tensor of the same shape as input
        """
        for layer in self.layers:
            x = layer(x,
                      attn_mask=attn_mask,
                      key_pad_mask=key_pad_mask,
                      activation_checkpoint=activation_checkpoint)
        return self.norm(x)


class ClassicModel(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 dim: int,
                 num_heads: int,
                 num_layers: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 causal: bool = False):
        """
        Implements a standard transformer model without rotary embeddings or performer attention.

        This model implements the classic transformer architecture with learned positional embeddings,
        following the original "Attention is All You Need" paper's design but with pre-layer
        normalization for better training stability.

        Args:
            num_tokens (int): Number of tokens in the vocabulary
            dim (int): Dimension of the model's hidden states
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            ffn_dim (int): Dimension of the feed-forward network
            dropout (float, optional): Dropout rate. Defaults to 0.1
            causal (bool, optional): Whether to use causal attention. Defaults to False

        Attributes:
            token_emb (nn.Embedding): Token embedding layer
            pos_emb (nn.Embedding): Positional embedding layer
            encoder (ClassicEncoder): The main encoder stack
            norm (nn.LayerNorm): Layer normalization

        Methods:
            init_params(): Initialize the model parameters
            forward(x: Tensor, attn_mask: Optional[Tensor] = None, key_pad_mask: Optional[Tensor] = None,
                    activation_checkpoint: bool = False) -> Tensor:
                Forward pass of the model

        Note:
            - The model dimension (dim) must be divisible by the number of heads
            - Uses learned positional embeddings instead of rotary embeddings
            - Implements pre-layer normalization for better training stability
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim

        if dim % num_heads != 0:
            raise ValueError(
                f"dim {dim} must be divisible by num_heads {num_heads}"
            )

        # Token and positional embeddings
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(1024, dim)  # Fixed max sequence length of 1024
        self.dropout = nn.Dropout(dropout)

        # Encoder and final normalization
        self.encoder = ClassicEncoder(
            dim,
            num_heads,
            ffn_dim,
            num_layers,
            dropout,
            causal
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_pad_mask: Optional[Tensor] = None,
                activation_checkpoint: bool = False) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of token indices
            attn_mask (Optional[Tensor], optional): Attention mask. Defaults to None
            key_pad_mask (Optional[Tensor], optional): Key padding mask. Defaults to None
            activation_checkpoint (bool, optional): Whether to use activation checkpointing. Defaults to False

        Returns:
            Tensor: Output tensor
        """
        # Get sequence length and create position indices
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Apply token and positional embeddings
        x = self.token_emb(x)
        x = x + self.pos_emb(positions)
        x = self.dropout(x)

        # Apply encoder and final normalization
        x = self.encoder(
            x,
            attn_mask=attn_mask,
            key_pad_mask=key_pad_mask,
            activation_checkpoint=activation_checkpoint
        )
        return self.norm(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]


class SimpleTransformerModel(nn.Module):
    """
    Vanilla transformer model that uses sinusoidal positional encoding.
    Uses pytorch-provided transformers.
    Useful for validating that custom models work.
    """
    def __init__(self,
                 num_tokens: int,
                 dim: int,
                 num_heads: int,
                 num_layers: int,
                 ffn_dim: int,
                 n_labels: int = 4,
                 dropout: float = 0.1,
                 max_seq_len: int = 5000):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim

        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        # Embedding layer
        self.token_emb = nn.Embedding(num_tokens, dim)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(dim, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, 
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.out_logit = nn.Linear(dim, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        
        # Apply token embedding
        x = self.token_emb(x)  # shape: (batch_size, seq_len, dim)
        
        # Apply positional encoding
        x = x.transpose(0, 1)  # shape: (seq_len, batch_size, dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # shape: (batch_size, seq_len, dim)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        return x
