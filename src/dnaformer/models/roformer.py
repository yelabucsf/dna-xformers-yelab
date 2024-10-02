#!/usr/bin/env python

from typing import cast, Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from rotary_embedding_torch import RotaryEmbedding
from performer_pytorch import FastAttention
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, performer: bool = False,
                 causal: bool = False):
        """
        Implements multi-head self-attention mechanism with optional rotary positional encoding and performer attention.

        This class provides a flexible implementation of self-attention, supporting both standard
        scaled dot-product attention and Performer's fast attention mechanism. It also incorporates
        rotary positional embeddings for enhanced position-aware representations.

        Args:
            dim (int): The input and output dimension of the model.
            num_heads (int): Number of attention heads.
            performer (bool, optional): Whether to use Performer's fast attention. Defaults to False.
            causal (bool, optional): Whether to apply causal masking. Defaults to False.

        Attributes:
            num_heads (int): Number of attention heads.
            head_dim (int): Dimension of each attention head.
            qkv (nn.Linear): Linear layer for computing query, key, and value.
            out (nn.Linear): Output linear layer.
            rotary_emb (RotaryEmbedding): Rotary positional embedding layer.
            causal (bool): Whether the attention is causal.
            performer (bool): Whether to use Performer's fast attention.
            attn_fn (Callable): The attention function to use (either fast attention or scaled dot-product).

        Methods:
            forward(x: Tensor, attn_mask: Optional[Tensor] = None, key_pad_mask: Optional[Tensor] = None,
                    rotary_pos_emb: bool = True) -> Tensor:
                Compute self-attention on the input tensor.

        Note:
            - The input dimension (dim) must be divisible by the number of heads.
            - When using Performer, the `FastAttention` class from `performer_pytorch` is utilized.
            - Rotary positional embeddings are applied by default but can be disabled.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.rotary_emb = RotaryEmbedding(self.head_dim//2,
                                          theta=10_000,
                                          interpolate_factor=1)
        self.causal = causal

        self.performer = performer
        if performer:
            self.attn_fn = FastAttention(
                dim_heads=self.head_dim,
                causal=causal,
            )
        else:
            self.attn_fn = F.scaled_dot_product_attention

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_pad_mask: Optional[Tensor] = None,
                rotary_pos_emb: bool = True) -> Tensor:
        """
        Run the forward pass of the multi-head self-attention layer.

        Parameters:
        -----------
        x : Tensor
            Input tensor.
        attn_mask : Optional[Tensor]
            A boolean mask where a value of True indicates that the element
            should NOT take part in attention. 
        key_pad_mask : Optional[Tensor]
            A boolean mask where a value of True indicates that the element
            is a padding token and should NOT take part in attention.
        rotary_pos_emb : bool
            A boolean flag indicating whether to apply rotary positional
            encoding to the queries and keys.
        """
        batch_size, seq_len, _ = x.shape
        # reshape to (qkv, batch_size, num_heads, seq_len, head_dim)
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary positional encoding
        if rotary_pos_emb:
            with torch.autocast(device_type=x.device.type, enabled=False):
                q = self.rotary_emb.rotate_queries_or_keys(q.float())
                k = self.rotary_emb.rotate_queries_or_keys(k.float())

        if attn_mask is not None:
            B: int = x.size(0)
            if attn_mask.ndim == 2:  # batched
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_mask = attn_mask.expand(B, self.num_heads, -1, -1)
            elif attn_mask.ndim == 3:  # unbatched
                attn_mask = attn_mask.unsqueeze(1)
                attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)
            attn_mask = ~attn_mask

        if key_pad_mask is not None:
            if key_pad_mask.ndim != 2:
                raise ValueError("key_pad_mask must have 2 dimensions")
            key_pad_mask = key_pad_mask.unsqueeze(1).unsqueeze(1)
            key_pad_mask = key_pad_mask.expand(-1, self.num_heads, -1, -1)
            key_pad_mask = ~key_pad_mask

            if attn_mask is None:
                attn_mask = key_pad_mask
            else:
                attn_mask = attn_mask + key_pad_mask

        if self.performer:
            attn = self.attn_fn(q, k, v)
        else:
            attn = self.attn_fn(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=self.causal,
            )

        x = attn.transpose(1, 2).contiguous()
        x = x.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        return self.out(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q: nn.Linear = nn.Linear(dim, dim)
        self.k: nn.Linear = nn.Linear(dim, dim)
        self.v: nn.Linear = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.rotary_emb = RotaryEmbedding(self.head_dim//2,
                                          theta=10_000,
                                          interpolate_factor=1)

    def forward(self, q, k, v, attn_mask=None, rotary_pos_emb=True):
        batch_size, q_len, _ = q.shape
        k_len = k.shape[1]

        q = self.q(q).reshape(
            batch_size, q_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        k = self.k(k).reshape(
            batch_size, k_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.v(v).reshape(
            batch_size, k_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        # Apply rotary positional encoding
        if rotary_pos_emb:
            with torch.autocast(device_type=q.device.type, enabled=False):
                q = self.rotary_emb.rotate_queries_or_keys(q.float())
                k = self.rotary_emb.rotate_queries_or_keys(k.float())

        attn = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.reshape(batch_size, q_len, self.dim)

        return self.out(attn)


class RoformerDecoderLayer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_dim,
                 dropout=0.1,
                 causal: bool = False):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, performer=False, causal=causal)
        self.enc_dec_attn = MultiHeadCrossAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, self_mask=None, cross_mask=None, activation_checkpoint=False):
        res = x
        x = self.norm1(x)
        if activation_checkpoint:
            x = cast(
                Tensor,
                checkpoint(
                    self.self_attn,
                    use_reentrant=False,
                    x=x,
                    attn_mask=self_mask,
                ),
            )
        else:
            x = self.self_attn(x, attn_mask=self_mask)
        x = self.dropout(x)
        x = res + x

        res = x
        x = self.norm2(x)
        if activation_checkpoint:
            x = cast(
                Tensor,
                checkpoint(
                    self.enc_dec_attn,
                    use_reentrant=False,
                    q=x, k=h, v=h,
                    attn_mask=cross_mask,
                ),
            )
        else:
            x = self.enc_dec_attn(q=x, k=h, v=h, attn_mask=cross_mask)
        x = self.dropout(x)
        x = res + x

        res = x
        x = self.norm3(x)
        if activation_checkpoint:
            x = cast(
                Tensor,
                checkpoint(self.ffn, use_reentrant=False, input=x),
            )
        else:
            x = self.ffn(x)
        x = res + x
        return x


class RoformerEncoderLayer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_dim,
                 performer: bool = False,
                 dropout=0.1,
                 causal: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, performer, causal)
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
                activation_checkpoint: bool = False):
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


class RoformerEncoder(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_dim,
                 num_layers,
                 performer: bool = False,
                 dropout=0.1,
                 causal: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            RoformerEncoderLayer(dim, num_heads, ffn_dim, performer, dropout, causal)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_pad_mask: Optional[Tensor] = None,
                activation_checkpoint: bool = False) -> Tensor:
        for layer in self.layers:
            x = layer(x,
                      attn_mask=attn_mask,
                      key_pad_mask=key_pad_mask,
                      activation_checkpoint=activation_checkpoint)
        return self.norm(x)


class RoformerModel(nn.Module):
    def __init__(self,
                 num_tokens,
                 dim,
                 num_heads,
                 num_layers,
                 ffn_dim,
                 n_labels: Optional[int] = 4,
                 performer: bool = False,
                 dropout=0.1,
                 causal: bool = False):
        """
        Implements the Roformer model, a transformer variant with rotary position embeddings.

        This model combines the power of transformer architectures with rotary position embeddings,
        allowing for better handling of sequential data without explicit position encodings.

        Args:
            num_tokens (int): Number of tokens in the vocabulary.
            dim (int): Dimension of the model's hidden states.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            ffn_dim (int): Dimension of the feed-forward network.
            n_labels (Optional[int], optional): Number of output labels. Defaults to 4.
            performer (bool, optional): Whether to use Performer attention. Defaults to False.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            causal (bool, optional): Whether to use causal attention. Defaults to False.

        Attributes:
            token_emb (nn.Embedding): Token embedding layer.
            encoder (RoformerEncoder): The main encoder stack.
            norm (nn.LayerNorm): Layer normalization.
            out_logit (nn.Linear): Output linear layer for classification (if n_labels is provided).

        Methods:
            init_params(): Initialize the model parameters.
            forward(x: Tensor, attn_mask: Optional[Tensor] = None, key_pad_mask: Optional[Tensor] = None,
                    activation_checkpoint: bool = False) -> Tensor:
                Forward pass of the model.

        Note:
            - The model dimension (dim) must be divisible by the number of heads.
            - If n_labels is provided, the model includes a classification head.
            - The model supports both standard attention and Performer attention.

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

        # embedding
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.encoder = RoformerEncoder(
            dim,
            num_heads,
            ffn_dim,
            num_layers,
            performer,
            dropout,
            causal
        )

        self.norm = nn.LayerNorm(dim)

        if n_labels is not None:
            self.out_logit = nn.Linear(dim, n_labels)

    def init_params(self):
        for name, param in self.named_parameters():
            if 'bias' not in name and 'norm' not in name and 'rotary_emb' not in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_pad_mask: Optional[Tensor] = None,
                activation_checkpoint: bool = False) -> Tensor:
        x = self.token_emb(x)
        x = self.encoder(
            x,
            attn_mask=attn_mask,
            key_pad_mask=key_pad_mask,
            activation_checkpoint=activation_checkpoint
        )
        return x


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

