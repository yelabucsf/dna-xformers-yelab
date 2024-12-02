#!/usr/bin/env python

from typing import cast, Optional
from torch import Tensor
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .mhsa import MultiHeadSelfAttention, MultiHeadCrossAttention


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
                 causal: bool = False,
                 rotary_pos_emb: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, performer, causal, rotary_pos_emb)
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
                 causal: bool = False,
                 rotary_pos_emb: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            RoformerEncoderLayer(dim, num_heads, ffn_dim, performer, dropout, causal, rotary_pos_emb)
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
                 performer: bool = False,
                 dropout=0.1,
                 causal: bool = False,
                 rotary_pos_emb: bool = True):
        """
        Implements the Roformer model, a transformer variant with optional rotary position embeddings.

        This model combines the power of transformer architectures with rotary position embeddings,
        allowing for better handling of sequential data without explicit position encodings.

        Args:
            num_tokens (int): Number of tokens in the vocabulary.
            dim (int): Dimension of the model's hidden states.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            ffn_dim (int): Dimension of the feed-forward network.
            performer (bool, optional): Whether to use Performer attention. Defaults to False.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            causal (bool, optional): Whether to use causal attention. Defaults to False.
            rotary_pos_emb (bool, optional): Whether to use rotary positional embeddings. Defaults to True.

        Attributes:
            token_emb (nn.Embedding): Token embedding layer.
            encoder (RoformerEncoder): The main encoder stack.
            norm (nn.LayerNorm): Layer normalization.

        Methods:
            init_params(): Initialize the model parameters.
            forward(x: Tensor, attn_mask: Optional[Tensor] = None, key_pad_mask: Optional[Tensor] = None,
                    activation_checkpoint: bool = False) -> Tensor:
                Forward pass of the model.

        Note:
            - The model dimension (dim) must be divisible by the number of heads.
            - The model supports both standard attention and Performer attention.
            - Rotary positional embeddings can be enabled/disabled at initialization.

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
            causal,
            rotary_pos_emb
        )

        self.norm = nn.LayerNorm(dim)

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
