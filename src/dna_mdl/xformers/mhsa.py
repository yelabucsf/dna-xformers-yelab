from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from performer_pytorch import FastAttention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, performer: bool = False,
                 causal: bool = False, rotary_pos_emb: bool = True):
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
            rotary_pos_emb (bool, optional): Whether to use rotary positional embeddings. Defaults to True.

        Attributes:
            num_heads (int): Number of attention heads.
            head_dim (int): Dimension of each attention head.
            qkv (nn.Linear): Linear layer for computing query, key, and value.
            out (nn.Linear): Output linear layer.
            rotary_emb (RotaryEmbedding): Rotary positional embedding layer if rotary_pos_emb is True.
            causal (bool): Whether the attention is causal.
            performer (bool): Whether to use Performer's fast attention.
            attn_fn (Callable): The attention function to use (either fast attention or scaled dot-product).
            use_rotary (bool): Whether to use rotary positional embeddings.

        Methods:
            forward(x: Tensor, attn_mask: Optional[Tensor] = None, key_pad_mask: Optional[Tensor] = None) -> Tensor:
                Compute self-attention on the input tensor.

        Note:
            - The input dimension (dim) must be divisible by the number of heads.
            - When using Performer, the `FastAttention` class from `performer_pytorch` is utilized.
            - Rotary positional embeddings can be enabled/disabled at initialization.

        Raises:
            ValueError: If dim is not divisible by num_heads.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.use_rotary = rotary_pos_emb
        if rotary_pos_emb:
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
                key_pad_mask: Optional[Tensor] = None) -> Tensor:
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
        """
        batch_size, seq_len, _ = x.shape
        # reshape to (qkv, batch_size, num_heads, seq_len, head_dim)
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary positional encoding if enabled
        if self.use_rotary:
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



