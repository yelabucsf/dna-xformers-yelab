#!/usr/bin/env python3
#
# Roformer pretrainer for DNA sequences

import torch
from torch import Tensor
from typing import Optional, Tuple

def mask_tensor(x: Tensor,
                mask_val: int,
                mask_prob: float = .15,
                frac_keep: float = 0.1,
                frac_rand: float = 0.0,
                prob_scale: Optional[Tensor] = None,
                rand_vals: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Mask a PyTorch tensor.

    Args:
        x (Tensor): Input tensor to be masked.
        mask_val (int): Value to use for masking.
        mask_prob (float): Probability of masking each element.
        frac_keep (float): Fraction of masked elements to keep unchanged.
        frac_rand (float): Fraction of masked elements to replace with random values.
        prob_scale (Optional[Tensor]): Tensor to scale masking probabilities.
        rand_vals (Optional[Tensor]): Tensor of random values to use for replacement.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: 
            - Masked tensor
            - Boolean tensor indicating masked positions
            - Original tensor (labels)

    Raises:
        ValueError: If probability values are not between 0 and 1.
    """
    if not 0 <= mask_prob <= 1 or not 0 <= frac_keep <= 1 or not 0 <= frac_rand <= 1:
        raise ValueError("Probability values must be between 0 and 1")

    _dev = x.device
    with torch.no_grad():
        x = x.cpu().to(torch.long)
        mask_probs = torch.full(x.shape, mask_prob, dtype=torch.float)

        if prob_scale is not None:
            prob_scale = prob_scale.cpu()
            mask_probs *= prob_scale

        g_dev = torch.Generator(device="cpu")
        mask_ixs = torch.bernoulli(mask_probs, generator=g_dev).bool()
        keep_ixs = torch.bernoulli(torch.full(x.shape, frac_keep), generator=g_dev).bool() & mask_ixs
        rand_ixs = torch.bernoulli(torch.full(x.shape, frac_rand), generator=g_dev).bool() & mask_ixs & (~keep_ixs)

        mskd = x.clone()
        mskd = torch.where(mask_ixs & ~keep_ixs, torch.full_like(x, mask_val), mskd)

        if frac_rand > 0.0:
            if rand_vals is None:
                rand_vals = x[prob_scale > 0].unique() if prob_scale is not None else x.unique()
            if rand_vals is None:
                raise ValueError("rand_vals must not be None if frac_rand > 0.0")
            if rand_vals.numel() == 0:
                raise ValueError("rand_vals must not be empty if frac_rand > 0.0")
            n_rand = int(rand_ixs.sum().item())
            rands = rand_vals[torch.randperm(rand_vals.numel(), generator=g_dev)[:n_rand]]
            mskd = torch.where(rand_ixs, rands, mskd)

        labels = x.clone()

    return mskd.to(_dev), mask_ixs.to(_dev), labels.to(_dev)
