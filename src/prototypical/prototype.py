"""Prototype computation (F5-02) [3.6, 3.7, Pers. 3.6/3.7].

    c_k = (1/|S_k|) * sum_{x_i in S_k} f_phi(x_i)

The prototype of class k is the mean embedding of its support samples.
"""
from __future__ import annotations

import torch


def compute_prototypes(
    support_embeddings: torch.Tensor, support_labels: torch.Tensor, n_way: int
) -> torch.Tensor:
    """support_embeddings: (N, dim), support_labels: (N,) with values in
    [0, n_way) -> prototypes: (n_way, dim), row k = mean of embeddings whose
    label == k (Pers. 3.6/3.7)."""
    dim = support_embeddings.shape[-1]
    prototypes = support_embeddings.new_zeros(n_way, dim)
    for k in range(n_way):
        mask = support_labels == k
        if not torch.any(mask):
            raise ValueError(f"No support samples found for class {k}")
        prototypes[k] = support_embeddings[mask].mean(dim=0)
    return prototypes
