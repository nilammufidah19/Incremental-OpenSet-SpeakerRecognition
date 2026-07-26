"""Distance-based classification (F5-03) [3.6, 3.7, Pers. 3.8-3.10].

    d(z_q, c_k) = ||z_q - c_k||_2                                  (3.8)
    p(y=k|x_q)  = exp(-d(z_q,c_k)) / sum_k' exp(-d(z_q,c_k'))      (3.9)
    L           = -log p(y=y_true | x_q)                          (3.10)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def euclidean_distance(query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """query_embeddings: (Q, dim), prototypes: (n_way, dim) -> (Q, n_way)
    pairwise Euclidean distances (Pers. 3.8)."""
    return torch.cdist(query_embeddings, prototypes, p=2)


def classify_log_probs(query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Log-probabilities (Q, n_way) via softmax over negative distance
    (Pers. 3.9), computed in log-space (log_softmax) for numerical stability."""
    distances = euclidean_distance(query_embeddings, prototypes)
    return F.log_softmax(-distances, dim=-1)


def negative_log_likelihood(log_probs: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
    """Pers. 3.10 training loss: mean NLL of the true class over the batch."""
    return F.nll_loss(log_probs, query_labels)


def predict(query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (predicted_label, min_distance) per query -- min_distance is
    what F5-08's open-set threshold check compares against (Bab 4.6)."""
    distances = euclidean_distance(query_embeddings, prototypes)
    min_distance, predicted_label = distances.min(dim=-1)
    return predicted_label, min_distance
