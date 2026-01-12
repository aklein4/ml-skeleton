import torch
import torch.nn.functional as F

import numpy as np


def random_subset_vectors(
    x: torch.FloatTensor,
    num_vectors: int,
    normalize: bool = False,
) -> torch.FloatTensor:
    """Select a random subset of vectors from the sequence,
    and return them as a batch of flattened vectors.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim).
        num_vectors (int): Number of vectors to select.
        normalize (bool): Whether to normalize the vectors.
    """
    x_seq = x.permute(1, 0, 2) # [S, B, D]
    
    perm_inds = torch.randperm(
        x_seq.shape[0],
        device=x_seq.device,
        dtype=torch.long
    )
    x_subset = x_seq[perm_inds[:num_vectors]] # [N, B, D]
    
    x_vectors = x_subset.permute(1, 0, 2).reshape(x.shape[0], -1) # [B, N, D], [B, N*D]

    if normalize:
        x_vectors = F.normalize(x_vectors, p=2, dim=-1)

    return x_vectors


def gram_values(x: torch.FloatTensor, normalize: bool=True) -> torch.FloatTensor:
    """Compute the Gram values for a given tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, feature_dim).
    """
    x_norm = x
    if normalize:
        x_norm = F.normalize(x, dim=-1, p=2)

    gram_matrix = torch.matmul(
        x_norm, x_norm.T
    )

    mask = ~torch.eye(
        gram_matrix.shape[0],
        device=gram_matrix.device,
        dtype=torch.bool
    )
    gram_values = torch.masked_select(
        gram_matrix,
        mask
    )

    return gram_values


def covariance_loss(
    x: torch.FloatTensor,
    return_gram: bool=False
) -> torch.FloatTensor:
    x = F.normalize(x, p=2, dim=-1)

    # compute the gram matrix and variance
    gram = gram_values(x, normalize=False)
    variance = x.var(0)

    cov_loss = (
        gram.pow(2).sum() -
        variance.pow(2).sum()
    )

    # scale the cov loss
    ideal_cov_loss = (
        (1 / x.shape[-1]) * gram.numel() -
        ((1 / x.shape[-1])**2) * variance.numel()
    )

    loss = torch.log2(cov_loss) - np.log2(ideal_cov_loss)

    if return_gram:
        return loss, gram
    return loss


def diagonal_variance_loss(
    x: torch.FloatTensor
) -> torch.FloatTensor:
    return (x.var(0) - 1).pow(2).mean()


def mean_loss(
    x: torch.FloatTensor
) -> torch.FloatTensor:
    
    mean_pow = x.mean(0).pow(2).mean()
    ideal_mean_pow = 1 / x.shape[0]

    return torch.log2(mean_pow) - np.log2(ideal_mean_pow)
