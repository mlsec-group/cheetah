import torch
from loguru import logger


def smote(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    SMOTE: synthetic minority oversampling technique

    1. take samples from minority class
    2. compute nearest neighbors
    3. create synthetic samples by interpolating between neighbors
    4. sub-sample majority class until class balance is reached

    """

    minority_class = torch.bincount(y).argmin().item()
    minority_idx = torch.argwhere(y == minority_class)[:, 0]
    majority_idx = torch.argwhere(y != minority_class)[:, 0]
    x_min = x[minority_idx]
    x_maj = x[majority_idx]

    # skip somote if there are fewer than 2 minority samples
    if minority_idx.shape[0] < 2:
        return x, y

    # minority oversampling
    distances = torch.cdist(x_min, x_min, p=2)
    _, neighbors = torch.topk(distances, k=2, largest=False)
    neighbors = neighbors[:, -1]

    synth_samples = torch.stack([x_min, x_min[neighbors]], dim=1).mean(dim=1)

    # add synth samples to minority class
    missing_minority = majority_idx.shape[0] - minority_idx.shape[0]
    if synth_samples.shape[0] > missing_minority:
        synth_samples = synth_samples[:missing_minority]
    missing_minority -= synth_samples.shape[0]

    # majority undersampling
    if missing_minority > 0:
        subsample_idx = torch.randperm(majority_idx.shape[0])[: majority_idx.shape[0] - missing_minority]
        x_maj = x_maj[subsample_idx]

    x_new = torch.concat([x_min, synth_samples, x_maj]).to(x)
    y_min = torch.ones(x_min.shape[0] + synth_samples.shape[0], dtype=torch.long) * minority_class
    y_maj = torch.ones(x_maj.shape[0], dtype=torch.long) * (1 - minority_class)
    y_new = torch.concat([y_min, y_maj]).to(y)
    return x_new, y_new


def projection_loss(emb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minimize cosine distance of intra-class samples and maximize it for inter-class pairs."""
    loss = torch.tensor(0.0, device=emb.device)
    if emb.shape[0] == 1 or torch.all(y == y[0]):
        # skip if there is only one sample
        # skip if there are only samples of the same class
        return loss

    for label in torch.unique(y):
        same = emb[torch.argwhere(y == label)[:, 0]]
        others = emb[torch.argwhere(y != label)[:, 0]]

        intra = pairwise_cosine_dist(same, same)
        inter = pairwise_cosine_dist(same, others)

        loss += intra.sum() / (same.numel() * same.numel())
        loss -= inter.sum() / (same.numel() * inter.numel())

    if torch.isnan(loss):
        logger.warning("Projection loss is NaN")
    return loss


def pairwise_cosine_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine distance between two tensors."""
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    if torch.any(torch.isnan(a_norm)):
        a_norm = a
    if torch.any(torch.isnan(b_norm)):
        b_norm = b

    if a_norm.sum() == 0 and b_norm.sum() == 0:
        # define distance to zero for zero vectors
        cos = torch.zeros(a_norm.shape[0], b_norm.shape[0]).to(a.device)
    else:
        cos = 1 - torch.mm(a_norm, b_norm.transpose(0, 1))
    return cos


def regularization_loss(emb: torch.Tensor) -> torch.Tensor:
    """Regularize the norm of the embeddings."""
    return torch.linalg.matrix_norm(emb)
