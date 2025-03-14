import torch


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

    x = torch.concat([x_min, synth_samples, x_maj])
    y_min = torch.ones(x_min.shape[0] + synth_samples.shape[0], dtype=torch.long) * minority_class
    y_maj = torch.ones(x_maj.shape[0], dtype=torch.long) * (1 - minority_class)
    y = torch.concat([y_min, y_maj])
    return x, y
