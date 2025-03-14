import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from sklearn.metrics import roc_curve
import numpy as np


class BalancedAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_neg", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_neg", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        pos_idx = torch.argwhere(target == 1)[:, 0]
        neg_idx = torch.argwhere(target == 0)[:, 0]

        if pos_idx.numel() > 0:
            self.correct_pos += torch.sum(preds[pos_idx] == target[pos_idx])
            self.total_pos += target[pos_idx].numel()
        if neg_idx.numel() > 0:
            self.correct_neg += torch.sum(preds[neg_idx] == target[neg_idx])
            self.total_neg += target[neg_idx].numel()

    def compute(self) -> torch.Tensor:
        pos_acc = self.correct_pos.float() / self.total_pos
        if self.total_pos == 0:
            pos_acc = 0

        neg_acc = self.correct_neg.float() / self.total_neg
        if self.total_neg == 0:
            neg_acc = 0
        return (neg_acc + pos_acc) / 2


class PositiveRate(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor) -> None:
        self.pos += torch.sum(preds)
        self.total += preds.numel()

    def compute(self) -> torch.Tensor:
        return self.pos / self.total


class VDScore(Metric):
    def __init__(self, target_fpr=0.005, **kwargs):
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.target_fpr = target_fpr

    def update(self, scores: torch.Tensor, target: torch.Tensor) -> None:
        if scores.dtype in [torch.int, torch.long]:
            raise ValueError("scores must be a float tensor")
        self.scores.append(scores.detach().cpu())
        self.targets.append(target.detach().cpu())

    def compute(self) -> torch.Tensor:
        scores = dim_zero_cat(self.scores).numpy()
        targets = dim_zero_cat(self.targets).numpy()
        self.vds, self.threshold = calculate_vul_det_score(scores, targets, self.target_fpr)
        return torch.tensor(self.vds)


def calculate_vul_det_score(predictions, ground_truth, target_fpr=0.005):
    """
    Calculate the vulnerability detection score (VD-S) given a tolerable FPR.

    Args:
    - predictions: List of model prediction probabilities for the positive class.
    - ground_truth: List of ground truth labels, where 1 means vulnerable class, and 0 means benign class.
    - target_fpr: The tolerable false positive rate.

    Returns:
    - vds: Calculated vulnerability detection score given the acceptable .
    - threshold: The classification threashold for vulnerable prediction.
    """

    # Calculate FPR, TPR, and thresholds using ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions)

    # Filter thresholds where FPR is less than or equal to the target FPR
    valid_indices = np.where(fpr <= target_fpr)[0]

    # Choose the threshold with the largest FPR that is still below the target FPR, if possible
    if len(valid_indices) > 0:
        idx = valid_indices[-1]  # Last index where FPR is below or equal to target FPR
    else:
        # If no such threshold exists (unlikely), default to the closest to the target FPR
        idx = np.abs(fpr - target_fpr).argmin()

    chosen_threshold = thresholds[idx]

    # Classify predictions based on the chosen threshold
    classified_preds = [1 if pred >= chosen_threshold else 0 for pred in predictions]

    # Calculate VD-S
    fn = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 0])
    tp = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 1])
    vds = fn / (fn + tp) if (fn + tp) > 0 else 0

    return vds, chosen_threshold
