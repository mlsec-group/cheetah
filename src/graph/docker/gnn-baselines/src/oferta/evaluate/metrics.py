import numpy as np
import torch
from torch.nn.functional import one_hot
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def get_ratio(labels):
    ones = (labels == 1).sum()
    zeros = (labels == 0).sum()
    return (ones / (ones + zeros)).item()


def evaluate_metrics(model, loader: DataLoader, device: torch.device, criterion, class_weights=None):
    model.eval()
    if class_weights is None:
        class_weights = torch.tensor([1, 1], dtype=torch.float32).to(model.device)

    with torch.no_grad(), logging_redirect_tqdm():
        all_predictions, all_targets = [], []
        all_losses = []
        for data in tqdm(loader, leave=False, disable=False):
            data = data.to(device)
            batch_class_weights = class_weights[data.y]

            target = data.y.view(-1)
            # data.x = data.x.to(torch.float32).to(device)
            # data.edge_index = data.edge_index.to(device)
            # data.batch = data.batch.to(device)

            out = model(data)
            loss = criterion(out, one_hot(target, num_classes=2).float())
            loss = (batch_class_weights.view(-1, 1) * loss).mean().item()

            target = target.cpu().detach().numpy()
            #pred = out.argmax(dim=1).cpu().detach().numpy()
            pred = out.cpu().detach().numpy()
            all_predictions.append(pred)
            all_losses.append(loss)
            all_targets.append(target)

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        acc = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        ratio = get_ratio(all_predictions)
        target_names = ["non-vul", "vul"]

        # TODO remove
        try:
            report = classification_report(
                all_targets,
                all_predictions,
                zero_division=0,
                target_names=target_names,
                output_dict=True,
            )
        except ValueError:
            report = {}
    model.train()
    return np.mean(all_losses), acc, f1, recall, precision, ratio, report
