import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Starcoder2ForSequenceClassification,
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from paths import model_slug, resolve_model_path


class Classifier(Starcoder2ForSequenceClassification):
    def __init__(self, config):
        config.num_labels = 1
        self.pos_weight = torch.tensor(1)
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            assert self.num_labels == 1, "We want to do binary classification with a single output"

            loss = F.binary_cross_entropy_with_logits(pooled_logits.view(-1), labels.view(-1).float(), pos_weight=self.pos_weight)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


def calculate_vul_det_score(ground_truth, predictions, target_fpr=0.005):
    # inspired by: https://github.com/DLVulDet/PrimeVul/blob/6f54687c84947b1d17486495440b37030d147289/calc_vd_score.py#L6C1-L43C33
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
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions, drop_intermediate=True)

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
    classified_preds = [1 if pred.item() >= chosen_threshold else 0 for pred in predictions]

    # Calculate VD-S
    fn = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 0])
    tp = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 1])
    vds = fn / (fn + tp) if (fn + tp) > 0 else 0

    return vds


def tokenize(sample, tokenizer, truncation_length):
    return tokenizer(sample["texts"], truncation=True, max_length=truncation_length)


def calc_metrics(prediction_scores, labels, threshold=0.5):
    cpu_labels = labels
    if torch.is_tensor(labels):
        cpu_labels = labels.detach().cpu().numpy()
    cpu_labels = np.asarray(cpu_labels).reshape(-1)
    if torch.is_tensor(prediction_scores):
        prediction_scores = prediction_scores.detach().cpu().numpy()
    prediction_scores = np.asarray(prediction_scores)
    predictions = (prediction_scores >= threshold)
    metrics = {
        "AUC": roc_auc_score(cpu_labels, prediction_scores),
        "AP": average_precision_score(cpu_labels, prediction_scores),
        "VD-S": calculate_vul_det_score(cpu_labels, prediction_scores),
        "MCC": matthews_corrcoef(cpu_labels, predictions),
        "BAcc": balanced_accuracy_score(cpu_labels, predictions),
        "F1": f1_score(cpu_labels, predictions),
        "Prec": precision_score(cpu_labels, predictions),
        "Rec": recall_score(cpu_labels, predictions),
    }

    return metrics


def evaluate_model(model, tokenizer, val_set, do_round=False):
    device = next(model.parameters()).device
    collator = DataCollatorWithPadding(tokenizer, padding=True)
    @torch.no_grad()
    def create_logits(sample):
        inputs = collator(sample)
        inputs = inputs.to(device)

        logits = model(**inputs).logits

        return dict(logits=logits.cpu().flatten().tolist())

    logits = val_set.map(create_logits, batched=True, batch_size=4, num_proc=None, load_from_cache_file=False)

    logits = torch.as_tensor(logits["logits"])
    predictions = torch.sigmoid(logits)

    return calc_metrics(predictions, val_set["labels"]), predictions.flatten().cpu().tolist()


def evaluate_checkpoint(checkpoint_path, tokenizer, val_set):
    model = Classifier.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = tokenizer.eos_token_id
    param = next(model.parameters())
    print(
        f"[eval] checkpoint={checkpoint_path} | torch.cuda.is_available()={torch.cuda.is_available()} | "
        f"device_count={torch.cuda.device_count()} | model_device={param.device} | dtype={param.dtype}",
        flush=True,
    )
    metrics, predictions = evaluate_model(model, tokenizer, val_set)

    with open(checkpoint_path / "results.json", "w", encoding="utf8") as f:
        json.dump(metrics, f)
    with open(checkpoint_path / "predictions.json", "w", encoding="utf8") as f:
        json.dump(predictions, f)

    print(json.dumps(metrics, indent=4))


def run_eval_starcoder(dataset_dir: Path, uid: int) -> None:
    base_model = "bigcode/starcoder2-7b"
    slug = model_slug(base_model)
    final_model_path = dataset_dir / "hf-model" / slug / str(uid)
    if not final_model_path.is_dir():
        raise ValueError("Provided path should be a directory containing all checkpoints to be tested in respective folders")

    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(base_model))

    dataset = load_from_disk(str(dataset_dir / "hf"))["test"]
    model_max_length = 512
    print(f"Tokenizing datasets with maximum length {model_max_length}")
    tokenize_with_tokenizer = lambda s: tokenize(s, tokenizer, model_max_length)
    columns_to_remove = [c for c in dataset.column_names if "label" not in c]
    print(f"Removing {columns_to_remove} after tokenization")
    dataset = dataset.map(
        tokenize_with_tokenizer, batched=True, num_proc=None, remove_columns=columns_to_remove, load_from_cache_file=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    for checkpoint in tqdm(final_model_path.iterdir()):
        evaluate_checkpoint(checkpoint, tokenizer, dataset)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("uid", type=int)
    args = p.parse_args()
    run_eval_starcoder(args.dataset_dir, args.uid)


if __name__ == "__main__":
    main()
