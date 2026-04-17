import argparse
import glob
import json
from pathlib import Path

import torch
from transformers import TrainingArguments

from graph.datasets.primevul import load
from graph.train import CustomTrainer
from graph.metrics import compute_metrics
from graph.gnn import load_model


def evaluate(ds_path: Path, model_path: Path):
    """Load a model checkpoint, predict on the test set and calculate metrics."""
    ds, _, _ = load(ds_path)
    test_ds = ds["test"]

    model_ckpts = glob.glob(f"{model_path}/*-best.ckpt")
    assert len(model_ckpts) == 1, f"Expected 1 best model checkpoint, got {len(model_ckpts)}"
    model_ckpt = model_ckpts[0]
    model = load_model(model_ckpt)

    trainer = CustomTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=model_path,
            per_device_eval_batch_size=64,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            log_level="info",
            label_names=["label"],
            dataloader_pin_memory=False,
        ),
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        get_data_collator_fn=model.get_collate_fn,
    )
    preds = trainer.predict(test_ds)
    metrics = compute_metrics(preds)
    preds = preds.predictions
    if preds.ndim > 1:
        preds = preds.squeeze(-1)
    preds = preds.tolist()

    model_path = Path(model_path)
    with open(model_path / "preds.json", "w") as f:
        json.dump(preds, f)
    with open(model_path / "test_results.json", "w") as f:
        json.dump(metrics, f, indent=4)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate graph model on test split.")
    p.add_argument("ds_path", type=Path)
    p.add_argument("model_path", type=Path)
    args = p.parse_args()
    evaluate(args.ds_path, args.model_path)


if __name__ == "__main__":
    main()
