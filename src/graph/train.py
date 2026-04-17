import argparse
from pathlib import Path
from shutil import rmtree
from math import ceil
import json
import socket

import torch
from transformers import TrainingArguments, IntervalStrategy
from torch.utils.data import WeightedRandomSampler, DataLoader
from transformers import Trainer
from datasets import Dataset

from graph.datasets.primevul import load
from graph.gnn import init_model
from graph.metrics import compute_metrics
from graph.callbacks import TorchProfilerCallback


RUNS_ON_HYDRA = socket.gethostname() == "hydra" or socket.gethostname().startswith("head")


class CustomTrainer(Trainer):
    def __init__(self, *args, get_data_collator_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_predictions = []
        self.epoch_labels = []
        if get_data_collator_fn is not None:
            self.data_collator = get_data_collator_fn(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        logits = outputs
        if isinstance(logits, tuple):
            logits = logits[-1]
        self.epoch_predictions.append(logits.detach().cpu().numpy())
        self.epoch_labels.append(inputs["label"].cpu().numpy())
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        sampler = WeightedRandomSampler(
            weights=self._get_sample_weights(train_dataset),
            num_samples=len(train_dataset),
            replacement=True
        )
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator
        )

    def _get_sample_weights(self, dataset):
        labels = torch.tensor(dataset["label"]).long()
        class_counts = torch.bincount(labels)
        weights = 1. / class_counts
        return weights[labels]


def calc_step_sizes(num_train_samples: int, batch_size: int, gradient_accumulation_steps: int, num_epochs: int):
    global_steps = num_epochs * num_train_samples // (batch_size * gradient_accumulation_steps)
    logging_steps = 5
    warmup_steps = ceil(global_steps / 20)
    return {
        "logging_steps": logging_steps,
        "save_steps": 5 * logging_steps,
        "eval_steps": 5 * logging_steps,
        "warmup_steps": warmup_steps,
    }


def train(
    model_name: str,
    ds_path: Path,
    out_root: Path,
    learning_rate: float = 2e-3,
    batch_size: int = 16,
    weight_decay: float = 0.0,
    task_id: int = 0,
    num_workers: int = 1,
    cleanup_intermediate_checkpoints: bool = True,
    use_profiler: bool = False,
):
    """Train a model on the dataset."""
    ds, _, node_type_vocab = load(ds_path, seed=task_id)
    train_ds = ds["train"]
    val_ds = ds["val"]

    # analyze_node_counts(train_ds)

    model = init_model(model_name, node_type_vocab)
    out_root = out_root / model_name
    checkpoint_dir = out_root / "checkpoints"
    tensorboard_dir = out_root / "tb_logs"
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "train_config.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "ds_path": str(ds_path),
            "out_root": str(out_root),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "task_id": task_id,
            "cleanup_intermediate_checkpoints": cleanup_intermediate_checkpoints,
            "use_profiler": use_profiler
        }, f, indent=4)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = out_root / f"{model.name}-best.ckpt"
    num_epochs = 10
    gradient_accumulation_steps = 16
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=tensorboard_dir,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=True,
        save_total_limit=2,
        dataloader_num_workers=num_workers,
        eval_strategy=IntervalStrategy.STEPS,
        log_level="info",
        report_to=["tensorboard"],
        seed=task_id,
        disable_tqdm=RUNS_ON_HYDRA,
        # label_names=["label"],
        **calc_step_sizes(len(train_ds), batch_size, gradient_accumulation_steps, num_epochs)
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        get_data_collator_fn=model.get_collate_fn,
    )

    if use_profiler:
        # profile a few training steps -> open tensorboard afterwards to see the results
        trainer.add_callback(TorchProfilerCallback(profile_dir=str(out_root / "tb_profile")))

    trainer.train()
    best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_model_dir)
    if cleanup_intermediate_checkpoints:
        rmtree(checkpoint_dir)


def analyze_node_counts(ds: Dataset):
    """Hacky way to analyze the node counts of the dataset."""
    # TODO refactor
    import matplotlib.pyplot as plt
    import seaborn as sns
    node_counts = torch.tensor([len(x_t) for x_t in ds["x_types"]])
    q95_percentile = node_counts.float().quantile(0.95)
    fig, ax = plt.subplots()
    sns.histplot(node_counts, ax=ax, kde=True)
    ax.axvline(q95_percentile, color="red", linestyle="--")
    fig.savefig("node_counts.png")
    print(f"max node count: {node_counts.max()}, 95th percentile: {q95_percentile}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train graph model.")
    p.add_argument("model_name")
    p.add_argument("ds_path", type=Path)
    p.add_argument("out_root", type=Path)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--no-cleanup-intermediate-checkpoints", action="store_true")
    p.add_argument("--use-profiler", action="store_true")
    args = p.parse_args()
    train(
        args.model_name,
        args.ds_path,
        args.out_root,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        task_id=args.task_id,
        num_workers=args.num_workers,
        cleanup_intermediate_checkpoints=not args.no_cleanup_intermediate_checkpoints,
        use_profiler=args.use_profiler,
    )


if __name__ == "__main__":
    main()
