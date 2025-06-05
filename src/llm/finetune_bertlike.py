import os
import math
import argparse

import torch
import torch.nn.functional as F

from typing import Optional, Union, Tuple

from datasets import load_from_disk

from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, balanced_accuracy_score


class Classifier(RobertaForSequenceClassification):
    def __init__(self, config):
        config.num_labels = 1
        self.pos_weight = 1
        super().__init__(config)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            assert self.num_labels == 1, "We want to do binary classification with a single output"
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).float(), pos_weight=self.pos_weight)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def compute_metrics(pred):
    """
    Calculates a few helpful metrics
    :param pred: list
    """
    true = pred.label_ids
    predicted = pred.predictions > 0
    return {
        "MCC": matthews_corrcoef(true, predicted),
        "F1": f1_score(true, predicted),
        "Acc": accuracy_score(true, predicted),
        "BAcc": balanced_accuracy_score(true, predicted),
    }


def tokenize(sample, tokenizer, truncation_length):
    return tokenizer(sample["texts"], truncation=True, max_length=truncation_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("base_model", default="microsoft/unixcoder-base-nine")
    parser.add_argument("uid", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = Classifier.from_pretrained(args.base_model, device_map="auto")

    dataset = load_from_disk(args.dataset_dir)

    final_model_path = os.path.join("hf-model", args.base_model.replace("/", "__"))
    tensorboard_path = os.path.join("tb_logs", args.base_model.replace("/", "__"))
    # find uid
    os.makedirs(final_model_path, exist_ok=True)

    uid = 0
    if args.uid is not None:
        uid = args.uid
    else:
        while os.path.isdir(os.path.join(final_model_path, uid)) or os.path.isdir(os.path.join(tensorboard_path, uid)):
            uid += 1
    final_model_path = os.path.join(final_model_path, uid)
    tensorboard_path = os.path.join(tensorboard_path, uid)

    # find the maximum supported length of the model
    # do not ask, why this needs to be so complicated, nobody knows
    model_max_length = min(
        tokenizer.model_max_length,
        2**int(math.log(model.config.max_position_embeddings, 2)), # biggest possible power of two that is smaller
    )
    print(f"Tokenizing datasets with maximum length {model_max_length}")
    tokenize_with_tokenizer = lambda s: tokenize(s, tokenizer, model_max_length)
    columns_to_remove = [c for c in dataset["train"].column_names if "label" not in c]
    print(f"Removing {columns_to_remove} after tokenization")
    dataset = dataset.map(tokenize_with_tokenizer, batched=True, num_proc=None, remove_columns=columns_to_remove, load_from_cache_file=False)
    NUMBER_TRAIN_SAMPLES = int(len(dataset["train"]) * 0.9)
    print(f"Selecting {NUMBER_TRAIN_SAMPLES} of {len(dataset['train'])} samples for training")
    train_set = dataset["train"].shuffle(load_from_cache_file=False).select(range(NUMBER_TRAIN_SAMPLES))
    val_set = dataset["val"]

    positives = torch.sum(torch.as_tensor(train_set["labels"]) == 1)
    total = len(train_set)
    pos_weight = (total - positives) / positives
    model.pos_weight = pos_weight

    training_args = TrainingArguments(
        output_dir=final_model_path,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=32,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir=tensorboard_path,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        logging_steps=2500,
        save_steps=2500,
        eval_steps=2500,
        save_total_limit=2,
        eval_strategy="steps",
        log_level="info",
        report_to=["tensorboard"],
        seed=1337,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer, padding=True),
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
    )

    trainer.train()