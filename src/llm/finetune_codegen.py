import argparse
import os
from typing import Optional, Union, Tuple, List

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, LlamaForSequenceClassification
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, balanced_accuracy_score


class Classifier(LlamaForSequenceClassification):
    def __init__(self, config):
        config.num_labels = 1
        self.pos_weight = 1
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


def run_finetune_codegen(dataset_dir: str, uid: Optional[str] = None) -> None:
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-multi_P", trust_remote_code=True, revision="c936d25f73e74984748a5ba39151b6bc7a056e2e")
    model = Classifier.from_pretrained("Salesforce/codegen25-7b-multi_P", device_map="auto", torch_dtype=torch.bfloat16, revision="c936d25f73e74984748a5ba39151b6bc7a056e2e")

    dataset = load_from_disk(dataset_dir)

    final_model_path = os.path.join("hf-model", "Salesforce/codegen25-7b-multi_P".replace("/", "__"))
    tensorboard_path = os.path.join("tb_logs", "Salesforce/codegen25-7b-multi_P".replace("/", "__"))
    os.makedirs(final_model_path, exist_ok=True)

    # find uid
    if uid is None:
        uid_i = 0
        while os.path.isdir(os.path.join(final_model_path, str(uid_i))) or os.path.isdir(
            os.path.join(tensorboard_path, str(uid_i))
        ):
            uid_i += 1
        uid = str(uid_i)
    final_model_path = os.path.join(final_model_path, uid)
    tensorboard_path = os.path.join(tensorboard_path, uid)

    model_max_length = 512 # taken from PrimeVul
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
        num_train_epochs=4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4*4,
        per_device_eval_batch_size=8,
        warmup_steps=2000,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir=tensorboard_path,
        load_best_model_at_end=True,
        bf16=torch.cuda.is_available(),
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

    tokenizer.pad_token = tokenizer.eos_token
    trainer.train()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir")
    p.add_argument("uid", nargs="?", default=None)
    args = p.parse_args()
    run_finetune_codegen(args.dataset_dir, args.uid)


if __name__ == "__main__":
    main()
