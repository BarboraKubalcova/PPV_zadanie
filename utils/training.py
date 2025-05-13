import json
import wandb
import torch
import evaluate
import pandas as pd
from torch import nn
from transformers.utils import logging
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    Trainer
)

from model.preprocessing_model import T5WithInversionHead
from dataloaders.CheXAgentDatataset import CheXAgentDataset

logger = logging.get_logger(__name__)


class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        inversion_flags = torch.tensor([f['inversion_flag'] for f in features], dtype=torch.float)

        for f in features:
            del f['inversion_flag']

        batch = super().__call__(features)

        batch['inversion_flag'] = inversion_flags

        return batch

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, model, args: Seq2SeqTrainingArguments, training_dataset=None, eval_dataset=None, device=torch.device('cpu'), **kwargs):
        super().__init__(model=model.to(device), args=args, train_dataset=training_dataset, eval_dataset=eval_dataset, compute_metrics=self.compute_metrics, **kwargs)

        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")

        self.inversion_funct = nn.BCEWithLogitsLoss()
        self.cross_entropy_funct = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        canonical_logits, inversion_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )

        gen_loss = self.cross_entropy_funct(canonical_logits.view(-1, canonical_logits.size(-1)), inputs["labels"].view(-1))
        inv_loss = self.inversion_funct(inversion_logits, inputs["inversion_flag"].float())

        total_loss = gen_loss + inv_loss

        if return_outputs:
            return total_loss, (canonical_logits, inversion_logits)

        return total_loss

    def prediction_step(self, model: nn.Module, inputs: dict, prediction_loss_only: bool, ignore_keys=None, **gey_kwargs):
        model.eval()

        input_ids = inputs["input_ids"].to(self.args.device)
        attention_mask = inputs["attention_mask"].to(self.args.device)
        labels = inputs["labels"].to(self.args.device)
        inv_flag = inputs["inversion_flag"].to(self.args.device)

        with torch.no_grad():
            canonical_logits, inversion_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            gen_loss = self.cross_entropy_funct(canonical_logits.view(-1, canonical_logits.size(-1)), labels.view(-1))
            inv_loss = self.inversion_funct(inversion_logits, inv_flag)

            total_loss = gen_loss + inv_loss

        if prediction_loss_only:
            return total_loss, None, None

        gen_kwargs = {
            "max_length": self.args.generation_max_length or 128,
            "num_beams": getattr(self.args, "generation_num_beams", 1),
            "early_stopping": True,
        }
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
        inv_flag_pred = (inversion_logits > 0).long()

        return total_loss, (generated_ids, inv_flag_pred), (labels, inv_flag.long())

    def compute_metrics(self, eval_preds):
        (canonical_pred, inv_flag_pred), (canonical_true, inv_flag_true) = eval_preds

        canonical_pred[canonical_pred == -100] = self.processing_class.pad_token_id
        canonical_true[canonical_true == -100] = self.processing_class.pad_token_id

        decoded_preds = self.processing_class.batch_decode(canonical_pred, skip_special_tokens=True)
        decoded_labels = self.processing_class.batch_decode(canonical_true, skip_special_tokens=True)

        bleu_score = self.bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

        rouge_score = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

        inv_acc = accuracy_score(torch.from_numpy(inv_flag_true).cpu(), torch.from_numpy(inv_flag_pred).cpu())

        return {
            'inversion_accuracy': inv_acc,
            'bleu_score': bleu_score["bleu"],
            "rouge1": rouge_score["rouge1"],
            "rouge2": rouge_score["rouge2"],
            "rougeL": rouge_score["rougeL"],
        }

    def _load_best_model(self):
        best_ckpt = self.state.best_model_checkpoint

        if best_ckpt is None:
            logger.warning("No best_model_checkpoint found, skipping load_best_model.")
            return

        logger.info(f"Loading best model from {best_ckpt} via from_pretrained()")

        new_model = type(self.model).from_pretrained(best_ckpt)
        self._move_model_to_device(new_model, self.args.device)

        self.model = new_model

        if hasattr(self, "model_wrapped"):
            self.model_wrapped = new_model

class TrainEvalCallback(TrainerCallback):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics and any(k.startswith("eval") for k in metrics):
            train_metrics = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset, metric_key_prefix="train")

            self.trainer.log(train_metrics)

        return control

class Training:
    def __call__(self, model, training_dataset, valid_dataset, testing_dataset, early_stopping, training_args: Seq2SeqTrainingArguments, wandb_login_key: str | None = None):
        if wandb_login_key is not None and wandb.run is None:
            wandb.login(key=wandb_login_key)

            wandb.init(
                project="PPV",
                entity="PPV",
                config=training_args.to_dict(),
            )

            training_args.output_dir = f"{wandb.run.name}-{training_args.output_dir}"

        training_args.report_to = "none" if wandb.run is None else "wandb"

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            training_dataset=training_dataset,
            eval_dataset=valid_dataset,
            processing_class=model.tokenizer,
            data_collator=CustomDataCollator(model.tokenizer),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        trainer.add_callback(TrainEvalCallback(trainer))
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=early_stopping))

        trainer.train()

        metrics = trainer.evaluate(testing_dataset, metric_key_prefix="test")
        trainer.log(metrics)

        pred_output = trainer.predict(testing_dataset)

        gen_ids, inv_flags_pred = pred_output.predictions
        label_ids, inv_flags_true = pred_output.label_ids

        gen_ids[gen_ids == -100] = model.tokenizer.pad_token_id
        label_ids[label_ids == -100] = model.tokenizer.pad_token_id

        decoded_preds = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        decoded_labels = model.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        orig_inputs = [ex["variation"] for ex in testing_dataset.data]

        table = wandb.Table(columns=[
            "input_question",
            "canonized_pred",
            "canonized_true",
            "inv_pred",
            "inv_true"
        ])

        for inp, pred, true, ip, it in zip(
                orig_inputs, decoded_preds, decoded_labels,
                inv_flags_pred.tolist(), inv_flags_true.tolist()
        ):
            table.add_data(inp, pred, true, int(ip), int(it))

        trainer.log({"test_predictions_table": table})

        df = pd.DataFrame(table.data, columns=table.columns)
        print(df.to_string(index=False))

if __name__ == '__main__':
    random_state = 42

    json_path = "../prompts_change/custom_dataset/variations.fixed.json"
    slake_json_path = "../prompts_change/custom_dataset/slake.dataset.json"

    with open(slake_json_path, 'r', encoding='utf-8') as slake_file:
        slake_data = json.load(slake_file)

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    item_ids = list(set(map(lambda item: item['question_id'], data)))
    slake_ids = list(set(map(lambda item: item['question_id'], slake_data)).difference(item_ids))

    slake_train_ids, slake_validation_ids = train_test_split(slake_ids, test_size=0.2, random_state=random_state)
    slake_test_ids, slake_validation_ids = train_test_split(slake_validation_ids, test_size=0.33, random_state=random_state)

    train_ids, validation_ids = train_test_split(item_ids, test_size=0.3, random_state=random_state)
    test_ids, validation_ids = train_test_split(validation_ids, test_size=0.33, random_state=random_state)

    train_ids += slake_train_ids
    validation_ids += slake_validation_ids
    test_ids += slake_test_ids

    test_data = list(filter(lambda item: item['question_id'] in test_ids, data)) + list(filter(lambda item: item['question_id'] in test_ids, slake_data))
    train_data = list(filter(lambda item: item['question_id'] in train_ids, data)) + list(filter(lambda item: item['question_id'] in train_ids, slake_data))
    validation_data = list(filter(lambda item: item['question_id'] in validation_ids, data)) + list(filter(lambda item: item['question_id'] in validation_ids, slake_data))

    preprocessing_model = T5WithInversionHead.from_pretrained('t5-base')

    test_dataset = CheXAgentDataset(test_data, preprocessing_model)
    train_dataset = CheXAgentDataset(train_data, preprocessing_model)
    validation_dataset = CheXAgentDataset(validation_data, preprocessing_model)

    wandb_key = "a9f105e8b3bc98e07700e93201d4b02c1c75106d"

    training_arguments = Seq2SeqTrainingArguments(
        output_dir="postproc-checkpoints",
        per_device_train_batch_size=9,
        per_device_eval_batch_size=9,
        num_train_epochs=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        weight_decay=0.05,
        predict_with_generate=True,
        remove_unused_columns=False,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        generation_max_length=128,
        generation_num_beams=4,
    )

    training_pipeline = Training()
    training_pipeline(preprocessing_model, train_dataset, validation_dataset, test_dataset, early_stopping=10, training_args=training_arguments, wandb_login_key=wandb_key)