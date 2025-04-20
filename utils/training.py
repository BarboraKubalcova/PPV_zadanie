import json
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, DataCollatorForSeq2Seq

from assignment1.model.preprocessing_model import T5WithInversionHead
from assignment1.dataloaders.CheXAgentDatataset import CheXAgentDataset


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
        super().__init__(model=model.to(device), args=args, train_dataset=training_dataset, eval_dataset=eval_dataset, **kwargs)

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
        # inversion flag loss
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
        #TODO premium chatGPT revised, should be working, but still needs to be checked
        (canonical_pred, inv_flag_pred), (canonical_true, inv_flag_true) = eval_preds

        decoded_preds = self.tokenizer.batch_decode(canonical_pred, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(canonical_true, skip_special_tokens=True)

        bleu_score = self.bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

        rouge_score = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

        inv_acc = accuracy_score(inv_flag_true.cpu(), inv_flag_pred.cpu())

        return {
            'inversion_accuracy': inv_acc,
            'bleu_score': bleu_score["bleu"],
            "rouge1": rouge_score["rouge1"],
            "rouge2": rouge_score["rouge2"],
            "rougeL": rouge_score["rougeL"],
        }

class Training:
    def __call__(self, model, training_dataset, validation_dataset, testing_dataset, early_stopping, training_args: Seq2SeqTrainingArguments):
        early_stop_callback = EarlyStoppingCallback(
            early_stopping_patience=early_stopping
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            processing_class=model.tokenizer,
            data_collator=CustomDataCollator(model.tokenizer),
            callbacks=[early_stop_callback]
        )

        trainer.train()

        metrics = trainer.evaluate(testing_dataset, metric_key_prefix="test")
        print(metrics)

if __name__ == '__main__':
    random_state = 42

    json_path = "C:\\Skola\\TUKE\\ING\\PPV\\assignment1\\prompts_change\\custom_dataset\\variations.json"

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    item_ids = set(map(lambda item: item['question_id'], data))

    train_ids, validation_ids = train_test_split(item_ids, test_size=0.3, random_state=random_state)
    test_ids, validation_ids = train_test_split(validation_ids, test_size=0.33, random_state=random_state)

    test_data = list(filter(lambda item: item['question_id'] in test_ids, data))
    train_data = list(filter(lambda item: item['question_id'] in train_ids, data))
    validation_data = list(filter(lambda item: item['question_id'] in validation_ids, data))

    preprocessing_model = T5WithInversionHead('t5-base')

    test_dataset = CheXAgentDataset(test_data, preprocessing_model)
    train_dataset = CheXAgentDataset(train_data, preprocessing_model)
    validation_dataset = CheXAgentDataset(validation_data, preprocessing_model)

    training_arguments = Seq2SeqTrainingArguments(
        output_dir="./postproc-checkpoints",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        weight_decay=0.01,
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
    training_pipeline(preprocessing_model, train_dataset, validation_dataset, test_dataset, early_stopping=5, training_args=training_arguments)