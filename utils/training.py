import json
import evaluate
from torch import device as dev
from sklearn.metrics import accuracy_score
from torch import nn, Tensor, tensor, float, no_grad
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq

from model.preprocessing_model import T5WithInversionHead
from dataloaders.CheXAgentDatataset import CheXAgentDataset


class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        inversion_flags = tensor([f['inversion_flag'] for f in features], dtype=float)

        for f in features:
            del f['inversion_flag']

        batch = super().__call__(features)

        batch['inversion_flag'] = inversion_flags

        return batch

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, model, args, train_dataset=None, eval_dataset=None, device=dev('cpu'),**kwargs):
        super().__init__(model=model.to(device), args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)
        self.device = device
        self.bleu_metric = evaluate.load("bleu")

    def training_step(self, model: nn.Module, inputs: dict[str, Tensor], num_items_in_batch = None):
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        labels = inputs['labels'].to(self.device)
        inversion_flag = inputs['inversion_flag'].to(self.device)

        canonical_question, inversion_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss_fct = nn.CrossEntropyLoss()
        generation_loss = loss_fct(canonical_question.view(-1, canonical_question.size(-1)), labels.view(-1))

        bce_loss_fct = nn.BCEWithLogitsLoss()
        inversion_loss = bce_loss_fct(inversion_logits, inversion_flag)

        total_loss = generation_loss + inversion_loss

        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gey_kwargs):
        model.eval()

        with no_grad():
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            labels = inputs["labels"].to(self.device)
            inversion_flag = inputs["inversion_flag"].to(self.device)

            canonical_question, inversion_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss_fct = nn.CrossEntropyLoss()
            generation_loss = loss_fct(canonical_question.view(-1, canonical_question.size(-1)), labels.view(-1))
            bce_loss_fct = nn.BCEWithLogitsLoss()
            inversion_loss = bce_loss_fct(inversion_logits, inversion_flag)

            loss = generation_loss + inversion_loss

            if prediction_loss_only:
                return loss, None, None

            return loss, (canonical_question, inversion_logits), (labels, inversion_flag)

    def compute_metrics(self, eval_preds):
        #TODO check eval_preds type, highly possible inversion flags are concatenated in a weird way that will interfere with decoding
        (text_logits, inversion_logits), (text_labels, inversion_flags) = eval_preds

        predictions = text_logits.argmax(dim=-1)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(text_labels, skip_special_tokens=True)

        bleu_score = self.bleu_metric.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )["bleu"]

        predicted_flags = (inversion_logits > 0).int()
        actual_flags = inversion_flags.int()

        inversion_accuracy = accuracy_score(actual_flags.cpu(), predicted_flags.cpu())

        return {
            'inversion_accuracy': inversion_accuracy,
            'bleu_score': bleu_score,
        }

class Training:
    def __call__(self, model, training_dataset, validation_dataset, testing_dataset, early_stopping, training_args: TrainingArguments):
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

        metrics = trainer.evaluate(testing_dataset)
        print(metrics)

if __name__ == '__main__':
    json_path = "C:\\Skola\\TUKE\\ING\\PPV\\assignment1\\prompts_change\\custom_dataset\\variations.json"
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    preprocessing_model = T5WithInversionHead('t5-base')

    train_dataset = CheXAgentDataset(data, preprocessing_model)

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
        metric_for_best_model="loss",
        save_total_limit=2,
    )

    training_pipeline = Training()
    training_pipeline(preprocessing_model, train_dataset, train_dataset, train_dataset, early_stopping=5, training_args=training_arguments)