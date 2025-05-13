import json
from pathlib import Path
import wandb
import numpy as np
from transformers import Seq2SeqTrainingArguments
from sklearn.model_selection import KFold, train_test_split


from training import Training
from dataloaders.CheXAgentDatataset import CheXAgentDataset
from model.preprocessing_model import T5WithInversionHead


class CrossValidation:
    def __init__(self, k_folds, random_seed):
        self.k_folds = k_folds
        self.seed = random_seed

    def __call__(self, preprocessing_model: T5WithInversionHead, slake_dataset, variation_dataset, early_stopping, training_args: Seq2SeqTrainingArguments, wandb_login_key):
        k_fold = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)

        base_model = preprocessing_model.config.tokenizer

        item_ids = list(set(map(lambda item: item['question_id'], variation_dataset)))
        slake_ids = list(set(map(lambda item: item['question_id'], slake_dataset)).difference(item_ids))

        item_ids, test_ids = train_test_split(item_ids, test_size=0.2, random_state=self.seed)
        slake_ids, slake_test_ids = train_test_split(slake_ids, test_size=0.2, random_state=self.seed)

        test_ids += slake_test_ids
        test_data = list(filter(lambda item: item['question_id'] in test_ids, variation_dataset)) + list(filter(lambda item: item['question_id'] in test_ids, slake_dataset))

        variation_splits = k_fold.split(np.zeros(len(item_ids)), item_ids)
        slake_splits = k_fold.split(np.zeros(len(slake_ids)), slake_ids)

        dataset_splits = [
            (train_slk_split.tolist() + train_var_split.tolist(), val_slk_split.tolist() + val_var_split.tolist())
            for (train_slk_split, val_slk_split), (train_var_split, val_var_split) in zip(slake_splits, variation_splits)
        ]

        wandb.login(key=wandb_login_key)

        wandb.init(
            project="PPV",
            entity="PPV",
            config=training_args.to_dict(),
        )
        run_name = wandb.run.name

        training_args.report_to = "wandb"

        test_dataset = CheXAgentDataset(test_data, preprocessing_model)
        util_folder = Path(__file__).resolve().parent

        preprocessing_model_type = type(preprocessing_model)

        for fold, (train_ids, validation_ids) in enumerate(dataset_splits):
            if wandb.run is not None:
                wandb.run.name += f"-fold-{fold}"
            elif wandb_login_key is not None:
                wandb.init(
                    name=f"{run_name}-fold-{fold}",
                    project="PPV",
                    entity="PPV",
                    config=training_args.to_dict(),
                )

            if wandb.run is None:
                new_weights_name = util_folder / f"{run_name}-fold-{fold}"
            else:
                new_weights_name = util_folder / wandb.run.name

            training_args.output_dir = str(new_weights_name)

            train_data = list(filter(lambda item: item['question_id'] in train_ids, variation_dataset)) + list(filter(lambda item: item['question_id'] in train_ids, slake_dataset))
            validation_data = list(filter(lambda item: item['question_id'] in validation_ids, variation_dataset)) + list(filter(lambda item: item['question_id'] in validation_ids, slake_dataset))

            preprocessing_model = preprocessing_model_type.from_pretrained(base_model)

            train_dataset = CheXAgentDataset(train_data, preprocessing_model)
            validation_dataset = CheXAgentDataset(validation_data, preprocessing_model)

            training_pipeline = Training()
            training_pipeline(preprocessing_model, train_dataset, validation_dataset, test_dataset,
                              early_stopping=early_stopping, training_args=training_args,
                              wandb_login_key=wandb_login_key)

            with open(str(new_weights_name / "fold_splits.json"), 'w') as f:
                json.dump({'train_ids': train_ids, 'validation_ids': validation_ids, 'test_ids': test_ids}, f,
                          ensure_ascii=False, indent=4)

            if wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    random_state = 42
    file_folder = Path(__file__).resolve().parent

    json_path = str(file_folder.parent / "prompts_change/custom_dataset/variations.fixed.json")
    slake_json_path = str(file_folder.parent / "prompts_change/custom_dataset/slake.dataset.json")

    with open(slake_json_path, 'r', encoding='utf-8') as slake_file:
        slake_data = json.load(slake_file)

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    wandb_key = "a9f105e8b3bc98e07700e93201d4b02c1c75106d"

    training_arguments = Seq2SeqTrainingArguments(
        output_dir="postproc-checkpoints",
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
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

    model = T5WithInversionHead.from_pretrained('t5-base')
    cross_validation = CrossValidation(5, 42)
    cross_validation(model, slake_data, data, 10, training_arguments, wandb_login_key=wandb_key)
