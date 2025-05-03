import os
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5PreTrainedModel, T5Config, GenerationMixin


class T5WithInversionHeadConfig(T5Config):
    def __init__(self, tokenizer : str = None, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["tokenizer"] = self.tokenizer

        return d

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        possible_tuple = super(T5WithInversionHeadConfig, cls).from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        if kwargs.get('return_unused_kwargs', False):
            base_cfg, unused_kwargs = possible_tuple
        else:
            base_cfg, unused_kwargs = possible_tuple, None

        is_wrapper = os.path.isdir(pretrained_model_name_or_path)

        if not is_wrapper:
            base_cfg.tokenizer = pretrained_model_name_or_path

        if kwargs.get('return_unused_kwargs', False):
            return base_cfg, unused_kwargs
        return base_cfg

class T5WithInversionHead(T5PreTrainedModel, GenerationMixin):
    config_class = T5WithInversionHeadConfig

    def __init__(self, config: T5WithInversionHeadConfig):
        super().__init__(config)
        self.t5_model = T5ForConditionalGeneration(config)
        self.tokenizer = T5Tokenizer.from_pretrained(config.tokenizer)
        self.inversion_classifier = nn.Linear(config.d_model, 1)

        self.post_init()

        self.t5_model.config.output_hidden_states = True

        self.config.output_hidden_states = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.t5_model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)

        canonical_question = outputs.logits

        enc = outputs.encoder_last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()

        summed = (enc * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)

        pooled = summed / counts

        inversion_logits = self.inversion_classifier(pooled).squeeze(-1)

        return canonical_question, inversion_logits

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        return self.t5_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **model_kwargs,
        )

    def generate(self, *args, **kwargs):
        return self.t5_model.generate(*args, **kwargs)

    def save_pretrained(self, save_dir: str, **kwargs):
        os.makedirs(save_dir, exist_ok=True)

        self.t5_model.save_pretrained(save_dir, **kwargs)
        self.t5_model.generation_config.save_pretrained(save_dir, **kwargs)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir, **kwargs)

        torch.save(
            self.inversion_classifier.state_dict(),
            os.path.join(save_dir, "inversion_classifier.bin"),
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        head_file = os.path.join(pretrained_model_name_or_path, "inversion_classifier.bin")
        is_wrapper = os.path.isdir(pretrained_model_name_or_path)
        possible_tuple = T5WithInversionHeadConfig.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        if kwargs.get('return_unused_kwargs', False):
            config, unused_kwargs = possible_tuple
        else:
            config, unused_kwargs = possible_tuple, None

        model = T5WithInversionHead(config)

        if is_wrapper:
            model.t5_model = T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )

            model.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

            state_dict = torch.load(head_file, map_location="cpu")
            model.inversion_classifier.load_state_dict(state_dict)

        return model

    def canonicalize_and_classify_from_text(self, input_text: str, max_length: int = 128):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        self.to(device)

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            padding='max_length',
        ).to(device)

        generated_ids = self.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
        )

        dummy_labels = inputs["input_ids"].clone()
        dummy_labels[dummy_labels == self.tokenizer.pad_token_id] = -100

        _, inversion_logits = self.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=dummy_labels
        )

        inv_flag_pred = (inversion_logits > 0).long()

        generated_ids[generated_ids == -100] = self.tokenizer.pad_token_id
        canonical_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return canonical_text, inv_flag_pred.item() == 1


if __name__ == "__main__":
    # model_name = 't5-base'
    model_with_inversion = T5WithInversionHead.from_pretrained("./checkpoint-12915")

    text = "Does the picture not contain brain?"
    canonical, inverted = model_with_inversion.canonicalize_and_classify_from_text(text)
    print(f"Canonical: {canonical}")
    print(f"Inverted: {inverted}")
