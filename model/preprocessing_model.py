import os
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5WithInversionHead(nn.Module):
    def __init__(self, model_path):
        super(T5WithInversionHead, self).__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(model_path)

        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.inversion_classifier = nn.Linear(self.t5_model.config.d_model, 1)  # Binary classification
        self.t5_model.config.output_hidden_states = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.t5_model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)

        canonical_question = outputs.logits

        enc = outputs.encoder_last_hidden_state  # (batch, seq_len, dim)
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)

        summed = (enc * mask).sum(dim=1)  # sum over seq_len → (batch, dim)
        counts = mask.sum(dim=1).clamp(min=1)  # how many real tokens → (batch, 1)

        pooled = summed / counts

        inversion_logits = self.inversion_classifier(pooled).squeeze(-1)

        return canonical_question, inversion_logits

    def generate(self, *args, **kwargs):
        return self.t5_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str):
        """
        1) Save tokenizer
        2) Use HF’s safe saver for T5 (dedup shared weights)
        3) Save the inversion head separately
        """
        os.makedirs(save_directory, exist_ok=True)

        # 1) tokenizer files
        self.tokenizer.save_pretrained(save_directory)

        # 2) T5’s own safe save (handles shared embeddings)
        self.t5_model.save_pretrained(save_directory)

        # 3) your inversion head
        torch.save(
            self.inversion_classifier.state_dict(),
            os.path.join(save_directory, "inversion_classifier.bin")
        )

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        model = cls(load_directory)

        # 3) T5 core (this re-ties all weights correctly)
        model.t5_model = T5ForConditionalGeneration.from_pretrained(load_directory)
        model.tokenizer = T5Tokenizer.from_pretrained(load_directory)

        # 4) inversion head
        head_sd = torch.load(
            os.path.join(load_directory, "inversion_classifier.bin"),
            map_location="cpu"
        )
        model.inversion_classifier.load_state_dict(head_sd)

        return model


if __name__ == "__main__":
    model_name = 't5-base'

    # Initialize the modified model
    model_with_inversion = T5WithInversionHead(model_name)