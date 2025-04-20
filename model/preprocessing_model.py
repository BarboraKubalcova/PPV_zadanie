from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn as nn


class T5WithInversionHead(nn.Module):
    def __init__(self, model_name):
        super(T5WithInversionHead, self).__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
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


if __name__ == "__main__":
    model = 't5-base'

    # Initialize the modified model
    model_with_inversion = T5WithInversionHead(model)