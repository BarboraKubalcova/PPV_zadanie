from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader, TensorDataset
import json

model_name = "t5-base"  # or "t5-base", or even a GPT model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

with open('dataset/check_if_negated_dataset.json', 'r') as fp:
    examples = json.load(fp)

inputs = tokenizer([ex["input_text"] for ex in examples], padding=True, truncation=True, return_tensors="pt")
labels = tokenizer([ex["target_text"] for ex in examples], padding=True, truncation=True, return_tensors="pt").input_ids
labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss

dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, torch.tensor(labels))
loader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

# for epoch in range(3):
#     for batch in loader:
#         input_ids, attention_mask, labels = batch
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         print(f"Loss: {loss.item()}")


test_input = "check if negated: Is there any bruised area near the chest?"
input_ids = tokenizer(test_input, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))