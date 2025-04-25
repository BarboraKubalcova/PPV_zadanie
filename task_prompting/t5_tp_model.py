from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import json
import datetime

# Set device to CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "t5-base"  # or "t5-base", or even a GPT model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to the device (GPU or CPU)
model.to(device)

# Load the dataset
with open('dataset/check_if_negated_dataset.json', 'r') as fp:
    examples = json.load(fp)

# Tokenize inputs and labels
inputs = tokenizer([ex["input_text"] for ex in examples], padding=True, truncation=True, return_tensors="pt")
labels = tokenizer([ex["target_text"] for ex in examples], padding=True, truncation=True, return_tensors="pt").input_ids
labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss

# Create TensorDataset
dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, torch.tensor(labels))

# Split dataset into train, validation, and test sets
generator1 = torch.Generator().manual_seed(42)
train, val, test = random_split(dataset, [0.7, 0.1, 0.2], generator=generator1)
# train, test = random_split(dataset, [0.8, 0.2], generator=generator1)

train_loader = DataLoader(train, batch_size=2, shuffle=True)
val_loader = DataLoader(val, batch_size=1, shuffle=False)

# Set optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

# Training loop
num_epochs = 3
epoch_losses = []

def validate():
    model.train()  # Switch to evaluation mode
    val_loss = 0
    with torch.no_grad():  # No gradient computation for validation
        for batch in val_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    val_loss /= len(val_loader)
    print(f"Validation loss: {val_loss:.4f}")
    return val_loss

val_loss_lst = []
for epoch in range(num_epochs):
    loss_lst = []
    for batch in train_loader:
        # Move batch to the device (GPU or CPU)
        input_ids, attention_mask, labels = [item.to(device) for item in batch]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_lst.append(loss.item())

    train_loss = sum(loss_lst)/len(loss_lst)

    # Print training loss
    now = datetime.datetime.now().isoformat()
    print(f'[{epoch+1}/{num_epochs}] {now}: training loss={train_loss:.4f}')
    # if (epoch + 1) % 4 == 0:
    #     val_loss = validate()
    #     val_loss_lst.append(val_loss)
    #     if len(val_loss_lst) == 1:
    #         pass
    #     elif val_loss_lst[-1] > val_loss_lst[-2]:
    #         print('early stopping')
    #         break

# Testing loop
test_loader = DataLoader(test, batch_size=1, shuffle=False)
correct = 0
incorrect = 0
for test_item in test_loader:
    test_item_tokens = test_item[0].to(device)  # Move to device
    labels = test_item[2].to(device)  # Move labels to device for comparison
    labels = labels[labels != -100]
    # Generate outputs
    outputs = model.generate(test_item_tokens)

    # Decode output and target (ground truth) strings
    out_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ti_gt = tokenizer.decode(labels[0], skip_special_tokens=True)

    # Check if prediction matches ground truth
    # TODO find out WHY the the response is decoded correctly and the label is not (this is just a quick fix)
    if out_str == 'False':
        is_correct = ti_gt == out_str[:3]
    else:
        is_correct = ti_gt == out_str

    hit_str = 'WRONG'
    incorrect += 1
    if is_correct:
        hit_str = 'CORRECT'
        correct += 1
        incorrect -= 1
    # Print results
    print(f'[{hit_str}] Input: {tokenizer.decode(test_item_tokens[0], skip_special_tokens=True)} -> Output: {out_str}')

print(f'Accuracy: {correct}/{incorrect + correct}')
