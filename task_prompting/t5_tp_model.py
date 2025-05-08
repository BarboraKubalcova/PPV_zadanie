from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import json
import datetime

def validate():
    model.train()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    val_loss /= len(val_loader)
    print(f"Validation loss: {val_loss:.4f}")
    return val_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "t5-base"  # or "t5-base", or even a GPT model
tokenizer = T5Tokenizer.from_pretrained(model_name)
special_tokens_dict = {'additional_special_tokens': ['<check_if_negated>', '<original>', '<variation>', '<canonize>']}
model = T5ForConditionalGeneration.from_pretrained(model_name)

num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

model.to(device)

with open('../prompts_change/custom_dataset/variations.fixed.json', 'r') as fp:
    examples1 = json.load(fp)

with open('../prompts_change/custom_dataset/slake.dataset.json', 'r') as fp:
    examples2 = json.load(fp)

examples = examples1 + examples2

to_input1 = [f'<check_if_negated> <variation> {ex["variation"]}' for ex in examples[:]]
to_input2 = [f'<canonize> <variation> {ex["variation"]}' for ex in examples[:]]
to_input = to_input1 + to_input2
to_label1 = [str(ex["is_negated"]) for ex in examples[:]]
to_label2 = [ex["original"] for ex in examples[:]]
to_label = to_label1 + to_label2

inputs = tokenizer(to_input, padding=True, truncation=True, return_tensors="pt")
labels = tokenizer(to_label, padding=True, truncation=True, return_tensors="pt").input_ids
e0 = [f'<check_if_negated> <variation> {ex["variation"]}' for ex in examples[:1]][0]
print(f'example input 0: {e0}')
labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss
dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, torch.tensor(labels))

generator1 = torch.Generator().manual_seed(42)
train, val, test = random_split(dataset, [0.7, 0.1, 0.2], generator=generator1)
train_loader = DataLoader(train, batch_size=8, shuffle=True)
val_loader = DataLoader(val, batch_size=1, shuffle=False)

def main():
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    num_epochs = 20
    epoch_losses = []

    val_loss_lst = []
    for epoch in range(num_epochs):
        loss_lst = []
        neg_loss_lst = []
        can_loss_lst = []
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
            for input_id in input_ids:
                out_str = tokenizer.decode(input_id, skip_special_tokens=False)
                if '<canonize>' in out_str:
                    can_loss_lst.append(loss.item())
                elif '<check_if_negated>' in out_str:
                    neg_loss_lst.append(loss.item())
                else:
                    return

            loss_lst.append(loss.item())

        train_loss = sum(loss_lst) / len(loss_lst)
        neg_loss = sum(neg_loss_lst) / len(neg_loss_lst)
        can_loss = sum(can_loss_lst) / len(can_loss_lst)

        # Print training loss
        now = datetime.datetime.now().isoformat()
        print(f'[{epoch + 1}/{num_epochs}] {now}: total={train_loss:.4f}; neg={neg_loss:.4f}; can={can_loss:.4f}')

    # Testing loop
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    correct = 0
    incorrect = 0
    for test_item in test_loader:
        test_item_tokens = test_item[0].to(device)
        test_labels = test_item[2].to(device)

        # THIS IS THE FIX ##################
        test_labels[test_labels == -100] = 0
        ####################################
        outputs = model.generate(test_item_tokens)

        out_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ti_gt = tokenizer.decode(test_labels[0], skip_special_tokens=True)

        is_correct = ti_gt == out_str
        hit_str = 'WRONG'
        incorrect += 1
        if is_correct:
            hit_str = 'CORRECT'
            correct += 1
            incorrect -= 1
        # Print results
        # print(
        #     f'[{hit_str}] Input: {tokenizer.decode(test_item_tokens[0], skip_special_tokens=True)} -> Output: {out_str}')

    print(f'Accuracy: {correct}/{incorrect + correct}')

if __name__ == '__main__':
    main()
