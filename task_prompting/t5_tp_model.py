import random

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import json
import datetime

def get_formatted_and_model(w_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    q_ids = [ex['question_id'] for ex in examples[:]] + [ex['question_id'] for ex in examples[:]]
    to_input1 = [f'<check_if_negated> <variation> {ex["variation"]}' for ex in examples[:]]
    to_input2 = [f'<canonize> <variation> {ex["variation"]}' for ex in examples[:]]
    to_input = to_input1 + to_input2
    to_label1 = ["Yes" if ex["is_negated"] else "No" for ex in examples[:]]
    to_label2 = [ex["original"] for ex in examples[:]]
    to_label = to_label1 + to_label2


    # SAVE DATA TO JSON
    # to_json = []
    # for q_id, input, label in zip(q_ids, to_input, to_label):
    #     to_json.append({
    #         'question_id': q_id,
    #         'input': input,
    #         'target': label,
    #     })
    #
    # with open('mixed_tasks.json', 'w') as fp:
    #     json.dump(to_json, fp, indent=4)

    inputs = tokenizer(to_input, padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer(to_label, padding=True, truncation=True, return_tensors="pt").input_ids
    # e0 = [f'<check_if_negated> <variation> {ex["variation"]}' for ex in examples[:1]][0]
    # print(f'example input 0: {e0}')
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss
    dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, torch.tensor(labels))

    rnd_num = random.randint(0, int(1e10))
    print(f'RANDOM SEED: {rnd_num}')
    generator1 = torch.Generator().manual_seed(rnd_num)
    train, val, test = random_split(dataset, [0.7, 0.1, 0.2], generator=generator1)
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    return_dct = {'model': model, 'tokenizer': tokenizer, 'train': train_loader, 'val': val_loader, 'test': test_loader,
            'w_path': w_path, 'device': device}
    print(f'-----------------------------\n:model={model_name}\n:w_path={w_path}\n:device={device}\n-----------------------------')
    return return_dct

def validate(model, val_loader, device):
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

def train(config_dct, num_epochs):
    model = config_dct['model']
    tokenizer = config_dct['tokenizer']
    train_loader = config_dct['train']
    val_loader = config_dct['val']
    to_save = config_dct['w_path']
    device = config_dct['device']
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    best_val_loss = 1e10
    patience = 0
    max_patience = 3
    best_state_dict = None
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
        val_loss = validate(model, val_loader, device)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            patience = 0
            model.save_pretrained(to_save)
            tokenizer.save_pretrained(to_save)
        else:
            patience += 1
            if max_patience == patience:
                print(f'max patience reached, early stopping')
                break
        # validate()


def test(config_dct):
    # 'weights/weights_9.5.25'
    test_loader = config_dct['test']
    device = config_dct['device']
    w_path = config_dct['w_path']
    test_model = T5ForConditionalGeneration.from_pretrained(w_path)
    test_tokenizer = T5Tokenizer.from_pretrained(w_path)
    test_model.to(device)
    correct_neg = 0
    incorrect_neg = 0
    correct_can = 0
    incorrect_can = 0
    for test_item in test_loader:
        test_item_tokens = test_item[0].to(device)
        test_labels = test_item[2].to(device)

        # THIS IS THE FIX ##################
        test_labels[test_labels == -100] = 0
        ####################################
        outputs = test_model.generate(test_item_tokens)

        out_str = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
        ti_gt = test_tokenizer.decode(test_labels[0], skip_special_tokens=True)
        input_str = test_tokenizer.decode(test_item_tokens[0], skip_special_tokens=False)
        # print(input_str)
        is_correct = ti_gt == out_str

        hit_str = 'WRONG'
        if '<canonize>' in input_str:
            incorrect_can += 1
            if is_correct:
                hit_str = 'CORRECT'
                correct_can += 1
                incorrect_can -= 1
        else:
            incorrect_neg += 1
            if is_correct:
                hit_str = 'CORRECT'
                correct_neg += 1
                incorrect_neg -= 1
        # Print results
        # print(
        #     f'[{hit_str}] Input: {tokenizer.decode(test_item_tokens[0], skip_special_tokens=True)} -> Output: {out_str}')

    print(f'Accuracy can: {correct_can}/{incorrect_can + correct_can}')
    print(f'Accuracy neg: {correct_neg}/{incorrect_neg + correct_neg}')
    correct_total = correct_can + correct_neg
    all_total = incorrect_can + correct_can + incorrect_neg + correct_neg
    total_acc = correct_total / all_total
    print(f'Accuracy total: {correct_total}/{all_total}')
    return total_acc

def crossvalidation(model_name, num_epochs, k):
    w_path_lst = [f'weights/w_t5_09.05.25_new_{i}' for i in range(k)]
    result_lst = []
    for idx in range(k):
        config_dct = get_formatted_and_model(w_path_lst[idx], model_name)
        train(config_dct, num_epochs)
        result_lst.append(test(config_dct))

    return result_lst

if __name__ == '__main__':
    # w_path = 'weights/w_t5_09.05.25_new'
    # model_name = 't5-base'
    # config_dct = get_formatted_and_model(w_path, model_name)
    # num_epochs = 15
    # train(config_dct, num_epochs)
    # test(config_dct)
    model_name = 't5-base'
    results = crossvalidation(model_name, 20, 5)
    print(results)
    print(f'Crossvalidation results: {sum(results)/len(results)}')
