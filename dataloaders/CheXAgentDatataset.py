from torch.utils.data import Dataset
import torch

class CheXAgentDataset(Dataset):
    def __init__(self, data, model, max_length=512):
        self.data = data
        self.model = model
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        inputs = self.model.tokenizer(example['variation'],
                                      max_length=self.max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt'
                                      )

        labels = self.model.tokenizer(example['original'],
                                      max_length=self.max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt'
                                      )

        inversion_flag = example['is_negated']

        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0),
            'inversion_flag': torch.tensor(inversion_flag, dtype=torch.long)  # Binary flag
        }