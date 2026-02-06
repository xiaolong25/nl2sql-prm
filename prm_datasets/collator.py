import torch
from torch.nn.utils.rnn import pad_sequence

class PRMCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=self.pad_token_id
        )
        attention_mask = pad_sequence(
            [b["attention_mask"] for b in batch],
            batch_first=True,
            padding_value=0
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
