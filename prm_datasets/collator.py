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


class PRMEvalCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        """
        batch size MUST be 1
        """
        assert len(batch) == 1, "Eval batch_size must be 1"

        sample = batch[0]

        input_ids = pad_sequence(
            sample["input_ids"],
            batch_first=True,
            padding_value=self.pad_token_id
        )  # [K, L]

        attention_mask = pad_sequence(
            sample["attention_mask"],
            batch_first=True,
            padding_value=0
        )  # [K, L]

        # labels = sample["labels"]  # [K]
        labels = torch.tensor(sample["labels"], dtype=torch.long)  # [K]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class PRMFeatureCollator:
    def __call__(self, batch):
        features = torch.stack([b["features"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        return {
            "features": features,
            "labels": labels
        }
