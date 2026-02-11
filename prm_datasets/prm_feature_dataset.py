import torch
from torch.utils.data import Dataset

class PRMFeatureDataset(Dataset):
    def __init__(self, feature_path):
        data = torch.load(feature_path)
        self.features = data["features"].float()   # [N, H], 确保 float
        self.labels = data["labels"].long()        # [N], 确保 long

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],  # ✅ 改成 labels
        }
