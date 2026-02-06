import torch.nn as nn

class PRMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)
