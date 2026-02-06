import torch
import torch.nn as nn

class PRMModel(nn.Module):
    def __init__(self, base_lm, prm_head):
        super().__init__()
        self.base_lm = base_lm
        self.prm_head = prm_head

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            features = self.base_lm(input_ids, attention_mask)  # [B, H]

        logits = self.prm_head(features)  # [B]
        return logits
