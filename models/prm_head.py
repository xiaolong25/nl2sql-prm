import torch.nn as nn

class PRMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: [B, H]
        # ---- 对齐 dtype ----
        if x.dtype != self.net[0].weight.dtype:
            x = x.to(self.net[0].weight.dtype)

        return self.net(x).squeeze(-1)  # [B]
