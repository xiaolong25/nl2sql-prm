import torch

def build_optimizer(model, lr, weight_decay):
    # 当前训练所有 requires_grad=True 的参数（即只训练 reward head，不训练 base LM）
    params = filter(lambda p: p.requires_grad, model.parameters())
    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay
    )
