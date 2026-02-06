import torch
import torch.nn as nn
from transformers import AutoModel

class BaseLM(nn.Module):
    def __init__(self, model_path, freeze=True, dtype=torch.float16):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            output_hidden_states=False  # ⭐ 关键：不要返回所有层
        )

        # PRM 训练必须关闭 cache（否则会存 KV）
        self.model.config.use_cache = False

        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        """
        返回：每个样本最后一个有效 token 的 hidden state
        shape: [B, H]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 只用最后一层 hidden state
        last_hidden = outputs.last_hidden_state  # [B, T, H]

        # 找到每个样本最后一个非 padding token
        lengths = attention_mask.sum(dim=1) - 1  # [B]
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)

        last_token_hidden = last_hidden[batch_idx, lengths]  # [B, H]
        return last_token_hidden
