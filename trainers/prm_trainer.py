from tqdm import tqdm
import torch


class PRMTrainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        losses = []

        for batch in tqdm(dataloader, desc="Training"):
            # ---- move to device ----
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device).float()

            # ---- forward ----
            logits = self.model(input_ids, attention_mask)
            logits = logits.squeeze(-1)  # (B,)

            # ---- loss ----
            loss = self.loss_fn(logits, labels)

            # ---- backward ----
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            losses.append(loss.item())

        return sum(losses) / len(losses)
