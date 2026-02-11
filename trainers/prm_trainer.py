# ================================================================
# File: prm_trainer.py
# Author: Xiaolong Ji
# Date: 2026/02/06
# 
# Description:
#   Trainer implementation for PRM.
# ================================================================


from tqdm import tqdm
import torch


class PRMTrainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, dataloader, tb_logger=None, epoch=None, cfg=None):
        self.model.train()
        losses = []

        for step, batch in enumerate(
            tqdm(dataloader, desc=f"Training (epoch={epoch})")
        ):
            # if(step>2):
            #     break
            # ---- move to device ----
            if not cfg["training"]["train_by_PreProcess_data"]:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device).float()
                # ---- forward ----
                logits = self.model(input_ids, attention_mask)
                logits = logits.squeeze(-1)  # (B,)
            else:
                features = batch["features"].to(self.device) 
                labels = batch["labels"].to(self.device).float()
                # ---- forward ----
                logits = self.model(features)

            # ---- loss ----
            loss = self.loss_fn(logits, labels)

            # ---- backward ----
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss_value = loss.item()
            losses.append(loss_value)

            # ---- TensorBoard: step-level loss ----
            if tb_logger is not None and epoch is not None:
                global_step = epoch * len(dataloader) + step
                tb_logger.log_train_step_loss(
                    loss_value,
                    global_step
                )

        return sum(losses) / max(len(losses), 1)
