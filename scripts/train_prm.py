import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from prm_datasets import PRMStepDataset, PRMCollator
from models import BaseLM, PRMHead, PRMModel
from losses import PRMLoss
from optim import build_optimizer, build_scheduler
from trainers import PRMTrainer
from utils.seed import set_seed
from utils.visualization import plot_losses
from utils.config import load_config
from evaluation.evaluate_prm import evaluate_prm_processbench


def main():
    cfg = load_config("configs/train.yaml")

    # ===== basic =====
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== tokenizer =====
    model_path = cfg["model"]["base_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # ===== dataset / dataloader =====
    train_dataset = PRMStepDataset(
        data_path=cfg["data"]["train_path"],
        tokenizer=tokenizer,
        max_length=cfg["data"]["max_length"]
    )
    test_dataset = PRMStepDataset(
        data_path=cfg["data"]["test_path"],
        tokenizer=tokenizer,
        max_length=cfg["data"]["max_length"]
    )

    collator = PRMCollator(tokenizer.pad_token_id)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=cfg["data"]["shuffle"],
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collator
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=cfg["data"]["shuffle"],
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collator
    )

    # ===== model =====
    base_lm = BaseLM(
        model_path=model_path,
        freeze=cfg["model"]["freeze_base"]
    )

    prm_head = PRMHead(
        hidden_size=base_lm.model.config.hidden_size
    )

    model = PRMModel(base_lm, prm_head).to(device)

    # ===== loss / optim =====
    loss_fn = PRMLoss()

    optimizer = build_optimizer(
        model,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    total_steps = len(train_dataloader) * cfg["training"]["num_epochs"]

    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps
    )

    # ===== trainer =====
    trainer = PRMTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    # ===== training loop =====
    all_losses = []
    for epoch in range(cfg["training"]["num_epochs"]):
        print(f"\n===== Start Epoch {epoch} =====")
        loss = trainer.train_epoch(train_dataloader)
        all_losses.append(loss)
        print(f"End Epoch {epoch}: loss={loss:.4f}")
        metrics = evaluate_prm_processbench(
            model=model,
            dataloader=test_dataloader,
            device=device,
            threshold=0.5,
        )
        print(
            f"[Eval:] "
            f"FirstErrorAcc={metrics['first_error_acc']:.4f} | "
            f"AUC={metrics['step_auc']:.4f} | "
            f"FalseEarly={metrics['false_early_rate']:.4f} | "
            f"Miss={metrics['miss_rate']:.4f}"
        )
    # ===== save =====
    plot_losses(all_losses, cfg["logging"]["loss_figure_path"])

    ckpt_path = cfg["logging"]["checkpoint_dir"] + "/prm_head.pt"
    torch.save(prm_head.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
    