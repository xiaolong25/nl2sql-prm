# ================================================================
# File: train_prm.py
# Author: Xiaolong Ji
#
# Description:
#   Main training entry for NL2SQL Process Reward Model (PRM).
#   - Supports step-level supervision
#   - Logs metrics to TensorBoard
#   - Saves best checkpoints based on ProcessBench metrics
# ================================================================


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
from utils.logger import setup_logger
from utils.tensorboard_logger import TensorboardLogger
from evaluation.evaluate_prm import evaluate_prm_processbench

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"    # DP\DDP
    cfg = load_config("configs/train.yaml")

    # ===== logger =====
    logger = setup_logger(cfg["logging"]["log_file"])
    logger.info("========== Start PRM Training ==========")

    # ===== basic =====
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: GPU-[{device}]")

    # ===== tokenizer =====
    model_path = cfg["model"]["base_model_path"]
    logger.info(f"Loading tokenizer from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # ===== dataset =====
    logger.info("Loading datasets...")
    train_dataset = PRMStepDataset(
        cfg["data"]["train_path"],
        tokenizer,
        cfg["data"]["max_length"],
    )
    test_dataset = PRMStepDataset(
        cfg["data"]["test_path"],
        tokenizer,
        cfg["data"]["max_length"],
    )

    logger.info(
        f"Train samples: {len(train_dataset)} | "
        f"Test samples: {len(test_dataset)}"
    )

    collator = PRMCollator(tokenizer.pad_token_id)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collator,
    )

    # ===== model =====
    logger.info("Building model...")

    base_lm = BaseLM(
        model_path=model_path,
        freeze=cfg["model"]["freeze_base"],
    )
    prm_head = PRMHead(
        hidden_size=base_lm.model.config.hidden_size
    )
    model = PRMModel(base_lm, prm_head).to(device)

    logger.info(
        f"Base model frozen: {cfg['model']['freeze_base']} | "
        f"Hidden size: {base_lm.model.config.hidden_size}"
    )

    # ===== optim =====
    loss_fn = PRMLoss()

    optimizer = build_optimizer(
        model,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(train_dataloader) * cfg["training"]["num_epochs"]
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
    )

    logger.info(
        f"Training config: "
        f"epochs={cfg['training']['num_epochs']}, "
        f"lr={cfg['training']['lr']}, "
        f"weight_decay={cfg['training']['weight_decay']}"
    )

    trainer = PRMTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # ===== tensorboard =====
    tb_logger = TensorboardLogger(
        log_dir=cfg["logging"]["tensorboard_dir"]
    )

    # ===== best metric tracker =====
    best_metrics = {
        "first_error_acc": -1.0,
        "step_auc": -1.0,
        "false_early_rate": float("inf"),
        "miss_rate": float("inf"),
    }

    ckpt_dir = cfg["logging"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ===== training loop =====
    all_losses = []

    for epoch in range(cfg["training"]["num_epochs"]):
        logger.info(f"\n===== Epoch {epoch+1} =====")

        train_loss = trainer.train_epoch(
                        train_dataloader,
                        tb_logger=tb_logger,
                        epoch=epoch
                        )

        all_losses.append(train_loss)

        logger.info(f"Train loss: {train_loss:.6f}")
        tb_logger.log_train_loss(train_loss, epoch)

        # ===== evaluation =====
        metrics = evaluate_prm_processbench(
            model=model,
            dataloader=test_dataloader,
            device=device,
            threshold=cfg["evaluate"]["threshold"],
        )

        logger.info("[Eval]")
        logger.info(
            "    %-15s : %.4f", "FirstErrorAcc", metrics["first_error_acc"]
        )
        logger.info(
            "    %-15s : %.4f", "StepAUC", metrics["step_auc"]
        )
        logger.info(
            "    %-15s : %.4f", "FalseEarly", metrics["false_early_rate"]
        )
        logger.info(
            "    %-15s : %.4f", "MissRate", metrics["miss_rate"]
        )


        tb_logger.log_eval_metrics(metrics, epoch)

        # ===== save best checkpoints =====
        logger.info("[BEST]")
        for key in best_metrics:
            better = (
                metrics[key] > best_metrics[key]
                if key in ["first_error_acc", "step_auc"]
                else metrics[key] < best_metrics[key]
            )

            if better:
                old = best_metrics[key]
                best_metrics[key] = metrics[key]

                ckpt_path = os.path.join(
                    ckpt_dir, f"best_{key}.pt"
                )
                torch.save(prm_head.state_dict(), ckpt_path)

                logger.info(
                    "    %-18s : %8.4f → %8.4f | %s",
                    key,
                    old,
                    metrics[key],
                    ckpt_path
                )

                tb_logger.log_best_metric(key, metrics[key], epoch)


    # ===== final save =====
    logger.info("Saving final model and figures...")

    plot_losses(all_losses, cfg["logging"]["loss_figure_path"])

    final_ckpt = os.path.join(ckpt_dir, "prm_head.pt")
    torch.save(prm_head.state_dict(), final_ckpt)

    logger.info(f"Final checkpoint saved to: {final_ckpt}")

    tb_logger.close()
    logger.info("========== Training Finished ==========")


if __name__ == "__main__":
    main()
