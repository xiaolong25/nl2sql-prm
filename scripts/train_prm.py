# ================================================================
# File: train_prm.py
# Author: Xiaolong Ji
# Date: 2026/02/06
#
# Description:
#   Main training entry for NL2SQL Process Reward Model (PRM).
# ================================================================


import torch
from datetime import datetime
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 暂时关掉多线程tokenizer，日志太吵

from prm_datasets import PRMStepDataset, PRMCollator, PRMChainEvalDataset, PRMEvalCollator, PRMFeatureDataset, PRMFeatureCollator
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

def build_logging_paths(cfg):
    now = datetime.now()
    time_tag = now.strftime("%Y%m%d_%H_%M_%S")

    exp_root = os.path.join(cfg["logging"]["out_put_path"], time_tag[2:])

    checkpoint_dir = os.path.join(exp_root, cfg["logging"]["checkpoint_dir"])
    tensorboard_dir = os.path.join(exp_root, cfg["logging"]["tensorboard_dir"])
    log_file = os.path.join(exp_root, cfg["logging"]["log_file"])
    loss_figure_path = os.path.join(exp_root, cfg["logging"]["loss_figure_path"])

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    logging_cfg = {
        "out_put_path": exp_root,
        "loss_figure_path": loss_figure_path,
        "checkpoint_dir": checkpoint_dir,
        "log_file": log_file,
        "tensorboard_dir": tensorboard_dir,
    }

    return logging_cfg

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"    
    # TODO
    #   支持 DP\DDP
    cfg = load_config("configs/train.yaml")
    paths_dict = build_logging_paths(cfg)
    # ===== logger =====
    logger = setup_logger(paths_dict["log_file"])
    logger.info("========== Start PRM Training ==========")
    logger.info(f"==> Saving at: {paths_dict['out_put_path']}")

    
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
    if not cfg["training"]["train_by_PreProcess_data"]:     # 正常通过basemodel来训练和测试
        train_dataset = PRMStepDataset(
            cfg["data"]["train_path"],
            tokenizer,
            cfg["data"]["max_length"],
        )
        test_dataset = PRMChainEvalDataset(
            cfg["data"]["test_path"],
            tokenizer,
            cfg["data"]["max_length"],
        )
        collator = PRMCollator(tokenizer.pad_token_id)
        test_collator = PRMEvalCollator(tokenizer.pad_token_id)
    else:                                                   # 不通过basemodel来训练和测试
        train_dataset = PRMFeatureDataset(cfg["pre_process"]["train_pt_save_path"])
        test_dataset = PRMFeatureDataset(cfg["pre_process"]["test_pt_save_path"])
        collator = PRMFeatureCollator()
        test_collator = PRMFeatureCollator()
    
    logger.info(
        f"Train samples: {len(train_dataset)} | "
        f"Test samples: {len(test_dataset)}"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["data"]["eval_batch"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=test_collator,
    )

    # ===== model =====
    logger.info("Building model...")
    base_lm = BaseLM(
        model_path=model_path,
        freeze=cfg["model"]["freeze_base"],
    )
    if not cfg["training"]["train_by_PreProcess_data"]:     # 正常通过basemodel来训练和测试
        logger.info(f"Mode: With BaseModel!")
        prm_head = PRMHead(
            hidden_size=base_lm.model.config.hidden_size
        )
        model = PRMModel(base_lm, prm_head).to(device)
    else:                                                   # 不通过basemodel来训练和测试
        logger.info(f"==> Mode: Without BaseModel! <==")
        hidden_size = base_lm.model.config.hidden_size
        model = PRMHead(hidden_size=hidden_size).to(device)

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
        log_dir=paths_dict["tensorboard_dir"]
    )

    # ===== best metric tracker =====
    best_metrics = {
        "first_error_acc": -1.0,
        "step_auc": -1.0,
        "false_early_rate": float("inf"),
        "miss_rate": float("inf"),
    }

    ckpt_dir = paths_dict["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ===== training loop =====
    all_losses = []

    for epoch in range(cfg["training"]["num_epochs"]):
        logger.info(f"\n===== Epoch {epoch+1} =====")

        train_loss = trainer.train_epoch(
                        train_dataloader,
                        tb_logger=tb_logger,
                        epoch=epoch,
                        cfg=cfg
                        )

        all_losses.append(train_loss)

        logger.info(f"Train loss: {train_loss:.6f}")
        tb_logger.log_train_loss(train_loss, epoch)

        # ===== evaluation =====
        # if not cfg["training"]["train_by_PreProcess_data"]:  # 正常通过basemodel来训练和测试
        metrics = evaluate_prm_processbench(
            model=model,
            dataloader=test_dataloader,
            device=device,
            threshold=cfg["evaluate"]["threshold"],
            cfg=cfg
        )
        # else:                                               # 不通过basemodel来训练和测试
        #     metrics = evaluate_prm_processbench_withoutBaseModel(
        #         model=model,
        #         dataloader=test_dataloader,
        #         device=device,
        #         threshold=cfg["evaluate"]["threshold"],
        #     )

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

            old = best_metrics[key]  # 每次循环都取旧值

            better = (
                metrics[key] > old
                if key in ["first_error_acc", "step_auc"]
                else metrics[key] < old
            )
            ckpt_path = os.path.join(
                ckpt_dir, f"best_{key}.pt"
            )

            if better:
                best_metrics[key] = metrics[key]
                if not cfg["training"]["train_by_PreProcess_data"]:
                    torch.save(prm_head.state_dict(), ckpt_path)
                else:
                    torch.save(model.state_dict(), ckpt_path)
                logger.info(
                    "    %-18s : %8.4f → %8.4f | %s",
                    key,
                    old,
                    metrics[key],
                    ckpt_path
                )

            else:
                logger.info(
                    "    %-18s : %8.4f → %8.4f | -",
                    key,
                    old,
                    old
                )

            tb_logger.log_best_metric(key, best_metrics[key], epoch)


    # ===== final save =====
    logger.info("Saving final model and figures...")

    plot_losses(all_losses, paths_dict["loss_figure_path"])

    final_ckpt = os.path.join(ckpt_dir, "prm_head.pt")
    if not cfg["training"]["train_by_PreProcess_data"]:  # 正常通过basemodel来训练和测试
        torch.save(prm_head.state_dict(), final_ckpt)
    else:
        torch.save(model.state_dict(), final_ckpt)
    logger.info(f"Final checkpoint saved to: {final_ckpt}")

    tb_logger.close()
    logger.info("========== Training Finished ==========")


if __name__ == "__main__":
    main()
