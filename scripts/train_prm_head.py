# ================================================================
# File: train_prm_head.py
# Author: Xiaolong Ji
# Date: 2026/02/09
#
# Description:
#   Training PRMHead using pre-extracted features and evaluating PRM metrics.
# ================================================================

import os
import sys
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.tensorboard_logger import TensorboardLogger
from utils.visualization import plot_losses
from losses import PRMLoss
from optim import build_optimizer, build_scheduler
from models import PRMHead, BaseLM
from prm_datasets.prm_feature_dataset import PRMFeatureDataset

# ===== first_error helper =====
def find_first_error_from_labels(labels):
    for i, l in enumerate(labels):
        if l == 0:
            return i
    return None

def find_first_error_from_scores(scores, threshold=0.5):
    for i, s in enumerate(scores):
        if s < threshold:
            return i
    return None

# ===== metrics for ProcessBench =====
def compute_prm_metrics(model, dataloader, device, threshold=0.5):
    """
    计算 PRM 指标，适配 step 级别特征。
    model: PRMHead
    dataloader: DataLoader, 每个 batch 是多步 step 的 feature
    device: torch.device
    threshold: float, 预测第一错误的阈值
    """
    model.eval()

    total_samples = 0            # 用于 first-error 计数（按 chain）
    correct_first_error = 0
    gt_has_error_cnt = 0
    false_early_cnt = 0
    miss_error_cnt = 0
    all_step_scores = []
    all_step_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PRM"):
            # batch features: [B, H], labels: [B]
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            scores = model(features).squeeze(-1)  # [B]
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)

            all_step_scores.extend(scores.cpu().tolist())
            all_step_labels.extend(labels.cpu().tolist())

            # ===== 如果有 chain 信息可以按 chain 计算 first-error =====
            # 暂时假设每个 batch 是一条 chain
            labels_list = labels.cpu().tolist()
            scores_list = scores.cpu().tolist()

            gt_first_error = find_first_error_from_labels(labels_list)
            pred_first_error = find_first_error_from_scores(scores_list, threshold)

            total_samples += 1
            if gt_first_error == pred_first_error:
                correct_first_error += 1

            if gt_first_error is not None:
                gt_has_error_cnt += 1
                if pred_first_error is None:
                    miss_error_cnt += 1
                elif pred_first_error < gt_first_error:
                    false_early_cnt += 1

    # ===== 计算指标 =====
    first_error_acc = correct_first_error / max(total_samples, 1)
    false_early_rate = false_early_cnt / gt_has_error_cnt if gt_has_error_cnt > 0 else 0.0
    miss_rate = miss_error_cnt / gt_has_error_cnt if gt_has_error_cnt > 0 else 0.0

    try:
        step_auc = roc_auc_score(all_step_labels, all_step_scores)
    except ValueError:
        step_auc = float("nan")

    metrics = {
        "first_error_acc": first_error_acc,
        "step_auc": step_auc,
        "false_early_rate": false_early_rate,
        "miss_rate": miss_rate,
        "num_samples": total_samples,
        "num_steps": len(all_step_labels),
    }

    return metrics

# ===== main training =====
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
    return {
        "out_put_path": exp_root,
        "loss_figure_path": loss_figure_path,
        "checkpoint_dir": checkpoint_dir,
        "log_file": log_file,
        "tensorboard_dir": tensorboard_dir,
    }

def main():
    cfg = load_config("configs/train.yaml")
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths_dict = build_logging_paths(cfg)
    logger = setup_logger(paths_dict["log_file"])
    logger.info("========== Start PRMHead Training ==========")
    logger.info(f"Output path: {paths_dict['out_put_path']} | Device: {device}")

    # ===== datasets =====
    train_dataset = PRMFeatureDataset("/root/autodl-tmp/git/nl2sql_prm/data/without_qwen/prm_train.pt")
    eval_dataset = PRMFeatureDataset("/root/autodl-tmp/git/nl2sql_prm/data/without_qwen/prm_test_2.pt")

    train_loader = DataLoader(train_dataset, batch_size=cfg["data"]["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)  # step-level eval

    logger.info(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    # ===== model =====
    base_lm = BaseLM(model_path=cfg["model"]["base_model_path"], freeze=True)
    hidden_size = base_lm.model.config.hidden_size
    prm_head = PRMHead(hidden_size=hidden_size).to(device)

    # ===== optimizer =====
    loss_fn = PRMLoss()
    optimizer = build_optimizer(prm_head, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    total_steps = len(train_loader) * cfg["training"]["num_epochs"]
    scheduler = build_scheduler(optimizer, total_steps=total_steps)

    tb_logger = TensorboardLogger(paths_dict["tensorboard_dir"])
    all_losses = []
    best_step_auc = -float("inf")

    for epoch in range(cfg["training"]["num_epochs"]):
        prm_head.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            features = batch["features"].to(device)
            labels = batch["labels"].to(device).float()

            optimizer.zero_grad()
            logits = prm_head(features).squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * features.size(0)

        epoch_loss /= len(train_loader.dataset)
        all_losses.append(epoch_loss)
        tb_logger.log_train_loss(epoch_loss, epoch)
        logger.info(f"Epoch {epoch+1} | Train loss: {epoch_loss:.6f}")

        # ===== eval =====
        metrics = compute_prm_metrics(prm_head, eval_loader, device, threshold=cfg["evaluate"]["threshold"])
        tb_logger.log_eval_metrics(metrics, epoch)

        logger.info(f"[Eval Epoch {epoch+1}]")
        logger.info(f"  Num samples   : {metrics['num_samples']}")
        logger.info(f"  FirstErrorAcc : {metrics['first_error_acc']:.4f}")
        logger.info(f"  StepAUC       : {metrics['step_auc']:.4f}")
        logger.info(f"  FalseEarly    : {metrics['false_early_rate']:.4f}")
        logger.info(f"  MissRate      : {metrics['miss_rate']:.4f}")

        # ===== save best checkpoint =====
        if metrics["step_auc"] > best_step_auc:
            best_step_auc = metrics["step_auc"]
            ckpt_path = os.path.join(paths_dict["checkpoint_dir"], "best_prm_head.pt")
            torch.save(prm_head.state_dict(), ckpt_path)
            logger.info(f"Saved best PRMHead checkpoint: {ckpt_path}")

    # ===== final save =====
    final_ckpt = os.path.join(paths_dict["checkpoint_dir"], "final_prm_head.pt")
    torch.save(prm_head.state_dict(), final_ckpt)
    plot_losses(all_losses, paths_dict["loss_figure_path"])
    tb_logger.close()
    logger.info("========== PRMHead Training Finished ==========")

if __name__ == "__main__":
    main()
