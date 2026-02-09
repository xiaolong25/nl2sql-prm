# utils/tensorboard_logger.py

from torch.utils.tensorboard import SummaryWriter
import os


class TensorboardLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    # ===== epoch-level =====
    def log_train_loss(self, loss, epoch):
        self.writer.add_scalar("train/loss_epoch", loss, epoch)

    # ===== step-level =====
    def log_train_step_loss(self, loss, global_step):
        """
        每个 optimizer step 的 loss
        global_step = epoch * steps_per_epoch + step
        """
        self.writer.add_scalar("train/loss_step", loss, global_step)

    # ===== eval metrics =====
    def log_eval_metrics(self, metrics: dict, epoch):
        """
        metrics: output of evaluate_prm_processbench
        """
        self.writer.add_scalar(
            "eval/first_error_acc",
            metrics["first_error_acc"],
            epoch
        )
        self.writer.add_scalar(
            "eval/step_auc",
            metrics["step_auc"],
            epoch
        )
        self.writer.add_scalar(
            "eval/false_early_rate",
            metrics["false_early_rate"],
            epoch
        )
        self.writer.add_scalar(
            "eval/miss_rate",
            metrics["miss_rate"],
            epoch
        )
        
    def log_best_metric(self, metric_name: str, value: float, step: int):
        """
        用于记录“当前最优”的指标（只在 best 时调用）
        TensorBoard 中会单独形成一条曲线
        """
        self.writer.add_scalar(
            f"best/{metric_name}",
            value,
            step
        )

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
