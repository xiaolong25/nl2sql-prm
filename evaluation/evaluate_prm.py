import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from .metrics import (
    find_first_error_from_labels,
    find_first_error_from_scores,
)


@torch.no_grad()
def evaluate_prm_processbench(
    model,
    dataloader,
    device,
    threshold=0.5,
    writer=None,
    epoch=None,
    cfg=None
):
    """
    ProcessBench-style evaluation for PRM (5-step NL2SQL)

    Each batch corresponds to ONE logical sample with K steps.
    """

    model.eval()

    total_samples = 0           # 总的样本数量
    correct_first_error = 0     # 判对第一个错误步骤的数量，用于计算first_error_acc

    gt_has_error_cnt = 0        # 真实步骤是错误的数量 - 计算recall
    false_early_cnt = 0         # 过早判步骤为错 - 惩罚错了（将正确步骤判错）
    miss_error_cnt = 0          # GT 有错误，但模型完全没检测到

    all_step_scores = []
    all_step_labels = []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluating PRM")):
        """
        Expected batch format:
        {
            "input_ids": [K, L],
            "attention_mask": [K, L],
            "labels": [K]
        }
        """
        # if(step>2):
        #     break
        if not cfg["training"]["train_by_PreProcess_data"]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).long()   # [K]
            # ---- forward ----
            scores = model(input_ids, attention_mask)    # [K] or [K,1]
            scores = scores.squeeze(-1)
        else:
            features = batch["features"].to(device) 
            labels = batch["labels"].to(device).float().view(-1)
            # ---- forward ----
            scores = model(features)
            scores = scores.view(-1)

        # ---- collect for AUC ----
        all_step_scores.extend(scores.cpu().tolist())
        all_step_labels.extend(labels.cpu().tolist())

        # ---- first error ----
        gt_first_error = find_first_error_from_labels(labels.tolist())
        pred_first_error = find_first_error_from_scores(
            scores.tolist(), threshold
        )

        total_samples += 1

        if gt_first_error == pred_first_error:
            correct_first_error += 1

        if gt_first_error is not None:
            gt_has_error_cnt += 1

            if pred_first_error is None:
                miss_error_cnt += 1
            elif pred_first_error < gt_first_error:
                false_early_cnt += 1

    # ---- metrics ----
    first_error_acc = correct_first_error / max(total_samples, 1)

    try:
        step_auc = roc_auc_score(all_step_labels, all_step_scores)
    except ValueError:
        step_auc = float("nan")

    false_early_rate = (
        false_early_cnt / gt_has_error_cnt
        if gt_has_error_cnt > 0 else 0.0
    )

    miss_rate = (
        miss_error_cnt / gt_has_error_cnt
        if gt_has_error_cnt > 0 else 0.0
    )

    metrics = {
        "first_error_acc": first_error_acc,
        "step_auc": step_auc,
        "false_early_rate": false_early_rate,
        "miss_rate": miss_rate,
        "num_samples": total_samples,
    }

    # ---- TensorBoard logging ----
    if writer is not None and epoch is not None:
        writer.add_scalar("eval/step_auc", step_auc, epoch)
        writer.add_scalar("eval/first_error_acc", first_error_acc, epoch)
        writer.add_scalar("eval/false_early_rate", false_early_rate, epoch)
        writer.add_scalar("eval/miss_rate", miss_rate, epoch)
        writer.add_scalar("eval/num_samples", total_samples, epoch)

    return metrics
