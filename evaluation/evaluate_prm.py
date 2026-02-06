# evaluation/evaluate_prm.py

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
):
    """
    ProcessBench-style evaluation for PRM (5-step NL2SQL)

    Each batch corresponds to ONE logical sample with K steps.
    """

    model.eval()

    total_samples = 0
    correct_first_error = 0

    gt_has_error_cnt = 0
    false_early_cnt = 0
    miss_error_cnt = 0

    all_step_scores = []
    all_step_labels = []

    for batch in tqdm(dataloader, desc="Evaluating PRM"):
        """
        Expected batch format:
        {
            "input_ids": [K, L],
            "attention_mask": [K, L],
            "labels": [K]
        }
        """

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).long()   # [K]

        # ---- forward ----
        scores = model(input_ids, attention_mask)    # [K] or [K,1]
        scores = scores.squeeze(-1)

        # ---- collect for AUC ----
        all_step_scores.extend(scores.cpu().tolist())
        all_step_labels.extend(labels.cpu().tolist())

        # ---- first error ----
        gt_first_error = find_first_error_from_labels(labels.tolist())
        pred_first_error = find_first_error_from_scores(
            scores.tolist(), threshold
        )

        total_samples += 1

        # First Error Accuracy
        if gt_first_error == pred_first_error:
            correct_first_error += 1

        # Error-only metrics
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

    return {
        "first_error_acc": first_error_acc,
        "step_auc": step_auc,
        "false_early_rate": false_early_rate,
        "miss_rate": miss_rate,
        "num_samples": total_samples,
    }
