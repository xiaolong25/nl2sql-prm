import os
import torch
import sys
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from prm_datasets import PRMStepDataset, PRMCollator, PRMChainEvalDataset, PRMEvalCollator
from models import BaseLM
from utils.config import load_config
from utils.seed import set_seed


@torch.no_grad()
def extract_features(train=True, test=True):
    cfg = load_config("configs/train.yaml")
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = cfg["model"]["base_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_lm = BaseLM(model_path=model_path, freeze=True).to(device)
    base_lm.eval()

    # ================= TRAIN =================
    if train:
        print("Extracting TRAIN features...")
        dataset = PRMStepDataset(
            cfg["data"]["train_path"],
            tokenizer,
            cfg["data"]["max_length"],
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=False,
            collate_fn=PRMCollator(tokenizer.pad_token_id),
        )

        all_features, all_labels = [], []
        for batch in tqdm(dataloader, desc="Train feature extraction"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            features = base_lm(input_ids, attention_mask)  # [B, H]
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        torch.save(
            {"features": features, "labels": labels},
            cfg["pre_process"]["train_pt_save_path"]
        )
        print("Saved TRAIN features:", features.shape, labels.shape)

    # ================= TEST =================
    if test:
        print("Extracting TEST features...")
        dataset = PRMChainEvalDataset(
            cfg["data"]["test_path"],
            tokenizer,
            cfg["data"]["max_length"],
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg["data"]["eval_batch"],  # 必须 1，保证每个 batch 是一条完整 chain
            shuffle=False,
            collate_fn=PRMEvalCollator(tokenizer.pad_token_id),
        )

        all_features, all_labels = [], []

        for batch in tqdm(dataloader, desc="Test feature extraction"):
            # batch size = 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            features = base_lm(input_ids, attention_mask)  # [B, H]
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

        features = torch.cat(all_features, dim=0)  # [总样本数, H]
        labels = torch.cat(all_labels, dim=0)      # [总样本数]

        torch.save(
            {"features": features, "labels": labels},
            cfg["pre_process"]["test_pt_save_path"]
        )
        print("Saved TEST features:", features.shape, labels.shape)


if __name__ == "__main__":
    extract_features(train=True, test=True)
