import json
import random
import os

# ===============================
# 配置区
# ===============================
INPUT_JSON = "/root/autodl-tmp/git/nl2sql_prm/data/processed/BIRD_with_qwen_reasoning_prefix_eval_v3.json"
OUTPUT_DIR = "/root/autodl-tmp/git/nl2sql_prm/data/processed/test3"

TRAIN_SIZE = 800
TEST_SIZE = 200
TOTAL_SIZE = TRAIN_SIZE + TEST_SIZE

RANDOM_SEED = 42

# ===============================
# 主逻辑
# ===============================
def main():
    random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 读取原始数据
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Input JSON must be a list of samples"

    total_samples = len(data)
    print(f"Total samples in file: {total_samples}")

    if total_samples < TOTAL_SIZE:
        raise ValueError(
            f"Not enough samples: required {TOTAL_SIZE}, but got {total_samples}"
        )

    # 2. 随机打乱
    random.shuffle(data)

    # 3. 抽取 800 条
    selected = data[:TOTAL_SIZE]

    train_data = selected[:TRAIN_SIZE]
    test_data = selected[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

    # 4. 保存
    train_path = os.path.join(OUTPUT_DIR, "train.json")
    test_path = os.path.join(OUTPUT_DIR, "test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print("Split finished:")
    print(f"  Train: {len(train_data)} samples → {train_path}")
    print(f"  Test : {len(test_data)} samples → {test_path}")
    print(f"  Random seed: {RANDOM_SEED}")


if __name__ == "__main__":
    main()
