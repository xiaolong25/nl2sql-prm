import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def build_prompt(question, schema, steps):
    prompt = (
        "### Question\n"
        f"{question}\n\n"
        "### Schema\n"
        f"{schema}\n\n"
        "### Reasoning So Far\n"
    )
    for i, step in enumerate(steps, start=1):
        prompt += f"Step {i}:\n{step}\n\n"
    return prompt.strip()


def analyze_prm_step_dataset(data_path, tokenizer, max_ctx):
    """
    对应 PRMStepDataset 的真实 token 长度分析（不截断）
    """
    data = json.load(open(data_path, "r", encoding="utf-8"))

    token_lens = []
    over_cnt = 0

    for ex in tqdm(data, desc="Analyzing PRMStepDataset"):
        question = ex["question"]
        schema = ex["schema"]
        reasoning = ex["reasoning"]

        step_keys = sorted(
            reasoning.keys(),
            key=lambda x: int(x.split("_")[1])
        )

        accumulated_steps = []

        for step_key in step_keys:
            accumulated_steps.append(reasoning[step_key])

            prompt = build_prompt(question, schema, accumulated_steps)

            # ⚠️ 不截断，统计真实长度
            ids = tokenizer(prompt, truncation=False)["input_ids"]
            L = len(ids)

            token_lens.append(L)
            if L > max_ctx:
                over_cnt += 1

    arr = np.array(token_lens)

    print("\n===== PRMStepDataset (TRAIN) =====")
    print(f"Num step samples: {len(arr)}")
    print(f"Max token length: {arr.max()}")
    print(f"Mean token length: {arr.mean():.1f}")
    print(f"P95 token length: {np.percentile(arr, 95):.0f}")
    print(f"P99 token length: {np.percentile(arr, 99):.0f}")
    print(
        f"> {max_ctx} tokens: {over_cnt} "
        f"({over_cnt / len(arr) * 100:.2f}%)"
    )


def analyze_prm_chain_eval_dataset(data_path, tokenizer, max_ctx):
    """
    对应 PRMChainEvalDataset（ProcessBench-style eval）
    """
    data = json.load(open(data_path, "r", encoding="utf-8"))

    step_lens = []
    sample_max_lens = []
    over_sample_cnt = 0

    for ex in tqdm(data, desc="Analyzing PRMChainEvalDataset"):
        question = ex["question"]
        schema = ex["schema"]
        reasoning = ex["reasoning"]

        step_keys = sorted(
            reasoning.keys(),
            key=lambda x: int(x.split("_")[1])
        )

        accumulated_steps = []
        cur_step_lens = []

        for step_key in step_keys:
            accumulated_steps.append(reasoning[step_key])

            prompt = build_prompt(question, schema, accumulated_steps)
            ids = tokenizer(prompt, truncation=False)["input_ids"]
            L = len(ids)

            step_lens.append(L)
            cur_step_lens.append(L)

        max_L = max(cur_step_lens)
        sample_max_lens.append(max_L)

        if max_L > max_ctx:
            over_sample_cnt += 1

    step_arr = np.array(step_lens)
    sample_arr = np.array(sample_max_lens)

    print("\n===== PRMChainEvalDataset (EVAL) =====")
    print(f"Num logical samples: {len(sample_arr)}")

    print("\n[Per-step token length]")
    print(f"Max: {step_arr.max()}")
    print(f"P95: {np.percentile(step_arr, 95):.0f}")
    print(f"P99: {np.percentile(step_arr, 99):.0f}")

    print("\n[Per-sample max step length]")
    print(f"Max: {sample_arr.max()}")
    print(f"P95: {np.percentile(sample_arr, 95):.0f}")
    print(f"P99: {np.percentile(sample_arr, 99):.0f}")

    print(
        f"\nSamples with max_step_len > {max_ctx}: "
        f"{over_sample_cnt} / {len(sample_arr)} "
        f"({over_sample_cnt / len(sample_arr) * 100:.2f}%)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--eval_json", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_ctx", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True
    )

    analyze_prm_step_dataset(
        args.train_json, tokenizer, args.max_ctx
    )
    analyze_prm_chain_eval_dataset(
        args.eval_json, tokenizer, args.max_ctx
    )


if __name__ == "__main__":
    main()

    '''
    python ./judge_token_lenth.py \
        --train_json /root/autodl-tmp/git/nl2sql_prm/data/processed/test/train.json \
        --eval_json /root/autodl-tmp/git/nl2sql_prm/data/processed/test/test.json \
        --model_name /root/autodl-tmp/pretrain_models/Public_Models/Qwen2.5-7B-Instruct \
        --max_ctx 2048
    '''
