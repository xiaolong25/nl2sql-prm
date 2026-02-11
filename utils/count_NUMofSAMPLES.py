import json
from collections import defaultdict

def analyze_prm_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)

    # ===== sample-level =====
    samples_with_error = 0
    samples_all_correct = 0

    # ===== step-level =====
    step_correct_cnt = defaultdict(int)
    step_error_cnt = defaultdict(int)

    total_step_correct = 0
    total_step_error = 0

    for ex in data:
        analysis = ex.get("analysis", {})

        has_error = False

        for k, v in analysis.items():
            if not k.startswith("analysis_step_"):
                continue

            step_id = k.replace("analysis_step_", "")
            result = v.get("verification_result")

            if result == "Correct":
                step_correct_cnt[step_id] += 1
                total_step_correct += 1
            else:
                step_error_cnt[step_id] += 1
                total_step_error += 1
                has_error = True

        if has_error:
            samples_with_error += 1
        else:
            samples_all_correct += 1

    # ===== 打印统计结果 =====
    print("\n========== PRM JSON Statistics ==========")
    print(f"Total samples           : {total_samples}")
    print(f"Samples with any error  : {samples_with_error}")
    print(f"Samples all correct     : {samples_all_correct}")

    print("\n----- Step-level breakdown -----")
    all_steps = sorted(
        set(step_correct_cnt.keys()) | set(step_error_cnt.keys()),
        key=lambda x: int(x)
    )

    for step_id in all_steps:
        c = step_correct_cnt[step_id]
        e = step_error_cnt[step_id]
        total = c + e
        err_rate = e / total if total > 0 else 0.0

        print(
            f"Step {step_id}: "
            f"Correct={c}, Incorrect={e}, "
            f"ErrorRate={err_rate:.2%}"
        )

    print("\n----- Global step stats -----")
    print(f"Total correct steps     : {total_step_correct}")
    print(f"Total incorrect steps   : {total_step_error}")
    print(
        f"Overall step error rate : "
        f"{total_step_error / max(total_step_correct + total_step_error, 1):.2%}"
    )

    print("========================================\n")


if __name__ == "__main__":
    json_path = "/root/autodl-tmp/git/nl2sql_prm/data/processed/BIRD_with_qwen_reasoning_prefix_eval_v3.json"
    analyze_prm_json(json_path)
