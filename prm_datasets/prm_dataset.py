import json
from torch.utils.data import Dataset


class PRMStepDataset(Dataset):
    """
    Each item corresponds to ONE reasoning step:
    Input: Q + Schema + previous steps + current step
    Label: correctness of the current step
    """

    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = []  # step-level samples

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data:
            self._process_example(ex)

    def _process_example(self, ex):
        question = ex["question"]
        schema = ex["schema"]
        reasoning = ex["reasoning"]
        analysis = ex["analysis"]

        # 保证 step 顺序
        step_keys = sorted(
            reasoning.keys(),
            key=lambda x: int(x.split("_")[1])
        )

        accumulated_steps = []

        for step_key in step_keys:
            step_id = int(step_key.split("_")[1])
            step_text = reasoning[step_key]

            accumulated_steps.append(step_text)

            # 取对应的 verification result
            analysis_key = f"analysis_step_{step_id}"
            verification = analysis[analysis_key]["verification_result"]

            label = 1.0 if verification == "Correct" else 0.0

            prompt = self.build_prompt(
                question=question,
                schema=schema,
                steps=accumulated_steps
            )

            self.samples.append({
                "input_text": prompt,
                "label": label,
                "step_id": step_id
            })

    def build_prompt(self, question, schema, steps):
        """
        Q + Schema + Step_1 ... Step_k
        """
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        enc = self.tokenizer(
            s["input_text"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": s["label"],
            "step_id": s["step_id"]
        }


class PRMChainEvalDataset(Dataset):
    """
    Each item corresponds to ONE full reasoning chain (one logical sample).

    Output:
        {
            "input_ids":      List[Tensor]  # length K
            "attention_mask": List[Tensor]  # length K
            "labels":         List[int]     # length K
        }
    """

    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data:
            self.samples.append(self._process_example(ex))

    def _process_example(self, ex):
        question = ex["question"]
        schema = ex["schema"]
        reasoning = ex["reasoning"]
        analysis = ex["analysis"]

        step_keys = sorted(
            reasoning.keys(),
            key=lambda x: int(x.split("_")[1])
        )

        step_input_ids = []
        step_attention_masks = []
        step_labels = []

        accumulated_steps = []

        for step_key in step_keys:
            step_id = int(step_key.split("_")[1])
            step_text = reasoning[step_key]
            accumulated_steps.append(step_text)

            # label
            analysis_key = f"analysis_step_{step_id}"
            verification = analysis[analysis_key]["verification_result"]
            label = 1 if verification == "Correct" else 0
            step_labels.append(label)

            # prompt (和训练时完全一致)
            prompt = self.build_prompt(
                question=question,
                schema=schema,
                steps=accumulated_steps
            )

            enc = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            step_input_ids.append(enc["input_ids"][0])
            step_attention_masks.append(enc["attention_mask"][0])

        return {
            "input_ids": step_input_ids,            # List[Tensor]
            "attention_mask": step_attention_masks, # List[Tensor]
            "labels": step_labels                   # List[int]
        }

    def build_prompt(self, question, schema, steps):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]