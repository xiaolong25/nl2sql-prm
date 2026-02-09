# NL2SQL-PRM

🚀 **NL2SQL with Process Reward Model (PRM)**
一个用于 **NL2SQL 推理过程建模与评估** 的 PRM 训练与评测工程，支持 step-level 监督、ProcessBench 风格评估，以及 TensorBoard 可视化。

---

## 📌 项目简介

在 NL2SQL 任务中，**最终 SQL 是否正确** 并不能完整反映模型的推理能力。
本项目引入 **Process Reward Model (PRM)**，对 **SQL 生成的中间推理步骤（reasoning steps）** 进行逐步打分，用于：

* 评估模型是否在「正确的时间犯错」
* 检测 **early false positive / missed error**
* 支持后续 **PRM-guided decoding / reranking / RL**

核心思想：

> 不只关心「SQL 对不对」，还关心「错在哪里、什么时候开始错」。

---

## 🧠 方法概览

* **Base Model**：冻结的大语言模型（如 Qwen2.5-7B）
* **PRM Head**：轻量级 MLP，对每个 reasoning step 输出 reward / correctness score
* **监督方式**：step-level binary label（是否仍在正确推理轨道）
* **训练目标**：学习从推理轨迹中预测 *first error* 及整体推理质量

---

## 📂 项目结构

```
nl2sql-prm/
├── configs/                 # 训练与评测配置
│   └── train.yaml
├── prm_datasets/            # PRM 数据集与 collator
│   ├── dataset.py
│   └── collator.py
├── models/                  # 模型结构
│   ├── base_lm.py            # 冻结的基础 LM
│   ├── prm_head.py           # PRM 预测头
│   └── prm_model.py
├── losses/                  # PRM loss 定义
│   └── prm_loss.py
├── trainers/                # 训练逻辑
│   └── prm_trainer.py
├── evaluation/              # ProcessBench 风格评测
│   └── evaluate_prm_processbench.py
├── utils/
│   ├── logger.py             # 日志工具
│   ├── tensorboard_logger.py # TensorBoard 可视化
│   ├── seed.py
│   └── visualization.py
├── scripts/
│   └── train_prm.py          # 主训练入口
└── README.md
```

---

## 📊 支持的评测指标

基于 **ProcessBench / PRM 常用指标**：

| 指标                   | 含义                      |
| -------------------- | ----------------------- |
| **FirstErrorAcc**    | 是否准确预测「第一次推理出错的位置」      |
| **Step AUC**         | step-level reward 的排序能力 |
| **False Early Rate** | 过早判错的比例                 |
| **Miss Rate**        | 错误发生但未检测到的比例            |

所有指标均支持 **TensorBoard 可视化**。

---

## ⚙️ 环境依赖

```bash
python >= 3.9
torch >= 2.0
transformers
tqdm
tensorboard
```

推荐使用 Conda：

```bash
conda create -n prm python=3.10
conda activate prm
pip install torch transformers tqdm tensorboard
```

---

## 🚄 快速开始

### 1️⃣ 准备配置

编辑 `configs/train.yaml`：

```yaml
training:
  num_epochs: 5
  lr: 1.0e-7
  weight_decay: 1.0e-2
  warmup_ratio: 0.05

data:
  batch_size: 20
  max_length: 2048
```

---

### 2️⃣ 启动训练

```bash
python scripts/train_prm.py
```

---

### 3️⃣ 启动 TensorBoard

```bash
tensorboard --logdir=./logs/tensorboard
```

你可以看到：

* epoch-level loss
* step-level loss
* PRM 各评测指标变化趋势

---

## 🧪 数据格式说明（简化）

每条样本包含一个完整推理轨迹：

```json
{
  "question": "...",
  "steps": [
    {"text": "Reasoning step 1", "label": 1},
    {"text": "Reasoning step 2", "label": 1},
    {"text": "Reasoning step 3", "label": 0}
  ]
}
```

含义：

* `label = 1`：推理仍在正确轨道
* `label = 0`：从该 step 开始出现错误

---

## 📈 设计上的一些注意点

* 同一条样本中，**相邻 step token overlap 高**
* label 可能呈现 `1 → 1 → 0` 的突变结构
* step-level loss 存在天然抖动，属正常现象
  👉 更应关注 **AUC / FirstErrorAcc 的趋势**

---

## 🔮 可扩展方向

* PRM-guided SQL decoding
* PRM + RLHF / PPO
* 多 PRM ensemble
* PRM 用于 chain-of-thought reranking

---

## 📜 License

License

---

## 🙌 Acknowledgement

* Process Reward Model
* ProcessBench
* NL2SQL / Text-to-SQL 社区

