# MiniSearch-R1 项目 AI Agent 指引文档

> 本文档同时作为 `CLAUDE.md`（Claude Code）和 `AGENTS.md`（Codex）使用。
> 任何 AI 编码 Agent 在本项目中执行任务前，**必须先阅读本文档全文**。

---

## 一、项目全局定义（不可偏离）

### 1.1 一句话定义

MiniSearch-R1 是一个**搜索增强推理 Agent**：在 Qwen2.5-1.5B-Instruct 上，通过 GRPO 强化学习（DAPO loss 变体），训练模型学会在推理过程中**自主决定何时搜索、搜索什么、如何利用搜索结果**，最终在多跳 QA 任务上提升准确率。

### 1.2 对标论文（权威参考源）

| 论文 | 作用 | 关键点 |
|------|------|--------|
| **Search-R1** (arXiv:2503.09516) | 主复现论文 | 多轮搜索交互 + GRPO + retrieved token masking |
| **R1-Searcher** (arXiv:2503.05592) | 辅助参考 | 两阶段 RL + 极少量训练数据(350条) |
| **DAPO** (arXiv:2503.14476) | RL 算法改进 | Clip-Higher + Dynamic Sampling + Token-level Loss + Overlong Filtering |

### 1.3 硬约束（红线，任何代码修改都不能违反）

```
基座模型:       Qwen2.5-1.5B-Instruct（不要换成其他模型）
量化方式:       4-bit QLoRA（r=16, lora_alpha=16）
RL 算法:        GRPO with DAPO loss（loss_type="dapo"）
训练框架:       Unsloth + TRL GRPOTrainer（不用 veRL）
检索方案:       本地 Wikipedia BM25 + E5-base-v2 + RRF 融合
训练硬件:       单卡 RTX 3090 24GB（AutoDL）
总预算:         ≤ ¥100 GPU 费用
显存上限:       24GB（所有配置必须在此范围内运行）
Python 版本:    3.10
```

### 1.4 不做什么（负面清单）

- **不要**使用 veRL / Ray 分布式训练（我们是单卡）
- **不要**使用 PPO（需要额外 Value Model，显存不够）
- **不要**调用外部搜索 API（如 Google/Bing）——只用本地检索服务器
- **不要**使用 LLM-as-Judge 做 reward（成本不可控）
- **不要**训练独立的 Reward Model 或 Process Reward Model
- **不要**修改 Qwen2.5-1.5B 的基座权重（只训练 LoRA）
- **不要**使用 `localStorage`/`sessionStorage`（如涉及前端）
- **不要**安装 apex 或其他非必要的 CUDA 编译包

---

## 二、项目目录结构（严格遵守）

```
MiniSearch-R1/
├── AGENTS.md                    # ← 本文件（Claude Code / Codex 指引）
├── README.md                    # 项目介绍
├── requirements.txt             # Python 依赖
├── .gitignore
│
├── configs/                     # 所有超参数配置
│   ├── sft_config.yaml          # SFT 冷启动配置
│   ├── grpo_stage1.yaml         # GRPO 阶段1（简单）
│   ├── grpo_stage2.yaml         # GRPO 阶段2（中等）
│   └── grpo_stage3.yaml         # GRPO 阶段3（困难）
│
├── data/                        # 数据处理
│   ├── download_datasets.py     # 下载 HotpotQA/MuSiQue/2Wiki/Bamboogle
│   ├── preprocess.py            # 统一格式化为 Search-R1 parquet 格式
│   ├── curriculum_split.py      # 按难度分三级课程
│   └── generate_sft_data.py     # 用 DeepSeek API 生成 SFT 冷启动轨迹
│
├── retrieval/                   # 检索服务
│   ├── build_bm25_index.py      # Pyserini BM25 索引构建
│   ├── build_dense_index.py     # E5-base-v2 FAISS 索引构建
│   ├── server.py                # FastAPI 混合检索服务器（BM25+E5+RRF）
│   └── test_retrieval.py        # 检索服务单元测试
│
├── training/                    # 训练代码
│   ├── sft_train.py             # SFT 冷启动训练
│   ├── grpo_train.py            # GRPO/DAPO 主训练脚本
│   ├── search_env.py            # 多轮搜索环境（environment_factory）
│   ├── rewards.py               # 所有 reward 函数
│   └── utils.py                 # 工具函数（token masking 等）
│
├── evaluation/                  # 评测代码
│   ├── evaluate.py              # 统一评测脚本
│   ├── metrics.py               # EM / F1 / 搜索效率指标
│   ├── ablation_runner.py       # 消融实验批量运行器
│   └── case_analysis.py         # 定性案例分析
│
├── scripts/                     # 运维脚本
│   ├── setup_autodl.sh          # AutoDL 环境一键部署
│   ├── start_retrieval.sh       # 启动检索服务器
│   ├── run_full_pipeline.sh     # 完整 pipeline 一键运行
│   └── auto_shutdown.py         # 训练完自动关机（省预算）
│
└── outputs/                     # 训练产出（git ignore）
    ├── sft_checkpoint/
    ├── grpo_stage1/
    ├── grpo_stage2/
    ├── grpo_stage3/
    ├── eval_results/
    └── ablation_results/
```

**规则**：新建文件必须放在对应目录下，不要在根目录散落脚本。

---

## 三、核心技术规范

### 3.1 搜索交互协议（标签格式）

模型在推理过程中使用以下特殊标签与检索环境交互：

```
<think>我需要查找关于 X 的信息</think>
<search>X related query</search>
<information>
[1] 检索结果文本 1...
[2] 检索结果文本 2...
[3] 检索结果文本 3...
</information>
<think>根据检索结果，我现在知道...</think>
<answer>最终答案</answer>
```

**关键规则**：
- `<search>` 和 `</search>` 之间是模型生成的查询，必须是**单行纯文本**
- `<information>` 和 `</information>` 之间是**环境注入的检索结果**，不是模型生成的
- `<answer>` 和 `</answer>` 之间是**最终答案**，必须简洁（通常是一个实体/短语）
- 最大搜索轮次：**3 次**（超过后强制要求回答）
- 每次检索返回 **top-3** passages，每条最多 **500 字符**

### 3.2 System Prompt（不要修改）

```python
SYSTEM_PROMPT = """Answer the given question. You must conduct reasoning inside <think> \
and </think> first every time you get new information. After reasoning, if you find you \
lack some knowledge, you can call a search engine by <search> query </search>, and it \
will return the top searched results between <information> and </information>. You can \
search as many times as you want. If you find no further external knowledge needed, you \
can directly provide the answer inside <answer> and </answer> without detailed \
illustrations. For example, <answer> xxx </answer>.
Question: {question}"""
```

### 3.3 Retrieved Token Masking 规范

在 RL loss 计算中，`<information>...</information>` 标签内的所有 token 的 loss mask 必须设为 0：

```python
# 正确 ✅：只对模型生成的 token 计算梯度
active_mask = completion_mask * retrieval_mask  # 两者都为 1 才有效
loss = masked_loss.sum() / (active_mask.sum() + 1e-8)

# 错误 ❌：对全序列计算梯度（包括检索内容）
loss = total_loss.mean()
```

### 3.4 Reward 函数规范

```python
# 主 reward：F1 Score（连续值 0-1，优于二值 EM）
def f1_reward(completions, ground_truth, **kwargs) -> list[float]:
    # 必须使用 SQuAD 标准归一化：lowercase + 去冠词 + 去标点 + 去多余空格
    ...

# 辅助 reward 1：格式正确性（二值 0/1）
def format_reward(completions, **kwargs) -> list[float]:
    # 检查 <answer> 标签是否存在且 <search> 标签是否成对
    ...

# 辅助 reward 2：搜索质量（0/0.1/0.3）
def search_quality_reward(completions, ground_truth, **kwargs) -> list[float]:
    # 答对 + 搜索次数 ≤ 3 → 0.3；答对 + 搜索次数 > 3 → 0.1；答错 → 0
    ...

# 组合权重
reward_weights = [1.0, 0.2, 0.1]  # [f1, format, search_quality]
```

### 3.5 GRPO/DAPO 超参数（已调优，不要随意修改）

```yaml
# === 这些参数已针对 3090 24GB + Qwen2.5-1.5B 调优 ===
loss_type: "dapo"
epsilon: 0.2                      # ε_low（标准裁剪下界）
epsilon_high: 0.28                # ε_high（Clip-Higher 非对称上界）
beta: 0.0                         # 无 KL 惩罚（RLVR 不需要）
mask_truncated_completions: true  # Overlong Filtering

per_device_train_batch_size: 1
gradient_accumulation_steps: 4    # 有效 batch = 4
num_generations: 8                # 每 prompt 采样 G=8 个 completion
max_prompt_length: 512
max_completion_length: 1536       # 总长 2048 = 512 + 1536

learning_rate: 5e-6
lr_scheduler_type: "cosine"
warmup_ratio: 0.1
optim: "paged_adamw_8bit"
bf16: true
max_grad_norm: 0.1
weight_decay: 0.1

# LoRA
r: 16
lora_alpha: 16                    # scaling = alpha/r = 1.0
target_modules: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
lora_dropout: 0
```

**如果需要调参**：只允许调 `learning_rate`（范围 1e-6 ~ 1e-5）、`num_generations`（4 或 8）、`max_steps`。其他参数不动。

### 3.6 课程学习三阶段

```
阶段 1（stage1_easy）  → HotpotQA level=easy           → 300 步
阶段 2（stage2_medium）→ HotpotQA level=medium + MuSiQue 2hop → 500 步
阶段 3（stage3_hard）  → HotpotQA level=hard + MuSiQue 3-4hop → 700 步
```

每个阶段从上一阶段 checkpoint 继续训练（`resume_from_checkpoint`）。

---

## 四、检索服务器规范

### 4.1 接口定义

检索服务器运行在 `http://127.0.0.1:8000`，提供单一接口：

```
POST /retrieve
Content-Type: application/json

Request:  {"query": "string", "topk": 3}
Response: {"query": "string", "results": [{"docid": "str", "score": float, "contents": "str"}, ...]}
```

### 4.2 RRF 融合公式

```python
score(d) = Σ_i 1 / (k + rank_i(d))   # k = 60
```

BM25 和 E5 各取 top-100，RRF 融合后返回 top-3。

### 4.3 依赖服务

训练前必须确保检索服务器已启动。训练脚本中应包含健康检查：

```python
import requests
try:
    r = requests.post("http://127.0.0.1:8000/retrieve",
                       json={"query": "test", "topk": 1}, timeout=5)
    assert r.status_code == 200
except:
    raise RuntimeError("检索服务器未启动！先运行: bash scripts/start_retrieval.sh")
```

---

## 五、数据格式规范

### 5.1 训练数据 Parquet 格式

每条记录必须包含以下字段：

```python
{
    "data_source": "hotpotqa",                          # 数据集来源
    "prompt": [{"role": "user", "content": "问题文本"}],  # 单轮对话格式
    "ability": "fact-reasoning",                        # 固定值
    "reward_model": {
        "style": "rule",
        "ground_truth": "标准答案"                        # 字符串或字符串列表
    },
    "extra_info": {
        "split": "train",
        "index": 0,
        "level": "easy",                                # 课程难度标签
        "hop_count": 2                                  # 跳数
    }
}
```

### 5.2 SFT 冷启动数据格式

```python
{
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "<think>...<search>...</search>\n<information>...</information>\n...</think>\n<answer>答案</answer>"}
    ]
}
```

### 5.3 评测结果输出格式

```json
{
    "model": "minisearch_r1_stage3",
    "dataset": "hotpotqa_dev",
    "metrics": {"em": 0.35, "f1": 0.42, "avg_searches": 1.8},
    "num_samples": 500,
    "timestamp": "2025-xx-xx"
}
```

---

## 六、常见陷阱与排错指南

### 6.1 显存 OOM

| 现象 | 原因 | 解决 |
|------|------|------|
| GRPO 训练 OOM | num_generations 过大 | 降为 4，或降低 max_completion_length 到 1024 |
| vLLM 推理 OOM | gpu_memory_utilization 过高 | 从 0.6 降到 0.5 |
| SFT 训练 OOM | batch_size 过大 | per_device_train_batch_size=2，gradient_accumulation=4 |

### 6.2 搜索标签解析失败

```python
# 鲁棒的标签提取（处理不完整标签）
import re
def extract_search_query(text):
    match = re.search(r'<search>(.*?)(?:</search>|$)', text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_answer(text):
    match = re.search(r'<answer>(.*?)(?:</answer>|$)', text, re.DOTALL)
    return match.group(1).strip() if match else ""
```

### 6.3 Reward 全为零

如果一个 batch 内所有 rollout 的 reward 都是 0，GRPO 的优势全为零，无有效梯度。解决：
- 确保 SFT 冷启动已完成（模型至少会生成正确格式）
- 课程学习从最简单的问题开始
- 检查答案归一化是否正确（常见 bug：未去除标点导致永远不匹配）

### 6.4 检索服务器崩溃

FAISS 索引**文件**约 90GB（29M × 768 dim × 4 bytes）。用 `np.memmap` 懒加载，
**驻内存约 15-20GB**（passage 原文 dict ~10GB + JVM + E5 模型 + 索引元数据）。
如果 AutoDL 实例内存/磁盘不够：
- 改用 `IndexIVFPQ` 压缩索引，磁盘可压到 ~15GB，精度下降 2-3%
- 或仅使用 BM25（不用 Dense 检索），将混合检索作为消融实验

### 6.5 Unsloth 版本兼容性

```bash
# 必须使用的版本组合（已验证）
pip install unsloth
pip install trl>=0.17
pip install transformers>=4.45
pip install vllm==0.6.3
```

如果 Unsloth 与 vLLM 版本冲突，优先保证 Unsloth 版本正确。

---

## 七、代码风格规范

### 7.1 通用规则

- 所有代码文件顶部必须有 docstring 说明用途
- 使用 `typing` 做类型标注
- 配置使用 YAML 文件，不要硬编码在 Python 中
- 训练脚本必须支持 `--config` 参数加载配置
- 所有路径使用 `pathlib.Path`，不要硬编码绝对路径
- 日志使用 `logging` 模块，不要用 `print`

### 7.2 命名约定

```python
# 文件名：snake_case
grpo_train.py, search_env.py, rewards.py

# 类名：PascalCase
class SearchEnv: ...
class RetrievalServer: ...

# 函数/变量：snake_case
def f1_reward(): ...
def create_retrieval_mask(): ...
```

### 7.3 Git 提交规范

```
feat: 添加 RRF 混合检索融合
fix: 修复 token masking 边界条件
exp: 完成消融实验 A（课程学习）
data: 生成 stage2 训练数据
docs: 更新 README 评测结果
```

---

## 八、任务执行检查清单

AI Agent 在执行任何任务前，请对照此清单：

- [ ] 我的修改是否违反了第一节的硬约束？
- [ ] 新代码是否放在了正确的目录下？
- [ ] 是否引入了不在 requirements.txt 中的新依赖？如果是，需说明理由
- [ ] 训练相关代码是否能在 24GB 显存内运行？
- [ ] 是否修改了 system prompt 或标签格式？（不允许）
- [ ] reward 函数是否使用了 SQuAD 标准归一化？
- [ ] 检索相关代码是否包含超时处理和错误恢复？
- [ ] 是否添加了适当的日志输出？
- [ ] 如果是新的训练配置，是否作为 YAML 文件保存在 configs/ 下？

---

## 九、模块开发优先级

如果被要求"开始实现项目"，请按以下顺序推进：

```
优先级 1（基础设施）：
  scripts/setup_autodl.sh → requirements.txt → configs/

优先级 2（数据层）：
  data/download_datasets.py → data/preprocess.py → data/curriculum_split.py

优先级 3（检索层）：
  retrieval/server.py → retrieval/test_retrieval.py

优先级 4（训练层 - SFT）：
  data/generate_sft_data.py → training/sft_train.py

优先级 5（训练层 - RL 核心）：
  training/rewards.py → training/utils.py → training/search_env.py → training/grpo_train.py

优先级 6（评测层）：
  evaluation/metrics.py → evaluation/evaluate.py → evaluation/ablation_runner.py
```

**每完成一个优先级，先测试验证再进入下一个。不要一次写完所有代码。**

---

## 十、测试验证标准

### 10.1 检索服务器验证

```bash
# 启动后运行
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "Albert Einstein birthplace", "topk": 3}'
# 期望：返回包含 Ulm, Germany 相关内容的 3 条结果
```

### 10.2 SFT 模型验证

SFT 训练后，模型应能生成正确格式的输出：
```python
# 输入：任意问题
# 期望输出：包含 <think>、<search>、<answer> 标签的文本
# 验证：>90% 的生成包含至少一个 <search> 标签和一个 <answer> 标签
```

### 10.3 GRPO 训练验证

训练 100 步后检查：
- reward 均值应 > 0（如果持续为 0，说明有 bug）
- reward 应有上升趋势（即使缓慢）
- 模型生成的 `<search>` 查询应与问题相关（抽样检查 5 条）

### 10.4 评测验证

```python
# HotpotQA dev set 前 100 条
# 期望 EM > 0.15, F1 > 0.25（SFT 后 baseline）
# GRPO 训练后期望 EM > 0.25, F1 > 0.35
```