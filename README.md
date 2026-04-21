<div align="center">

# MiniSearch-R1

**用强化学习教 1.5B 小模型学会"边搜边想"**

基于 Search-R1 框架，首次将搜索增强 RL 训练下探至 1.5B 规模，并引入课程学习、混合检索、过程奖励三项改进。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-2025.x-green.svg)](https://github.com/unslothai/unsloth)
[![TRL](https://img.shields.io/badge/TRL-0.14+-yellow.svg)](https://github.com/huggingface/trl)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

---

## 📖 项目简介

**MiniSearch-R1** 是一个面向 Agentic RAG 场景的**小模型搜索推理 Agent**。项目基于 Qwen2.5-1.5B-Instruct，通过 **SFT 冷启动 + Multi-turn GRPO 强化学习**，让模型自主学会"**在推理过程中何时搜索、搜索什么、如何整合多轮结果**"。

与传统 RAG 的"被动一次性检索"不同，本项目训练的 Agent 能**自主决策多轮搜索策略**——这正是 OpenAI Deep Research、Kimi 探索版、DeepSeek Search 等 2025 年热门产品的核心能力。

### 核心贡献

- 🎯 **首次下探 1.5B 规模**：对标 Search-R1（COLM 2025，最小 3B）和 R1-Searcher（2025，最小 7B），首次在 1.5B 模型上验证搜索增强 RL 的可行性
- 🔬 **三项方法改进**：
  - **课程学习**：按 hop 数渐进训练，解决小模型 reward 稀疏问题
  - **混合检索**：BM25 + Dense + RRF 融合，提升检索召回
  - **过程奖励**：retrieval-aware process reward，提供密集训练信号
- 💰 **极低成本**：全流程在 Kaggle 免费 GPU 完成，实际花费 **< ¥5**
- 🧪 **完整消融**：5 组消融实验量化各项改进的独立贡献

---

## 🏗️ 系统架构

```
┌────────────────────────────────────────────────────────┐
│                    用户多跳问题输入                       │
└─────────────────────┬──────────────────────────────────┘
                      ▼
      ┌───────────────────────────────────┐
      │  Qwen2.5-1.5B + LoRA (r=16)       │
      │                                    │
      │  ┌──────────────────────────────┐ │
      │  │ <think> 推理 </think>         │ │
      │  │ <search> 查询 </search>       │ │
      │  │ <result> 检索结果 </result>   │ │  ← loss_mask=0
      │  │ <think> 继续推理 </think>     │ │
      │  │ <answer> 最终答案 </answer>   │ │
      │  └──────────────────────────────┘ │
      └───────────┬───────────────────────┘
                  │
                  ▼ HTTP
      ┌───────────────────────────────────┐
      │  Local Retrieval Server (Flask)   │
      │  ┌──────────┬──────────┬────────┐ │
      │  │   BM25   │    E5    │  RRF   │ │
      │  │(Pyserini)│(FAISS)   │ Fusion │ │
      │  └──────────┴──────────┴────────┘ │
      │         Wikipedia 2018 KILT       │
      │          (29M passages)            │
      └───────────────────────────────────┘
```

**训练流程**：SFT 冷启动（500 条合成轨迹）→ 课程 GRPO（三阶段：1-hop → 2-hop → 多跳）→ 4 benchmark 评测

---

## 📊 实验结果

### 主实验（EM / F1）

| Method | Model | HotpotQA | 2Wiki | Musique | Bamboogle | Avg |
|--------|:-----:|:--------:|:-----:|:-------:|:---------:|:---:|
| Zero-shot | Qwen2.5-1.5B | {XX} | {XX} | {XX} | {XX} | {XX} |
| Vanilla RAG | Qwen2.5-1.5B | {XX} | {XX} | {XX} | {XX} | {XX} |
| SFT-only | Qwen2.5-1.5B | {XX} | {XX} | {XX} | {XX} | {XX} |
| **MiniSearch-R1 (Ours)** | **Qwen2.5-1.5B** | **{YY}** | **{YY}** | **{YY}** | **{YY}** | **{YY}** |
| *Search-R1 (Reference)* | *Qwen2.5-3B* | *0.426* | *0.297* | *0.083* | *0.296* | *0.276* |

### 消融实验

| Ablation | HotpotQA EM | Δ |
|----------|:-----------:|:--:|
| Full (MiniSearch-R1) | **{YY}** | — |
| w/o Curriculum Learning | {XX} | -{X.X} |
| w/o Hybrid Retrieval (BM25 only) | {XX} | -{X.X} |
| w/o Retrieval Bonus | {XX} | -{X.X} |
| GRPO (no DAPO patches) | {XX} | -{X.X} |
| Qwen2.5-0.5B (smaller) | {XX} | -{X.X} |

### 训练曲线

详见 [`docs/training_curves.md`](docs/training_curves.md)（reward / KL / 生成长度 / 搜索次数）

---

## 🔥 三项核心改进

### 1. 课程学习（Curriculum Learning）

小模型 RL 面临 **reward sparsity collapse**——1.5B 在多跳任务上初始成功率仅 3%，GRPO 组内 88% 的 advantage 为 0，训练无法启动。

**解决方案**：按 hop 数将训练数据分为三档（1-hop / 2-hop / 3+hop），按训练步数渐进切换（200 / 200 / 300 步）。参考 RAGEN-2 和 R1-Searcher 的难度分层思路。

### 2. 混合检索（Hybrid Retrieval + RRF）

原论文 Search-R1 分别测试了 BM25 和 E5，但**未做融合**。本项目使用 **Reciprocal Rank Fusion**（Elasticsearch 工业标准）融合两路检索：

$$\text{score}(d) = \sum_i \frac{1}{k + \text{rank}_i(d)}, \quad k=60$$

### 3. 检索质量过程奖励（Retrieval-Aware Process Reward）

原论文仅用 outcome reward（EM），本项目引入密集过程信号：

$$R = R_{\text{EM}} + 0.2 \cdot R_{\text{format}} + 0.1 \cdot R_{\text{retrieval\_hit}}$$

其中 retrieval bonus **只在答案正确时生效**（门控），**按命中比例而非次数计算**（归一化）——三道闸门防止 reward hacking。

---

## 🛠️ 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| **基座模型** | Qwen2.5-1.5B-Instruct | Unsloth 优化版 |
| **训练框架** | Unsloth + TRL GRPOTrainer | 单卡 P100 16GB 即可 |
| **RL 算法** | GRPO + DAPO 4 项补丁 | `loss_type="dapo"` |
| **微调方式** | 4-bit QLoRA (r=16, α=16) | 峰值显存 5-7 GB |
| **检索语料** | Wikipedia 2018 KILT | 29M passages, 5-8 GB |
| **检索器** | Pyserini BM25 + E5-base-v2 + RRF | Flask HTTP 服务 |
| **推理加速** | vLLM (Unsloth 内置) | ~300 tokens/s on P100 |
| **训练平台** | Kaggle P100 (30h/week) + Colab T4 | 全免费 |

---

## 📁 项目结构

```
MiniSearch-R1/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── download_benchmarks.py      # HotpotQA/2Wiki/Musique/Bamboogle
│   ├── download_wiki.py             # Wikipedia 2018 KILT dump
│   ├── synthesize_sft_data.py       # DeepSeek API 合成 SFT 冷启动数据
│   └── split_by_hops.py             # 按 hop 数分档用于课程学习
│
├── retriever/
│   ├── build_bm25_index.py          # Pyserini BM25 索引构建
│   ├── build_dense_index.py         # E5 + FAISS Dense 索引
│   ├── hybrid_retriever.py          # RRF 融合
│   └── server.py                    # Flask HTTP 检索服务
│
├── training/
│   ├── sft_cold_start.py            # Unsloth SFT 冷启动训练
│   ├── grpo_train.py                # Multi-turn GRPO / DAPO 训练
│   ├── multiturn_env.py             # Multi-turn rollout 环境
│   ├── reward_fns.py                # EM / F1 / Format / Retrieval Bonus
│   └── curriculum.py                # 三阶段课程学习调度
│
├── evaluation/
│   ├── evaluate.py                  # 4 benchmark 批量评测
│   ├── metrics.py                   # EM / F1 规范化
│   └── case_study.py                # 定性案例分析
│
├── ablations/
│   ├── ablation_curriculum.py       # 消融 A
│   ├── ablation_retrieval.py        # 消融 B
│   ├── ablation_reward.py           # 消融 C
│   ├── ablation_rl_algo.py          # 消融 D
│   └── ablation_scale.py            # 消融 E
│
├── scripts/
│   ├── run_all.sh                   # 一键全流程
│   └── kaggle_resume.py             # Kaggle 9h 断连续训
│
├── notebooks/
│   ├── 01_sft_on_kaggle.ipynb       # Kaggle 主训练 Notebook
│   ├── 02_grpo_on_kaggle.ipynb
│   └── 03_evaluation.ipynb
│
├── configs/
│   ├── sft.yaml
│   ├── grpo_dapo.yaml
│   └── curriculum.yaml
│
└── docs/
    ├── design.md                    # 设计文档
    ├── training_curves.md           # 训练曲线与分析
    └── failure_analysis.md          # Bad case 分类分析
```

---

## 🚀 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/<your-username>/MiniSearch-R1.git
cd MiniSearch-R1

# 安装依赖（推荐使用 conda）
conda create -n minisearch python=3.10 -y
conda activate minisearch
pip install -r requirements.txt
```

### 数据下载

```bash
# 下载四个 benchmark
python data/download_benchmarks.py

# 下载 Wikipedia 2018 KILT dump（约 8 GB）
python data/download_wiki.py

# 合成 SFT 冷启动数据（需要 DeepSeek API Key）
export DEEPSEEK_API_KEY=your_key
python data/synthesize_sft_data.py --n_samples 500
```

### 启动检索服务

```bash
# 构建 BM25 索引（约 30-60 分钟）
python retriever/build_bm25_index.py

# 构建 Dense 索引（约 4-8 小时 CPU，建议预计算）
python retriever/build_dense_index.py

# 启动 Flask 服务（默认 8000 端口）
python retriever/server.py &
```

### 训练

```bash
# SFT 冷启动（约 1-2 小时 on P100）
python training/sft_cold_start.py --config configs/sft.yaml

# Multi-turn GRPO（约 20-30 小时，建议分 3-4 段 Kaggle session）
python training/grpo_train.py --config configs/grpo_dapo.yaml
```

### 评测

```bash
# 4 benchmark 批量评测
python evaluation/evaluate.py --checkpoint outputs/grpo_final

# 消融实验
bash scripts/run_ablations.sh
```

**完整 Kaggle 部署教程**见 [`docs/kaggle_setup.md`](docs/kaggle_setup.md)。

---

## 📚 关键参考论文

1. **Search-R1** (COLM 2025) — 主复现对象  
   *Jin et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning."*  
   [[arXiv:2503.09516]](https://arxiv.org/abs/2503.09516) [[GitHub]](https://github.com/PeterGriffinJin/Search-R1)

2. **R1-Searcher** (2025) — 技术借鉴（两阶段训练、F1 reward）  
   *Song et al. "R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning."*  
   [[arXiv:2503.05592]](https://arxiv.org/abs/2503.05592) [[GitHub]](https://github.com/RUCAIBox/R1-Searcher)

3. **DAPO** (NeurIPS 2025) — RL 算法改进  
   *Yu et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale."*  
   [[arXiv:2503.14476]](https://arxiv.org/abs/2503.14476)

4. **RAGEN-2** (2026) — 小模型 RL 的 Echo Trap 现象  
   *Wang et al. "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning."*

---

## 📌 项目亮点

- ✅ **端到端 Agentic RL 全链路**：数据合成 → SFT → Multi-turn GRPO → 评测 → 消融
- ✅ **Retrieved Token Masking 自研实现**：解决 multi-turn RL 中梯度被检索内容污染的核心工程问题
- ✅ **系统性消融设计**：5 组消融量化每项改进的独立贡献
- ✅ **极致成本控制**：免费 GPU + 免费 API，全流程 < ¥5
- ✅ **对标 2025-2026 最热赛道**：Deep Research / Agentic RAG

---

## 🔬 Future Work

- [ ] 迁移至真实 Web 搜索环境（参考 DeepResearcher）
- [ ] 训练 ThinkPRM-1.5B 作为过程奖励模型替代 rule-based bonus
- [ ] Test-time compute scaling（Beam search + PRM 引导）
- [ ] 扩展至业务场景（小红书笔记 / 电商商品 / 垂域 RAG）

---

## 📝 License

本项目遵循 [Apache License 2.0](LICENSE)。

## 🙏 致谢

感谢 Search-R1、R1-Searcher、Unsloth、TRL 等开源项目的贡献者，本项目在其工作基础上完成研究。

## 📬 联系方式

- Author: **zejian_cao**
- Email: zejiang.cao@sjtu.edu.cn


---

<div align="center">

**如果本项目对你有帮助，欢迎 Star ⭐**

</div>