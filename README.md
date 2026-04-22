<div align="center">

# MiniSearch-R1

**用强化学习让 1.5B 小模型学会“边搜边想”**

面向 **Search Agent / Agentic RAG** 的小模型搜索增强推理项目，基于 Search-R1 范式，在 **Qwen2.5-1.5B-Instruct** 上系统探索 **课程学习、混合检索、过程奖励** 对多轮搜索推理能力的影响。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-2025.x-green.svg)](https://github.com/unslothai/unsloth)
[![TRL](https://img.shields.io/badge/TRL-0.14+-yellow.svg)](https://github.com/huggingface/trl)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

> **一句话概括**  
> MiniSearch-R1 关注的不是“给模型外挂检索器”，而是让模型在推理过程中学会 **何时检索、检索什么、如何利用检索结果继续推理**。

---

## 目录

- [项目简介](#项目简介)
- [核心亮点](#核心亮点)
- [系统架构](#系统架构)
- [方法设计](#方法设计)
- [实验设计](#实验设计)
- [实验结果](#实验结果)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [关键参考论文](#关键参考论文)
- [项目亮点（面向面试）](#项目亮点面向面试)
- [Roadmap](#roadmap)
- [License](#license)

---

## 项目简介

**MiniSearch-R1** 是一个面向 **Agentic RAG / Search Agent** 场景的小模型搜索推理项目。项目以 **Qwen2.5-1.5B-Instruct** 为基座，通过 **SFT 冷启动 + Multi-turn GRPO 强化学习**，训练模型逐步掌握以下能力：

- 在推理过程中判断是否需要检索
- 生成有信息增益的搜索查询
- 综合多轮检索结果完成最终回答

与传统 RAG 的“一次性被动检索”不同，本项目更关注 **模型在推理链中主动调用检索工具并迭代修正策略** 的能力。这种能力是 Search Agent、Deep Research、Agentic RAG 等方向的关键基础，也是大模型从“回答问题”走向“解决问题”的重要一步。

本项目的目标并非仅做论文复现，而是在 **1.5B 规模小模型** 上系统评估搜索增强强化学习链路的可行性，并尝试给出更适合轻量模型的训练设计与工程实现。

---

## 核心亮点

- **小模型设定**：聚焦 1.5B 规模模型，验证搜索增强推理能力能否在轻量参数规模上被有效训练出来
- **方法改进清晰**：围绕课程学习、混合检索、过程奖励三条主线展开，便于做可解释分析与消融
- **任务链路完整**：覆盖数据准备、检索服务、SFT、GRPO、评测与案例分析，形成较完整的 Agentic RL 工程闭环
- **研究与工程并重**：既关注 reward design、训练稳定性、多跳推理能力，也重视检索服务、索引构建与评测组织
- **面向真实场景**：问题设定直接对应搜索推理型大模型能力，与 Search Agent / Deep Research 类方向高度相关

---

## 系统架构

```text
┌────────────────────────────────────────────────────────┐
│                    用户多跳问题输入                    │
└─────────────────────┬──────────────────────────────────┘
                      ▼
      ┌───────────────────────────────────┐
      │  Qwen2.5-1.5B + LoRA (r=16)       │
      │                                   │
      │  ┌─────────────────────────────┐  │
      │  │ <think> 推理 </think>        │  │
      │  │ <search> 查询 </search>      │  │
      │  │ <result> 检索结果 </result>  │  │
      │  │ <think> 继续推理 </think>    │  │
      │  │ <answer> 最终答案 </answer>  │  │
      │  └─────────────────────────────┘  │
      └───────────┬───────────────────────┘
                  │
                  ▼ HTTP
      ┌───────────────────────────────────┐
      │   Retrieval Server (Flask)        │
      │  ┌──────────┬──────────┬────────┐ │
      │  │   BM25   │    E5    │  RRF   │ │
      │  │(Pyserini)│(FAISS)   │ Fusion │ │
      │  └──────────┴──────────┴────────┘ │
      │      Wikipedia 2018 / KILT        │
      └───────────────────────────────────┘
```

**整体流程**：SFT 冷启动 → 课程化 GRPO → 多 benchmark 评测 → 消融分析 → case study

---

## 方法设计

### 1. 课程学习（Curriculum Learning）

小模型在多跳强化学习任务中容易出现 **reward sparsity**：初始策略既不会有效搜索，也无法稳定给出正确答案，导致优势值稀疏、训练难以启动。

为缓解这一问题，本项目按推理难度对训练样本进行课程化组织，将数据拆分为：

- `1-hop`
- `2-hop`
- `3-hop+`

训练时按照“从易到难”的顺序逐步切换数据分布，让模型先学会基础检索与简单推理，再进入复杂多跳问题。

### 2. 混合检索（Hybrid Retrieval + RRF）

多跳问答场景中，单一路径检索通常难以同时兼顾关键词匹配和语义召回：

- `BM25` 更擅长关键词命中
- `Dense Retrieval` 更擅长语义相似性召回

因此本项目采用 **BM25 + Dense Retrieval + RRF** 的混合检索方案，在不显著增加系统复杂度的前提下提升检索质量。

Reciprocal Rank Fusion 形式如下：

$$
\text{score}(d)=\sum_i \frac{1}{k+\text{rank}_i(d)}, \quad k=60
$$

### 3. 过程奖励（Retrieval-Aware Process Reward）

仅依赖最终答案是否正确作为奖励，往往不足以支撑小模型的稳定学习。为此，本项目在 outcome reward 之外引入过程级信号：

$$
R = R_{\text{answer}} + \alpha \cdot R_{\text{format}} + \beta \cdot R_{\text{retrieval}}
$$

其中：

- `R_answer`：答案正确性奖励
- `R_format`：输出是否符合多轮推理格式
- `R_retrieval`：检索是否命中有效支持证据

其核心目的，是为多轮搜索推理过程提供更密集、更可学习的训练信号。

---

## 实验设计

### Benchmark

项目计划在以下多跳问答 benchmark 上进行评测：

- HotpotQA
- 2WikiMultiHopQA
- MuSiQue
- Bamboogle

### 对比设置

主实验包含以下对比项：

- Zero-shot Base Model
- Vanilla RAG
- SFT-only
- MiniSearch-R1
- Search-R1 / R1-Searcher 公开结果参考

### 消融实验

计划进行以下消融：

- 去掉课程学习
- 去掉混合检索，仅保留 BM25
- 去掉过程奖励
- 去掉 RL 算法增强项
- 更小模型尺度对比

---

## 实验结果

> 下列表格为最终实验结果预留区域。待全部实验完成后统一补充。

### 主实验（EM / F1）

| Method | Model | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg |
|--------|:-----:|:--------:|:-----:|:-------:|:---------:|:---:|
| Zero-shot | Qwen2.5-1.5B | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| Vanilla RAG | Qwen2.5-1.5B | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| SFT-only | Qwen2.5-1.5B | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| **MiniSearch-R1 (Ours)** | **Qwen2.5-1.5B** | **待补充** | **待补充** | **待补充** | **待补充** | **待补充** |
| *Search-R1 (Reference)* | *Qwen2.5-3B* | *待补充* | *待补充* | *待补充* | *待补充* | *待补充* |

### 消融实验

| Ablation | HotpotQA EM | Δ |
|----------|:-----------:|:--:|
| Full (MiniSearch-R1) | 待补充 | — |
| w/o Curriculum Learning | 待补充 | 待补充 |
| w/o Hybrid Retrieval | 待补充 | 待补充 |
| w/o Retrieval Reward | 待补充 | 待补充 |
| w/o RL Enhancement | 待补充 | 待补充 |
| Smaller Backbone | 待补充 | 待补充 |

### 训练曲线

最终将补充以下分析内容：

- reward 曲线
- KL 曲线
- 平均生成长度
- 搜索次数分布
- 检索命中率变化

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| **基座模型** | Qwen2.5-1.5B-Instruct | 小模型搜索增强推理基座 |
| **训练框架** | Unsloth + TRL | 用于 SFT 与 GRPO 训练 |
| **强化学习算法** | GRPO | 多轮搜索推理的核心优化算法 |
| **参数高效微调** | QLoRA / LoRA | 降低训练成本与显存占用 |
| **检索器** | Pyserini + E5 + FAISS | 词法检索与语义检索结合 |
| **融合策略** | RRF | 混合检索结果融合 |
| **服务框架** | Flask | 检索服务接口 |
| **评测指标** | EM / F1 / Retrieval Hit | 联合衡量答案质量与检索质量 |

---

## 项目结构

```text
MiniSearch-R1/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── sft.yaml
│   ├── grpo.yaml
│   └── retriever.yaml
│
├── data/
│   ├── download_benchmarks.py
│   ├── download_wiki.py
│   ├── synthesize_sft_data.py
│   └── split_by_hops.py
│
├── retriever/
│   ├── build_bm25_index.py
│   ├── build_dense_index.py
│   ├── hybrid_retriever.py
│   └── server.py
│
├── training/
│   ├── sft_cold_start.py
│   ├── grpo_train.py
│   ├── multiturn_env.py
│   ├── reward_fns.py
│   └── curriculum.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── case_study.py
│
├── ablations/
│   ├── ablation_curriculum.py
│   ├── ablation_retrieval.py
│   ├── ablation_reward.py
│   ├── ablation_rl_algo.py
│   └── ablation_scale.py
│
├── docs/
│   ├── design.md
│   ├── training_curves.md
│   └── failure_analysis.md
│
└── notebooks/
    ├── 01_data_preparation.ipynb
    ├── 02_training.ipynb
    └── 03_evaluation.ipynb
```

---

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/<your-username>/MiniSearch-R1.git
cd MiniSearch-R1

conda create -n minisearch python=3.10 -y
conda activate minisearch
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载 benchmark
python data/download_benchmarks.py

# 下载 Wikipedia / KILT 语料
python data/download_wiki.py

# 合成 SFT 冷启动数据
python data/synthesize_sft_data.py --n_samples 500

# 按 hop 数拆分课程学习数据
python data/split_by_hops.py
```

### 3. 检索服务

```bash
# 构建 BM25 索引
python retriever/build_bm25_index.py

# 构建 Dense 索引
python retriever/build_dense_index.py

# 启动检索服务
python retriever/server.py
```

### 4. 训练

```bash
# SFT 冷启动
python training/sft_cold_start.py --config configs/sft.yaml

# Multi-turn GRPO
python training/grpo_train.py --config configs/grpo.yaml
```

### 5. 评测

```bash
# 批量评测
python evaluation/evaluate.py --checkpoint outputs/grpo_final

# 消融实验
bash scripts/run_ablations.sh
```

---

## 关键参考论文

1. **Search-R1**  
   *Jin et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning."*  
   [[arXiv:2503.09516]](https://arxiv.org/abs/2503.09516) [[GitHub]](https://github.com/PeterGriffinJin/Search-R1)

2. **R1-Searcher**  
   *Song et al. "R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning."*  
   [[arXiv:2503.05592]](https://arxiv.org/abs/2503.05592) [[GitHub]](https://github.com/RUCAIBox/R1-Searcher)

3. **DAPO**  
   *Yu et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale."*  
   [[arXiv:2503.14476]](https://arxiv.org/abs/2503.14476)

4. **RAGEN**  
   *Wang et al. "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning."*

---

## 项目亮点（面向面试）

- **方向契合度高**：覆盖 Search Agent、Agentic RAG、多轮推理强化学习等大模型热点方向
- **算法问题明确**：围绕 reward sparsity、检索召回、多轮推理稳定性展开
- **方法设计完整**：从课程学习、混合检索到过程奖励形成相对完整的方法闭环
- **工程实现清晰**：训练、检索、评测模块分离，具备复现实验和扩展系统的工程基础
- **研究表达友好**：便于从问题定义、方法设计、实验验证、失败案例四个维度向面试官讲述

---

## Roadmap

- [ ] 完成主实验结果补充
- [ ] 完成消融实验与 case study
- [ ] 补充训练曲线与错误分析文档
- [ ] 接入更真实的开放域搜索环境
- [ ] 研究更强的过程奖励模型与 test-time scaling 策略

---

## License

本项目代码采用 [MIT License](LICENSE)。  
模型权重、数据集与第三方依赖遵循各自原始许可证。

## 致谢

感谢 Search-R1、R1-Searcher、Unsloth、TRL、Pyserini 等开源项目为本项目提供研究基础与工程支持。

## 联系方式

- Author: **待补充**
- Email: `待补充`
- Blog / Portfolio: `待补充`

---

<div align="center">

如果这个项目对你有帮助，欢迎 Star ⭐

</div>
