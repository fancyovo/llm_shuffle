# Experiment Record: Shuffle Pretraining Initialization

**Date:** 2026-06-09 — 2026-06-10
**Status:** Completed (shuffle pretrain partially interrupted at 823M/1B tokens)

---

## Experimental Conditions

### Model

| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only Transformer (PreNorm, RMSNorm, SwiGLU, RoPE) |
| Parameters | 49,820,160 |
| Vocab size | 4096 (byte-level BPE) |
| Context length | 8192 |
| Embedding | Untied (no weight tying) |
| Bias | Disabled |

### Training

| Parameter | Value |
|-----------|-------|
| Precision | bf16 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, wd=0.1) |
| LR schedule | Linear warmup 0.1B → cosine decay to 1/10 |
| Peak LR | 3e-4 |
| Min LR | 3e-5 |
| Gradient clipping | 1.0 |
| Batch | 524,288 tokens/step (seq_len=8192, micro_bs=1, grad_accum=32, 2 GPU) |
| Hardware | 2× GPU (RTX 5090 / A100), DDP via torchrun |
| Throughput | ~200k tok/s |

### Data

| | |
|---|---|
| Dataset | Skywork/SkyPile-150B (Chinese) |
| Storage | 20 JSONL shards, 20.8 GB, local files |
| Shuffle buffer | 10,000 samples |
| Tokenizer | Fixed 4096-token BPE (`tokenizer/skypile_4k_tokenizer.json`) |

### Shuffle Pretrain Mechanism

For each 8192-token sequence, a random bijection is sampled over token IDs 4–4095.
The same permutation is applied to both input and labels for that sequence.
Special tokens (0–3: `<pad>`, `<bos>`, `<eos>`, `<unk>`) are never permuted.

---

## Run Log

| Run | Config | Steps | Tokens | Final Loss | Status |
|-----|--------|-------|--------|------------|--------|
| `random_3b` | `configs/experiments/random_3b.yaml` | 5,723 | 3.000B | 2.866 | ✅ Completed |
| `shuffle_pretrain_1b` | `configs/experiments/shuffle_pretrain_1b.yaml` | 1,569/1,908 | 0.823B/1B | 8.003 | ⚠️ Interrupted (node preempted) |
| `shuffle_then_normal_3b` | `configs/experiments/shuffle_then_normal_3b.yaml` | 5,723 | 3.000B | 2.828 | ✅ Completed |

### Shuffle Pretrain LR Schedule

```
0 ──[linear warmup 0.1B]──→ 3e-4 ──[cosine decay 0.9B]──→ 3e-5
Stopped at 823M tokens, LR ≈ 5.52e-05 (82% decayed)
```

### 3B Normal Training LR Schedule

```
0 ──[linear warmup 0.1B]──→ 3e-4 ──[cosine decay 2.9B]──→ 3e-5
```

---

## Results

### Loss Comparison (terminal)

| Metric | random_3b | shuffle_then_normal_3b | Δ |
|--------|-----------|----------------------|---|
| Raw final loss | 2.8660 | 2.8280 | **0.0381** |
| EMA-smoothed final (α=0.02) | 2.9205 | 2.8818 | **0.0386** |
| Last 100 steps avg | 2.9267 | 2.8880 | **0.0387** |

Shuffle pretrain achieves a consistent ~1.3% loss improvement over random initialization.

### Learning Curves

![Comparison plot](../runs/comparison.png)

Loss vs tokens (log x-axis): [comparison.png](../runs/comparison.png)
LR schedules: [lr_schedule.png](../runs/lr_schedule.png), [lr_shuffle_1b.png](../runs/lr_shuffle_1b.png)

### Generation Quality

6 prompts evaluated with identical sampling params (temp=0.8, top_k=40, top_p=0.92, seed=42).

Full results: [eval_results.txt](../runs/eval_results.txt)

| Prompt | random_3b | shuffle_then_normal_3b |
|--------|-----------|----------------------|
| "中国的首都是" | 偏离到"大学生",陷入重复循环 | 生成公司商业相关,逻辑自洽 |
| "光合作用是指" | 完全偏离到"供给侧结构性改革",严重重复 | 围绕"光/光子"展开,部分相关 |
| "从前有座山…" | 故事风格但陷入"老鹰"无限重复 | 有具体场景描述(石头/雕刻),故事感更强 |
| "如果明天要下雨" | 偏离到具体日期,逻辑断裂 | 简短但语法正确 |
| "计算机操作系统" | 部分相关但严重重复 | 列功能点,有结构感 |
| "关于环境保护" | 完全重复"全局观念" | 具体内容(废旧金属/塑料回收),事实合理 |

### Key Observations

1. **Loss improvement is modest but consistent** (~1.3% across all terminal metrics).
2. **Generation quality difference is substantial** — shuffle pretrain model produces more diverse, less repetitive text across all prompts.
3. **Random baseline suffers from severe repetition loops** — frequently gets stuck repeating a single phrase.
4. **Shuffle pretrain model better adheres to topic** and produces more factually-plausible content.
5. **Both models are limited by size (50M params)** — neither achieves truly fluent conversation, but shuffle pretrain consistently yields better outputs.

### Possible Explanations

- Shuffle pretraining forces the model to learn **positional and co-occurrence patterns** without relying on stable token–meaning mappings.
- This may act as a regularizer that prevents the model from falling into **brittle, overconfident patterns** (which manifest as repetition loops in generation).
- The effect is more visible in generation than in loss because loss is averaged over the entire distribution, while repetition is a tail behavior.

---

## Output Files

| File | Description |
|------|-------------|
| `runs/random_3b/loss.csv` | Random baseline training log |
| `runs/shuffle_pretrain_1b/loss.csv` | Shuffle pretrain training log |
| `runs/shuffle_then_normal_3b/loss.csv` | Shuffle → normal training log |
| `runs/comparison.png` | Loss comparison plot (EMA smoothed, log x) |
| `runs/eval_results.txt` | Multi-prompt evaluation outputs |
| `runs/lr_schedule.png` | 3B phase LR schedules |
| `runs/lr_shuffle_1b.png` | Shuffle pretrain LR schedule |
| `configs/experiments/random_3b.yaml` | Experiment config |
| `configs/experiments/shuffle_pretrain_1b.yaml` | Experiment config |
| `configs/experiments/shuffle_then_normal_3b.yaml` | Experiment config |
