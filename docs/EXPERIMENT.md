# Experiment Specification

## Objective

Compare normal pretraining from a random initialization against normal pretraining from an initialization produced by a special token-id permutation pretraining stage.

## Fixed Components

- Dataset: `Skywork/SkyPile-150B`, split `train`, text field `text`.
- Tokenizer: fixed 4096-token BPE tokenizer saved at `tokenizer/skypile_4k_tokenizer.json`.
- Context length: 8192.
- Model family: decoder-only causal Transformer.
- Architecture: PreNorm, RMSNorm, SwiGLU, RoPE, no bias, untied embedding and classifier.
- Precision: bf16 on GPU.
- Optimizer: AdamW.
- Distributed mode: 2 GPU DDP through `torchrun`.

## Tokenizer

SkyPile-150B is stored as many large JSONL shards. The repository currently exposes 437 `data/*.jsonl` files totaling about 665GB, with most shards around 1-2GB. Each record has a `text` field.

The tokenizer training script reads from the first JSONL shard and stops after 100MB of UTF-8 text:

```bash
python scripts/train_tokenizer.py --max-files 1 --max-docs 0 --max-text-bytes 100000000 --num-threads 8
```

Do not feed a full 1-2GB shard into BPE training on a memory-limited machine. The trainer maintains BPE frequency and merge candidate structures, so peak RAM can be much larger than input size.

Special tokens are fixed at the front of the vocabulary:

| Token | ID |
| --- | --- |
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<unk>` | 3 |

Special tokens are never permuted during shuffle pretraining.

## Model Hyperparameters

```yaml
vocab_size: 4096
max_seq_len: 8192
n_layers: 12
d_model: 512
n_heads: 8
head_dim: 64
d_ff: 1792
norm_eps: 1.0e-6
rope_theta: 10000.0
dropout: 0.0
attention_dropout: 0.0
tie_embeddings: false
bias: false
```

Expected parameter count is approximately 49.8M. Verify with:

```bash
PYTHONPATH=src python scripts/count_params.py --config configs/model_50m.yaml
```

## Training Hyperparameters

```yaml
seq_len: 8192
precision: bf16
compile: true
gradient_checkpointing: true
flash_attention: true
micro_batch_size: 1
grad_accum_steps: 32
world_size: 2
global_batch_tokens: 524288
learning_rate: 3.0e-4
min_learning_rate: 3.0e-5
beta1: 0.9
beta2: 0.95
weight_decay: 0.1
grad_clip: 1.0
scheduler: cosine
warmup_tokens: 100000000
```

Each optimizer step sees:

```text
8192 seq_len * 1 micro_batch * 32 grad_accum * 2 GPUs = 524288 tokens
```

Approximate step counts:

| Stage | Tokens | Optimizer steps |
| --- | ---: | ---: |
| Random baseline normal train | 3B | 5723 |
| Shuffle pretrain | 1B | 1908 |
| Normal train after shuffle pretrain | 3B | 5723 |

## Strategies

### Random baseline

Run `configs/experiments/random_3b.yaml`.

This starts from random initialization and trains normally for 3B tokens.

### Shuffle pretraining stage

Run `configs/experiments/shuffle_pretrain_1b.yaml`.

For every 8192-token sequence, a fresh random bijection is sampled over token IDs 4 through 4095. The same bijection is applied to both inputs and labels for that sequence. IDs 0 through 3 remain unchanged.

This preserves the relative equality pattern inside the sequence while destroying the stable lexical meaning of ordinary token IDs.

### Normal training after shuffle

Run `configs/experiments/shuffle_then_normal_3b.yaml`.

This loads `runs/shuffle_pretrain_1b/checkpoints/latest.pt`, discards the shuffle mode, and trains normally for 3B tokens.

## Outputs

Each run writes:

- `runs/<name>/loss.csv`
- `runs/<name>/checkpoints/latest.pt`
- `runs/<name>/checkpoints/step_*.pt`
- Slurm stdout and stderr under `logs/slurm/`

CSV columns:

```text
step,tokens_seen,loss,lr,tokens_per_second,elapsed_seconds
```
