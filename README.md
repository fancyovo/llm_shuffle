# LLM Shuffle Pretraining Experiment

This project compares two pretraining initializations for a small decoder-only Transformer on a SkyPile-150B subset:

1. Random initialization, then normal language-model pretraining for 3B tokens.
2. Random initialization, then 1B tokens of per-sequence token-id permutation pretraining, then normal language-model pretraining for 3B tokens.

The model uses PreNorm, RMSNorm, SwiGLU, RoPE, untied input embeddings and classifier, a 4096-token fixed tokenizer, and an 8192-token context length.

Start with `docs/AGENT_RUNBOOK.md` when deploying on the server. See `docs/HANDOFF.md` for operational details and `docs/EXPERIMENT.md` for the experiment specification.

Current recorded result: `random_3b` completed 3B tokens with final loss 2.8660, and `shuffle_then_normal_3b` completed 3B normal tokens from a partially interrupted 823M-token shuffle checkpoint with final loss 2.8280. See `docs/EXPERIMENT_RECORD.md`.

For follow-up generation-quality validation, use the blind A/B workflow in `docs/GENERATION_BLIND_EVAL.md`. API keys, model weights, logs, generated outputs, and judgement results must stay out of Git.
