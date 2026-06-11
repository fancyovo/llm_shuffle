# LLM Shuffle Pretraining Experiment

This project compares two pretraining initializations for a small decoder-only Transformer on a SkyPile-150B subset:

1. Random initialization, then normal language-model pretraining for 3B tokens.
2. Random initialization, then 1B tokens of per-sequence token-id permutation pretraining, then normal language-model pretraining for 3B tokens.

The model uses PreNorm, RMSNorm, SwiGLU, RoPE, untied input embeddings and classifier, a 4096-token fixed tokenizer, and an 8192-token context length.

Start with `docs/AGENT_RUNBOOK.md` when deploying on the server. See `docs/HANDOFF.md` for operational details, `docs/EXPERIMENT.md` for the experiment specification, and `docs/EXPERIMENT_RECORD.md` for the experiment ledger.

Current recorded result: the constant-LR rerun completed with `random_3b_constant` final loss 2.8689 and `shuffle_then_normal_3b_constant` final loss 2.8023, but DeepSeek force-choice judging did not show a significant generation-quality win. A new shuffle-only 3B dynamics run is in progress. See `docs/EXPERIMENT_RECORD.md`.

For follow-up generation-quality validation, use the blind A/B workflow in `docs/GENERATION_BLIND_EVAL.md`. API keys, model weights, logs, generated outputs, and judgement results must stay out of Git.

For the constant-LR rerun, use `configs/experiments_constant/` and submit
`sbatch/random_3b_constant_lr.sbatch` plus
`sbatch/shuffle_1b_then_normal_3b_constant_lr.sbatch`. These write to
`runs_constant/` and do not overwrite the earlier cosine-decay runs.

For the shuffle-pretraining dynamics run, use
`configs/experiments_shuffle_dynamics/shuffle_pretrain_3b.yaml` and submit
`sbatch/shuffle_pretrain_3b_dynamics.sbatch`. It writes to
`runs_shuffle_dynamics/`. To continue the completed 3B checkpoint for another
7B tokens on new shards, submit `sbatch/shuffle_pretrain_continue_7b.sbatch`.
