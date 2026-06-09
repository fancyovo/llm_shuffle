from __future__ import annotations

import torch

from llm_shuffle.data import apply_sequence_permutation
from llm_shuffle.model import TransformerConfig, TransformerLM, count_parameters


def main() -> None:
    cfg = TransformerConfig(
        vocab_size=128,
        max_seq_len=64,
        n_layers=2,
        d_model=64,
        n_heads=4,
        head_dim=16,
        d_ff=128,
    )
    model = TransformerLM(cfg)
    input_ids = torch.randint(4, cfg.vocab_size, (2, 32))
    labels = torch.randint(4, cfg.vocab_size, (2, 32))
    _, loss = model(input_ids, labels)
    assert loss is not None
    loss.backward()

    x = [0, 1, 2, 3, 4, 5, 6, 7]
    y = [1, 2, 3, 4, 5, 6, 7, 8]
    px, py = apply_sequence_permutation(x, y, 4, 127)
    assert px[:4] == x[:4]
    assert py[:3] == y[:3]
    assert len(px) == len(x)
    assert len(py) == len(y)
    print(f"smoke_ok params={count_parameters(model):,} loss={float(loss):.6f}")


if __name__ == "__main__":
    main()
