from __future__ import annotations

import argparse

from llm_shuffle.config import load_config
from llm_shuffle.model import TransformerConfig, build_model, count_parameters, estimate_param_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_50m.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)["model"]
    model = build_model(cfg)
    exact = count_parameters(model)
    estimated = estimate_param_count(TransformerConfig(**cfg))
    print(f"exact_params={exact:,}")
    print(f"estimated_params={estimated:,}")


if __name__ == "__main__":
    main()
