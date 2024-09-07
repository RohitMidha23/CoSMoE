# CoSMoE - Covariant and Specialized Experts in Mixture of Experts

This repository contains the source code for CoSMoE - Covariant and Specialized Experts in Mixture of Experts, my MSc Thesis project, submitted for MSc Computing (AI ML) 2024.

## Structure

`src/pre-training` contains the code for all the pre-training experiments carried out on tiny-mixtral and mini-mixtral.

`src/instruction-tuning` contains all the experiments carried out on Mixtral-8x7b.

`vis.py` contains code for visualizing the cross-correlation matrix.

## Evaluations

For evaluations, we use [`lighteval`](https://github.com/huggingface/lighteval).

Following are examples of commands used to evaluate the instruction tuned models:

```bash
lighteval accelerate --use_chat_template --model_config_path="eval-configs/mixtral_base.yaml" --tasks "lighteval|arc:easy|0|0,leaderboard|arc:challenge|0|0,original|mmlu|0|0" --output_dir "llm-results"

lighteval accelerate --use_chat_template --model_config_path="eval-configs/mixtral_cosmoe.yaml" --tasks "lighteval|arc:easy|0|0,leaderboard|arc:challenge|0|0,original|mmlu|0|0" --output_dir "llm-results"
```
