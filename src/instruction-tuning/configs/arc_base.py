import os

from dotenv import load_dotenv

load_dotenv()

config = {
    "model": {
        "is_cosmoe": False,
        "model_id": "mistralai/Mixtral-8x7B-v0.1",
    },
    "dataset": {
        "name": "jungledude23/arc-mistral-instruct",
        "train_subset": "train",
        "eval_subset": "test",
    },
    "training": {
        "num_train_epochs": 4,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 8,
        "warmup_ratio": 0.03,
        "logging_steps": 4,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "learning_rate": 2.5e-5,
        "bf16": True,
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "push_to_hub": True,
        "hub_token": os.environ.get("HUB_TOKEN"),
        "output_dir": "jungledude23/mixtral-base-arc",
    },
}


def get_config(path):
    keys = path.split(".")
    result = config
    for key in keys:
        result = result[key]
    return result
