import os

from dotenv import load_dotenv

load_dotenv()

config = {
    "model": {
        "is_cosmoe": True,
        "model_id": "jungledude23/mini-mixtral-v1",
        "is_tiny": True,
        "add_lb": False,
    },
    "training": {
        "max_steps": 12500,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 4,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 2,
        "warmup_ratio": 0.05,
        "logging_steps": 4,
        "save_strategy": "steps",
        "eval_strategy": "steps",
        "eval_steps": 10000,
        "save_steps": 10000,
        "learning_rate": 2.5e-5,
        "bf16": True,
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "push_to_hub": True,
        "hub_token": os.environ.get("HUB_TOKEN"),
        "output_dir": "jungledude23/mini-mixtral-cosmoe-lb",
        "report_to": "wandb",
    },
}


def get_config(path):
    keys = path.split(".")
    result = config
    for key in keys:
        result = result[key]
    return result
