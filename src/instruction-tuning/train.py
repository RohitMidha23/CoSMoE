import argparse
import os

import torch
import transformers
import wandb
from dataset_utils import tokenize_prompt
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trainer_utils import CustomTrainer
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers.utils import logging
from utils import CustomWandbCallback, print_trainable_parameters

logger = logging.get_logger(__name__)
# Load environment variables
load_dotenv()


def load_config(file_path):
    """
    Dynamically load a configuration dictionary from a Python file.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", file_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def setup_model_and_tokenizer(model_id, is_cosmoe):
    """
    Setup the model and tokenizer based on the configuration.
    """
    if is_cosmoe:
        from model import MixtralForCausalLM
    else:
        from transformers import MixtralForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = MixtralForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=nf4_config,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules=[
            # "q_proj",
            # "k_proj",
            # "v_proj",
            # "o_proj",
            "w1",
            "w2",
            "w3",
            "lm_head",
        ],  # target only the expert weights
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        print(torch.cuda.device_count())
        model.is_parallelizable = True
        model.model_parallel = True
    return model, tokenizer


def prepare_datasets(dataset_name, train_subset="train", eval_subset="test"):
    """
    Load and prepare datasets for training.
    """
    dataset = load_dataset(dataset_name)
    train_data = dataset[train_subset].select_columns("text")
    test_data = dataset[eval_subset].select_columns("text")

    return train_data, test_data


def train_model(model, train_data, test_data, tokenizer, training_args):
    """
    Setup the trainer and execute training.
    """
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[CustomWandbCallback],
    )

    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.tokenizer.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if training_args.push_to_hub:
        output_dir = training_args.output_dir
        trainer.model.push_to_hub(output_dir)


def main(config_path):
    """
    Main function to orchestrate the training process using a config file.
    """
    config = load_config(config_path).get_config

    model, tokenizer = setup_model_and_tokenizer(
        config("model.model_id"), config("model.is_cosmoe")
    )
    train_data, test_data = prepare_datasets(
        config("dataset.name"),
        config("dataset.train_subset"),
        config("dataset.eval_subset"),
    )
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    train_data = train_data.shuffle(seed=23).map(
        lambda data_point: tokenize_prompt(
            data_point, tokenizer=tokenizer, cutoff_len=1024
        )
    )
    test_data = test_data.shuffle(seed=23).map(
        lambda data_point: tokenize_prompt(
            data_point, tokenizer=tokenizer, cutoff_len=1024
        )
    )
    training_args = TrainingArguments(**config("training"))
    train_model(model, train_data, test_data, tokenizer, training_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model based on a configuration file."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the Python configuration file."
    )
    args = parser.parse_args()

    main(args.config_path)
