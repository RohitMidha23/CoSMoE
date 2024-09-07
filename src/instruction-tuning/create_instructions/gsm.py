"""
Convert the GSM8K dataset to the required format for Mistral.
"""

from datasets import load_dataset


def convert_to_mistral_format(example):
    # Constructing the prompt in the required format
    prompt = f"<s>[INST] Question: {example['question']}\nAnswer: [/INST]{example['answer']} </s>"
    return {"text": prompt}


gsm = load_dataset("openai/gsm8k", "main")

gsm = gsm.map(convert_to_mistral_format)
gsm.push_to_hub("jungledude23/gsm8k-mistral-instruct")
