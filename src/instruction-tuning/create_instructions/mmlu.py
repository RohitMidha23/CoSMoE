"""
Convert the MMLU dataset to the Mistral format.
"""

from datasets import DatasetDict, load_dataset

answers = ["A", "B", "C", "D", "E"]


def convert_to_mistral_format(example):
    # Constructing the choices text
    choices_text = "\n".join(
        [f"{answers[i]}. {text}" for i, text in enumerate(example["choices"])]
    )
    prefix_text = "The following are multiple choice questions (with answers)"
    if example["subject"]:
        prefix_text += f" about {example['subject']}"
    prefix_text += "."

    # Constructing the prompt in the required format
    prompt = f"<s>[INST]{prefix_text}\n\nQuestion:{example['question']}\n{choices_text}\nAnswer: [/INST]{answers[example['answer']]} </s>"
    return {"text": prompt}


train = load_dataset("cais/mmlu", "all", split="auxiliary_train")
validation = load_dataset("cais/mmlu", "all", split="dev+validation")
test = load_dataset("cais/mmlu", "all", split="test")

train = train.map(convert_to_mistral_format)
validation = validation.map(convert_to_mistral_format)
test = test.map(convert_to_mistral_format)


dataset = DatasetDict({"train": train, "validation": validation, "test": test})
dataset.push_to_hub("jungledude23/mmlu-mistral-instruct")
