"""
Convert the ARC dataset to the required format for Mistral.
"""

from datasets import load_dataset


def convert_to_mistral_format(example):
    # Constructing the choices text
    choices_text = " ".join(
        [
            f"({label}) {text}"
            for label, text in zip(
                example["choices"]["label"], example["choices"]["text"]
            )
        ]
    )
    prefix_text = "Below is an question that is followed by multiple choices. Provide an answer to the question.\\n"
    # Constructing the prompt in the required format
    prompt = f"<s> [INST]{prefix_text}{example['question']} {choices_text} [/INST]\nAnswer: {example['answerKey']} </s>"
    return {"text": prompt}


dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")
transformed_dataset = dataset.map(convert_to_mistral_format)
transformed_dataset.push_to_hub("jungledude23/arc-easy-mistral-instruct")

dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
transformed_dataset = dataset.map(convert_to_mistral_format)
transformed_dataset.push_to_hub("jungledude23/arc-challenge-mistral-instruct")
