def tokenize(prompt, tokenizer, cutoff_len=1024, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=True,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def tokenize_prompt(data_point, tokenizer, cutoff_len=1024):
    tokenized_prompt = tokenize(
        data_point["text"], tokenizer, cutoff_len, add_eos_token=True
    )
    return tokenized_prompt
