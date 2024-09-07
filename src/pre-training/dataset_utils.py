# This function will ensure that each input is added with eos_token and chunked
# to the context length size.
def tokenize(elements, tokenizer, context_length=1024):
    # Tokenize all elements in the batch without truncation
    outputs = tokenizer(
        elements["text"],
        truncation=False,
        return_attention_mask=False,
    )

    all_token_ids = []
    for input_ids in outputs["input_ids"]:
        all_token_ids.extend(input_ids + [tokenizer.eos_token_id])

    # Chunk the concatenated token IDs
    chunks = []
    for i in range(0, len(all_token_ids), context_length):
        chunk = all_token_ids[i : i + context_length]
        if len(chunk) == context_length:
            chunks.append(chunk)

    return {"input_ids": chunks}
