import os
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset, Value, concatenate_datasets, DatasetDict


# Tiny Codes
tiny_codes = load_dataset("nampdn-ai/tiny-codes")

# filter and select rows which have programming language python
tiny_codes["train"] = tiny_codes["train"].filter(
    lambda x: x["programming_language"] == "python"
)

split_ratio = 0.1
split_index = int(len(tiny_codes["train"]) * split_ratio)

# Formatting tiny_codes
tiny_codes_val = tiny_codes["train"].select(range(split_index))
tiny_codes_test = tiny_codes["train"].select(range(split_index, 2 * split_index))
tiny_codes_train = tiny_codes["train"].select(
    range(2 * split_index, len(tiny_codes["train"]))
)


tiny_codes_train = tiny_codes_train.map(
    lambda x: {
        "text": f"{x['prompt']}\n###Response:{x['response']}",
        "source_dataset": "tiny_codes",
    }
)

tiny_codes_val = tiny_codes_val.map(
    lambda x: {
        "text": f"{x['prompt']}\n###Response:{x['response']}",
        "source_dataset": "tiny_codes",
    }
)

tiny_codes_test = tiny_codes_test.map(
    lambda x: {
        "text": f"{x['prompt']}\n###Response:{x['response']}",
        "source_dataset": "tiny_codes",
    }
)

tiny_codes_train = tiny_codes_train.select_columns(["text", "source_dataset"])
tiny_codes_train = tiny_codes_train.cast_column("text", Value("large_string"))

tiny_codes_val = tiny_codes_val.select_columns(["text", "source_dataset"])
tiny_codes_val = tiny_codes_val.cast_column("text", Value("large_string"))

tiny_codes_test = tiny_codes_test.select_columns(["text", "source_dataset"])
tiny_codes_test = tiny_codes_test.cast_column("text", Value("large_string"))

# Tiny Textbooks
tiny_tbs = load_dataset("nampdn-ai/tiny-textbooks")

split_ratio = 0.1
split_index = int(len(tiny_codes["train"]) * split_ratio)

tiny_tbs_val = tiny_tbs["train"].select(range(split_index))
tiny_tbs_test = tiny_tbs["train"].select(range(split_index, 2 * split_index))
tiny_tbs_train = tiny_tbs["train"].select(
    range(2 * split_index, len(tiny_tbs["train"]))
)

tiny_tbs_train = tiny_tbs["train"].map(
    lambda x: {"text": x["text"], "source_dataset": "tiny_tbs"}
)
tiny_tbs_val = tiny_tbs_val.map(
    lambda x: {"text": x["text"], "source_dataset": "tiny_tbs"}
)
tiny_tbs_test = tiny_tbs_test.map(
    lambda x: {"text": x["text"], "source_dataset": "tiny_tbs"}
)


tiny_tbs_train = tiny_tbs_train.select_columns(["text", "source_dataset"])
tiny_tbs_val = tiny_tbs_val.select_columns(["text", "source_dataset"])
tiny_tbs_test = tiny_tbs_test.select_columns(["text", "source_dataset"])


tiny_tbs_train = tiny_tbs_train.cast_column("text", Value("large_string"))
tiny_tbs_val = tiny_tbs_val.cast_column("text", Value("large_string"))
tiny_tbs_test = tiny_tbs_test.cast_column("text", Value("large_string"))

combined_train = concatenate_datasets([tiny_codes_train, tiny_tbs_train])
combined_val = concatenate_datasets([tiny_codes_val, tiny_tbs_val])
combined_test = concatenate_datasets([tiny_codes_test, tiny_tbs_test])


combined_dataset = DatasetDict(
    {"train": combined_train, "validation": combined_val, "test": combined_test}
)

combined_dataset.push_to_hub("jungledude23/tiny-datasets-combined")
