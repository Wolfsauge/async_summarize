import os
import re
import json
import yaml

from icecream import ic  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


def get_yaml_config(buck_slip_filename: str) -> dict:
    buck_slip = {
        "httpx_max_connections": 1,
        "httpx_max_keepalive_connections": 1,
        "model_identifier": "empty",
        "api_key": "empty",
    }

    try:
        with open(buck_slip_filename, "r", encoding="utf-8") as file:
            buck_slip = yaml.safe_load(file)
        ic(buck_slip)

    except (IOError, OSError) as my_exception:
        warning_message = f"{my_exception}"
        ic(warning_message)
    buck_slip["model_local_identifier"] = str(buck_slip["model_identifier"]).replace(
        "/", "_"
    )

    return buck_slip


def get_length_of_chunk_in_tokens(my_chunk: str, buck_slip: dict) -> int:
    my_result = buck_slip["tokenizer"](my_chunk)
    input_ids = my_result.input_ids
    length_of_chunk_in_tokens = len(input_ids)

    return length_of_chunk_in_tokens


def get_text_splitter(buck_slip: dict) -> TextSplitter:
    batched_tokenization = buck_slip["use_batched_tokenization"]
    if batched_tokenization is True:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=buck_slip["tokenizer"],
            chunk_size=buck_slip["chunk_size"],
            chunk_overlap=buck_slip["chunk_overlap"],
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "."],
            chunk_size=buck_slip["chunk_size"],
            chunk_overlap=buck_slip["chunk_overlap"],
            length_function=lambda x: get_length_of_chunk_in_tokens(x, buck_slip),
            is_separator_regex=False,
        )
    ic(type(text_splitter))
    ic(batched_tokenization)

    return text_splitter


def get_output_filename(my_input_filename: str, buck_slip: dict) -> str:
    my_local_identifier = buck_slip["model_local_identifier"]
    replacement = f"-analysis-{my_local_identifier}.json"
    output_filename = os.path.basename(my_input_filename)
    output_filename = re.sub("\\.txt$", replacement, output_filename)

    return output_filename


def write_output_file(output_filename: str, data: dict) -> None:
    with open(output_filename, "w", encoding="utf-8") as my_fp:
        json.dump(data, my_fp)


def update_result(result: dict, buck_slip: dict) -> dict:
    result["model_identifier"] = buck_slip["model_identifier"]
    result["chunk_size"] = buck_slip["chunk_size"]
    result["chunk_overlap"] = buck_slip["chunk_overlap"]
    result["max_tokens"] = buck_slip["max_tokens"]
    result["api_url"] = buck_slip["api_url"]
    result["length_of_sample_text_in_characters"] = buck_slip[
        "length_of_sample_text_in_characters"
    ]

    return result
