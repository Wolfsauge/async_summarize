import os
import sys
import re
import json
import yaml
import rich.progress

from icecream import ic  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


def get_buck_slip_config(buck_slip_filename: str) -> dict:
    buck_slip = {
        "httpx_max_connections": 1,
        "httpx_max_keepalive_connections": 1,
        "model_identifier": "empty",
        "api_key": "empty",
    }

    try:
        ic(buck_slip_filename)
        with rich.progress.open(buck_slip_filename, "r", encoding="utf-8") as file:
            buck_slip = yaml.safe_load(file)
        ic(buck_slip)

    except (IOError, OSError) as my_exception:
        warning_message = f"{my_exception}"
        ic(warning_message)
    buck_slip["model_local_identifier"] = str(buck_slip["model_identifier"]).replace(
        "/", "_"
    )

    return buck_slip


def get_prompt_template(prompt_template_filename: str) -> str:
    try:
        ic(prompt_template_filename)
        with rich.progress.open(prompt_template_filename, "r", encoding="utf-8") as file:
            prompt_template = yaml.safe_load(file)
            prompt_template = prompt_template["prompt_template"]
        ic(prompt_template)

    except (IOError, OSError) as my_exception:
        warning_message = f"{my_exception}"
        ic(warning_message)

    return prompt_template


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
            chunk_size=buck_slip["chunk_size"],
            chunk_overlap=buck_slip["chunk_overlap"],
            length_function=lambda x: get_length_of_chunk_in_tokens(x, buck_slip),
        )
    ic(type(text_splitter))

    return text_splitter


def get_output_filename(my_input_filename: str, buck_slip: dict) -> str:
    my_index = 0
    does_exist = False
    while my_index < 1000:
        my_index_str = f"{my_index:04d}"
        my_local_identifier = buck_slip["model_local_identifier"]
        replacement = f"-analysis-{my_local_identifier}-{my_index_str}.json"
        output_filename = os.path.basename(my_input_filename)
        output_filename = re.sub("\\.txt$", replacement, output_filename)
        does_exist = os.path.exists(output_filename)
        if does_exist is True:
            my_index += 1
        else:
            break

    if does_exist is True:
        ic("ERROR: Can't find output filename.")
        sys.exit(1)

    ic(output_filename)

    return output_filename


def write_output_file(output_filename: str, data: dict) -> None:
    with open(output_filename, "w", encoding="utf-8") as my_fp:
        json.dump(data, my_fp)
    ic(output_filename)


def update_result(result: dict, buck_slip: dict) -> dict:
    # Stringify runtime components of the buck slip for reference
    buck_slip["tokenizer"] = str(buck_slip["tokenizer"])
    buck_slip["text_splitter"] = str(buck_slip["text_splitter"])
    buck_slip["api_client"] = str(buck_slip["api_client"])

    # Add the finalized buck slip to the result dict
    result["buck_slip"] = buck_slip

    return result
