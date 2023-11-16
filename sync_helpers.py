import yaml
import json
from icecream import ic  # type: ignore

from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_yaml_config(my_config_filename: str) -> dict:
    my_config = {
        "httpx_max_connections": 1,
        "httpx_max_keepalive_connections": 1,
        "model_identifier": "empty",
        "api_key": "empty",
    }

    try:
        with open(my_config_filename, "r", encoding="utf-8") as file:
            my_config = yaml.safe_load(file)
        ic(len(my_config))

    except (IOError, OSError) as my_exception:
        warning_message = f"{my_exception}"
        ic(warning_message)
    my_config["model_local_identifier"] = str(my_config["model_identifier"]).replace(
        "/", "_"
    )

    return my_config


def get_length_of_chunk_in_tokens(my_chunk: str, my_config: dict) -> int:
    result = my_config["tokenizer"](my_chunk)
    my_input_ids = result.input_ids

    return len(my_input_ids)


def get_text_splitter(my_config: dict):
    my_text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "."],
        chunk_size=my_config["chunk_size"],
        chunk_overlap=my_config["chunk_overlap"],
        length_function=lambda x: get_length_of_chunk_in_tokens(x, my_config),
        is_separator_regex=False,
    )
    return my_text_splitter


def write_output_file(output_filename: str, data: dict) -> None:
    ic(output_filename)
    with open(output_filename, "w", encoding="utf-8") as my_fp:
        json.dump(data, my_fp)
