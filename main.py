#!/usr/bin/env python3

import sys

import argparse
import asyncio

from time import perf_counter
from dataclasses import dataclass

from icecream import ic  # type: ignore

from sync_helpers import (
    get_yaml_config,
    get_text_splitter,
    write_output_file,
    get_output_filename,
)

from async_helpers import (
    # get_httpx_client,
    enter_recursion,
    get_file_contents,
    get_tokenizer,
    get_api_client,
)


# command line dataclass
@dataclass
class CommandlineArguments:
    file: str


async def main(my_args: CommandlineArguments) -> None:
    time_t0: float
    time_t1: float
    time_delta: float
    duration: str
    result: dict

    # Determine input file name
    input_filename = my_args.file
    ic(input_filename)

    # Read input file
    sample_text = await get_file_contents(my_args.file)

    # Initialize my_config buck slip dict
    my_config = get_yaml_config("config.yaml")

    # Determine output file name
    output_filename = get_output_filename(input_filename, my_config)
    ic(output_filename)

    # Enable fast tokenizer
    tokenizer = await get_tokenizer(my_config)
    encoding = tokenizer("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    ic(type(tokenizer))
    ic(tokenizer.is_fast)
    ic(encoding.is_fast)
    if tokenizer.is_fast is not True or encoding.is_fast is not True:
        sys.exit(1)
    my_config["tokenizer"] = tokenizer

    # Enable text splitter
    my_config["text_splitter"] = get_text_splitter(my_config)

    # Enable OpenAI-compatible API
    my_config["api_client"] = await get_api_client(my_config)

    # Measure beginning of recursive summarization
    time_t0 = perf_counter()

    # Enter recursion
    result = {}
    depth = 0
    result["summary"] = await enter_recursion(sample_text, depth, my_config)

    # Measure ending of recursive summarization
    time_t1 = perf_counter()
    time_delta = time_t1 - time_t0
    duration = f"{time_delta:.2f} seconds"

    # Shutdown httpx AsyncClient
    # await my_config["my_httpx_client"].aclose()

    # Create result dictionary entries
    result["duration"] = duration
    result["model_identifier"] = my_config["model_identifier"]
    result["chunk_size"] = my_config["chunk_size"]
    result["chunk_overlap"] = my_config["chunk_overlap"]
    result["max_tokens"] = my_config["max_tokens"]
    result["api_url"] = my_config["api_url"]

    # Peek at result
    ic(result)

    # Write the output file
    write_output_file(output_filename, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="myfile.txt",
        help="inspect file (default myfile.txt)",
    )
    parsed_args = CommandlineArguments(**vars(parser.parse_args()))
    asyncio.run(main(parsed_args))
