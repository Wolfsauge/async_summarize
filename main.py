#!/usr/bin/env python3

import os
import argparse
import asyncio

import re

from time import perf_counter
from dataclasses import dataclass

from icecream import ic  # type: ignore

from sync_helpers import get_yaml_config, get_text_splitter, write_output_file
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

    input_filename = my_args.file
    ic(input_filename)

    # read input text file
    sample_text = await get_file_contents(my_args.file)

    # Initialize config
    my_config = get_yaml_config("config.yaml")

    # Add
    # my_config["my_httpx_client"] = await get_httpx_client(my_config)
    my_config["tokenizer"] = await get_tokenizer(my_config)
    my_config["text_splitter"] = get_text_splitter(my_config)
    my_config["api_client"] = await get_api_client(my_config)

    # Measure Begin
    time_t0 = perf_counter()

    # enter recursion
    result = {}
    depth = 0
    result["summary"] = await enter_recursion(sample_text, depth, my_config)

    # Measure End
    time_t1 = perf_counter()
    time_delta = time_t1 - time_t0
    duration = f"{time_delta:.2f} seconds"

    # Shutdown httpx AsyncClient
    # await my_config["my_httpx_client"].aclose()

    # Output results
    result["duration"] = duration
    result["model_identifier"] = my_config["model_identifier"]
    result["chunk_size"] = my_config["chunk_size"]
    result["chunk_overlap"] = my_config["chunk_overlap"]
    result["max_tokens"] = my_config["max_tokens"]
    result["api_url"] = my_config["api_url"]
    # ic(duration)
    # ic(len(result))
    ic(result)
    # model_local_identifier
    my_local_identifier = my_config["model_local_identifier"]
    replacement = f"-analysis-{my_local_identifier}.json"
    output_filename = os.path.basename(input_filename)
    output_filename = re.sub("\\.txt$", replacement, output_filename)
    ic(output_filename)
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
