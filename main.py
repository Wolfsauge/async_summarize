#!/usr/bin/env python3

import argparse
import asyncio

from time import perf_counter
from dataclasses import dataclass
from icecream import ic  # type: ignore
from sync_helpers import (
    get_buck_slip_config,
    get_prompt_template,
    get_tokenizer,
    get_text_splitter,
    get_api_client,
    get_file_contents,
    get_output_filename,
    insert_buckslip_into_result,
    write_output_file,
)
from async_helpers import enter_recursion


# Dataclass for commandline arguments
@dataclass
class CommandlineArguments:
    config: str
    prompt: str
    file: str


async def main(my_args: CommandlineArguments) -> None:
    time_t0: float
    time_t1: float
    time_delta: float
    summarize_duration: float
    result: dict

    # Initialize buck_slip dict
    config_filename = my_args.config
    buck_slip = get_buck_slip_config(config_filename)
    buck_slip["config_filename"] = config_filename

    # Get prompt_template
    prompt_template_filename = my_args.prompt
    buck_slip["prompt_template"] = get_prompt_template(prompt_template_filename)
    buck_slip["prompt_template_filename"] = prompt_template_filename

    # Get tokenizer
    buck_slip["tokenizer"] = get_tokenizer(buck_slip)

    # Get text splitter
    buck_slip["text_splitter"] = get_text_splitter(buck_slip)

    # Get OpenAI-compatible API
    buck_slip["api_client"] = get_api_client(buck_slip)

    # Determine input file name
    input_filename = my_args.file
    buck_slip["input_filename"] = input_filename
    ic(input_filename)

    # Get the input
    sample_text = get_file_contents(my_args.file, buck_slip)

    # Determine output file name
    output_filename = get_output_filename(input_filename, buck_slip)

    # Initialize runtime variables
    result = {}
    recursion_depth = 0
    ic("Init complete.")

    # Measure beginning of recursive summarization
    time_t0 = perf_counter()

    # Enter recursion
    summary = await enter_recursion(sample_text, recursion_depth, buck_slip)

    # Measure ending of recursive summarization
    time_t1 = perf_counter()
    time_delta = time_t1 - time_t0
    summarize_duration = float(f"{time_delta:.2f}")
    buck_slip["summarize_duration_seconds"] = float(summarize_duration)
    ic(summarize_duration)
    ic(summary)

    # Create result dictionary
    result["summary"] = summary

    # Update result with buck slip information
    result = insert_buckslip_into_result(result, buck_slip)

    # Write the output file
    write_output_file(output_filename, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="use config file (default config.yaml)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="prompt.yaml",
        help="use prompt template (default prompt.yaml)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="myfile.txt",
        help="summarize file (default myfile.txt)",
    )
    parsed_args = CommandlineArguments(**vars(parser.parse_args()))
    asyncio.run(main(parsed_args))
