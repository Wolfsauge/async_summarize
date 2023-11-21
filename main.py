#!/usr/bin/env python3

import os
import sys
import argparse
import asyncio
from time import perf_counter
from dataclasses import dataclass
from icecream import ic  # type: ignore

from sync_helpers import (
    get_buck_slip_config,
    get_prompt_template,
    get_tokenizer,
    get_api_client,
    get_jinja2_environment,
    get_file_contents,
    get_output_filename,
    insert_buckslip_into_result,
    write_output_file,
)
from async_helpers import do_the_work


# Dataclass for commandline arguments
@dataclass
class CommandlineArguments:
    config: str
    prompt: str
    mode: str
    file: str


async def main(my_args: CommandlineArguments) -> None:
    time_t0: float
    time_t1: float
    time_delta: float
    summarize_duration: float
    result: dict

    # Check if all files given on the command line do exist
    # error out if not.
    # tbc.

    # Check if summarizing mode is understood.
    mode = my_args.mode
    while mode not in ["hm"]:
        ic("ERROR: mode {mode} not implemented.")
        sys.exit(1)

    # Initialize buck_slip dict
    config_filename = my_args.config
    buck_slip = get_buck_slip_config(config_filename)
    buck_slip["config_filename"] = config_filename

    # Get prompt_template
    prompt_template_filename = my_args.prompt
    buck_slip["prompt_templates"] = get_prompt_template(prompt_template_filename)
    buck_slip["prompt_template_filename"] = prompt_template_filename

    # Get tokenizer
    buck_slip["tokenizer"] = get_tokenizer(buck_slip)

    # Get OpenAI-compatible API
    buck_slip["api_client"] = get_api_client(buck_slip)

    # Get Jinja2 environment
    buck_slip["jinja2_env"] = get_jinja2_environment()

    # Get lock
    buck_slip["lock"] = asyncio.Lock()

    # Determine input file name
    input_filename = my_args.file
    buck_slip["input_filename"] = os.path.basename(input_filename)
    ic(input_filename)

    # Get the input
    sample_text = get_file_contents(my_args.file, buck_slip)

    # Determine output file name
    output_filename = get_output_filename(input_filename, buck_slip)

    # Initialize runtime variables
    result = {}
    ic("Init complete.")

    # Measure beginning of recursive summarization
    time_t0 = perf_counter()

    # Enter recursion
    summary = await do_the_work(sample_text, buck_slip, mode)

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
        "-m",
        "--mode",
        type=str,
        default="hm",
        help="mode (default hm)",
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
