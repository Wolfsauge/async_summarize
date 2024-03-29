#!/usr/bin/env poetry run python3

# Feature requests:
# - ftfy
# - read PDF

import sys
import json
from dataclasses import dataclass
from typing import Any
from collections import OrderedDict
import argparse
from pathlib import Path
import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv  # type: ignore
import yaml
import jinja2
import httpx
from openai import AsyncOpenAI, types
from semantic_text_splitter import HuggingFaceTextSplitter  # type: ignore
from tokenizers import Tokenizer  # type: ignore
from tqdm import tqdm  # type: ignore

from icecream import ic  # type: ignore


@dataclass
class CommandlineArguments:
    url: str
    config: str
    prompt: str
    file: str
    output: str


@dataclass
class BuckSlip:
    shared_config: dict
    jinja2_env: jinja2.environment.Environment | None = None
    httpx_client: httpx.AsyncClient | None = None
    api_client: AsyncOpenAI | None = None


def get_prompt_template(prompt_template_filename: str) -> str:
    try:
        ic(prompt_template_filename)
        with open(prompt_template_filename, "r", encoding="utf-8-sig") as file:
            prompt_template = yaml.safe_load(file)
            prompt_template = prompt_template["prompt_templates"]
        ic(prompt_template)

    except (IOError, OSError) as exception:
        ic(exception)
        ic("Exit.")
        sys.exit(1)

    return prompt_template


def get_jinja2_environment() -> jinja2.environment.Environment:
    ic("Initializing Jinja2 environment.")
    jinja2_env = jinja2.Environment()

    return jinja2_env


def get_httpx_client(buckslip: BuckSlip) -> BuckSlip:
    limits = httpx.Limits(
        max_keepalive_connections=buckslip.shared_config["max_keepalive_connections"],
        max_connections=buckslip.shared_config["max_connections"],
    )

    buckslip.httpx_client = httpx.AsyncClient(
        base_url=buckslip.shared_config["api_base_url"],
        limits=limits,
        timeout=None,
    )

    return buckslip


async def get_hf_model_id(buckslip: BuckSlip) -> BuckSlip:
    ic("Discovering API URL.")
    if buckslip.httpx_client is not None:
        response = await buckslip.httpx_client.get("/models")

        if response.status_code == httpx.codes.OK:
            response_dict = json.loads(response.text)
            hf_model_id = response_dict["data"][0]["id"]
            buckslip.shared_config["hf_model_id"] = hf_model_id
        else:
            print(f"Received HTTP {response.status_code} on {response.url}.")
            print("Exit.")
            await buckslip.httpx_client.aclose()
            sys.exit(1)
        ic(hf_model_id)
    else:
        print("ERROR: Can't use httpx client.")
        print("Exit.")
        sys.exit(1)

    return buckslip


def get_api_key(variable_name: str) -> str | None:
    load_dotenv()
    api_key = os.getenv(variable_name)

    if api_key is None:
        print("ERROR: api key cannot be determined.")
        print("Exit.")
        sys.exit(1)

    return api_key


def get_api_client(buckslip: BuckSlip) -> BuckSlip:
    buckslip.api_client = AsyncOpenAI(
        api_key=buckslip.shared_config["api_key"],
        base_url=buckslip.shared_config["api_base_url"],
        http_client=buckslip.httpx_client,
    )

    return buckslip


def get_semantic_text_splitter(buckslip: BuckSlip) -> HuggingFaceTextSplitter:
    tokenizer = Tokenizer.from_pretrained(
        buckslip.shared_config["semantic_splitter_model"]
    )

    text_splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=True)
    ic(type(text_splitter))

    return text_splitter


def create_semantic_chunking(
    input_chunk: str, chunk_size: tuple, buckslip: BuckSlip
) -> list:
    chunking = get_semantic_text_splitter(buckslip).chunks(
        input_chunk, chunk_capacity=chunk_size
    )

    return chunking


def build_prompt(
    chunk: str,
    context: str,
    stage: str,
    buckslip: BuckSlip,
) -> str:
    if buckslip.jinja2_env is not None:
        template = buckslip.jinja2_env.from_string(
            buckslip.shared_config["prompt_templates"][stage]
        )

        prompt = template.render(
            text=chunk,
            context=context,
        )
    else:
        print("ERROR: Can't use Jinja2.")
        print("Exit.")
        sys.exit(1)

    return prompt


async def completions_create(
    prompt: str, temperature: float, buckslip: BuckSlip
) -> types.Completion:
    if buckslip.api_client is not None:
        completion = await buckslip.api_client.completions.create(
            model=buckslip.shared_config["hf_model_id"],
            prompt=prompt,
            max_tokens=buckslip.shared_config["max_tokens"],
            temperature=temperature,
            stop=buckslip.shared_config["stop_sequence"],
        )

    return completion


async def create_single_generation(
    chunk_index: int,
    chunk_content: str,
    buckslip: BuckSlip,
    stage: str,
    context: str = "",
) -> tuple[dict, int]:
    prompt = build_prompt(
        chunk_content,
        context,
        stage,
        buckslip,
    )

    bad_counter = 0
    attempt_counter = 0

    completion_attempts = []

    while attempt_counter <= buckslip.shared_config["max_retries"]:
        temperature = buckslip.shared_config["temperature"] + attempt_counter * 0.1
        completion = await completions_create(prompt, temperature, buckslip)

        completion_finish_reason = completion.choices[0].finish_reason

        if completion.usage is not None:
            last_generation_attempt = {
                "chunk_index": chunk_index,
                "completion_id": completion.id,
                "max_tokens": int(buckslip.shared_config["max_tokens"]),
                "temperature": temperature,
                "finish_reason": completion_finish_reason,
                "completion_text": completion.choices[0].text.strip(),
                "completion_tokens": completion.usage.completion_tokens,
                "completion_prompt_tokens": completion.usage.prompt_tokens,
                "completion_total_tokens": completion.usage.total_tokens,
                "attempt": attempt_counter,
            }
        else:
            print("ERROR: API completions are lacking metrics.")
            print("Exit.")
            sys.exit(1)

        completion_attempts.append(last_generation_attempt)
        attempt_counter += 1

        if completion_finish_reason == "stop":
            tqdm.write(f"INFO: finish_reason == 'stop'. Saving chunk #{chunk_index}.")
            break
        bad_counter += 1
        tqdm.write(f"ERROR: finish_reason != 'stop', retrying chunk #{chunk_index}.")

    last_generation_attempt.update({"prompt": prompt})
    # last_generation_attempt.update({"attempts": completion_attempts.copy()})

    return last_generation_attempt, chunk_index


def change_to_sessiondir(my_sessiondir) -> None:
    if not Path(my_sessiondir).is_dir():
        os.mkdir(my_sessiondir)
    if os.getcwd() != my_sessiondir:
        os.chdir(my_sessiondir)


def dump_data_to_json_file(
    my_data: list | dict, my_filename: str, buckslip: BuckSlip
) -> None:
    change_to_sessiondir(buckslip.shared_config["sessiondir"])

    with open(my_filename, "w", encoding="utf-8-sig") as json_file:
        json.dump(my_data, json_file)
    tqdm.write(f"Wrote {len(my_data)} elements to file {my_filename}.")


def read_list_from_json_file(my_filename: str) -> list:
    with open(my_filename, "r", encoding="utf-8-sig") as json_file:
        my_list = json.load(json_file)
    tqdm.write(f"Loaded {len(my_list)} elements from file {my_filename}.")
    return my_list


async def compute_first_pass(buckslip: BuckSlip) -> list:
    # Semantic chunking
    chunks = create_semantic_chunking(
        buckslip.shared_config["input_text"],
        (
            buckslip.shared_config["semantic_splitter_lo_threshold"],
            buckslip.shared_config["semantic_splitter_hi_threshold"],
        ),
        buckslip,
    )

    # Prepare generations
    generations: list[dict[Any, Any]]
    generations = [{} for x in range(len(chunks))]

    # Generate asyncio task list
    asyncio_task_list = [
        create_single_generation(chunk_index, chunk_content, buckslip, "first_pass")
        for chunk_index, chunk_content in enumerate(chunks)
    ]

    # Await the completion of asyncio tasks
    for my_task in tqdm(
        asyncio.as_completed(asyncio_task_list),
        total=len(asyncio_task_list),
    ):
        result, chunk_index = await my_task

        # Store the results
        generations[chunk_index] = result

    return generations


async def compute_second_pass(buckslip: BuckSlip) -> list:
    # Initialize empty context (sequential)
    context = "No previous context existing."

    # Acquire chunks
    chunks = []
    for element in buckslip.shared_config["first_pass_generations"]:
        chunks.append(str(element["completion_text"]))

    # Prepare generations
    generations: list[dict[Any, Any]]
    generations = [{} for x in range(len(chunks))]

    # Iterate through the chunks sequentially
    for chunk_index, chunk_content in enumerate(chunks):
        if chunk_index == 0:
            # Initialize context with the first (or only) element
            tqdm.write("Using first chunk as initial context.")
            context = chunk_content
        else:
            # Generate on chunk with context
            tqdm.write(f"Merging context with chunk #{chunk_index}.")
            result, _ = await create_single_generation(
                chunk_index,
                chunk_content,
                buckslip,
                "second_pass",
                context,
            )
            # Set context for the next generation
            context = result["completion_text"]

            # Store the results
            generations[chunk_index] = result

    return generations


async def compute_pass(buckslip: BuckSlip, stage: str) -> None:
    output_filename = f"generations-{stage}.json"

    if Path(output_filename).is_file():
        if stage == "first_pass":
            buckslip.shared_config["first_pass_generations"] = read_list_from_json_file(
                output_filename
            )
        elif stage == "second_pass":
            buckslip.shared_config[
                "second_pass_generations"
            ] = read_list_from_json_file(output_filename)
    else:
        if stage == "first_pass":
            generations = await compute_first_pass(buckslip)
            buckslip.shared_config["first_pass_generations"] = generations
        elif stage == "second_pass":
            generations = await compute_second_pass(buckslip)
            buckslip.shared_config["second_pass_generations"] = generations

        # Write the task results to a JSON file
        dump_data_to_json_file(generations, output_filename, buckslip)


def show_some_intermediate_results(buckslip: BuckSlip) -> None:
    list_of_lengths_in_tokens = []
    for element in buckslip.shared_config["first_pass_generations"]:
        list_of_lengths_in_tokens.append(str(element["completion_tokens"]))
    my_string = ", ".join(list_of_lengths_in_tokens)
    print(f"Lengths of first pass chunks in units of tokens: [{my_string}].")


def read_input_file(input_filename: str) -> str:
    ic("Reading input file.")

    try:
        with open(input_filename, "r", encoding="utf-8-sig") as input_fp:
            input_text = input_fp.read()

    except (IOError, OSError) as exception:
        ic(exception)
        print("Exit.")
        sys.exit(1)

    if len(input_text) == 0:
        print("Error, input file is empty.")
        print("Exit.")
        sys.exit(1)

    return input_text


def get_input_file(input_filename: str, buckslip: BuckSlip) -> BuckSlip:
    buckslip.shared_config["input_text"] = read_input_file(input_filename)
    buckslip.shared_config["input_filename"] = os.path.basename(input_filename)

    return buckslip


def normalize_filename(filename: str) -> str:
    normalized_filename = os.path.basename(filename)
    normalized_filename = normalized_filename.replace(" ", "_")
    normalized_filename, _ = os.path.splitext(normalized_filename)

    return normalized_filename


def determine_sessiondir(buckslip: BuckSlip, output_dir: str) -> BuckSlip:
    now_datetime = datetime.now()
    now_isoformat = now_datetime.isoformat()
    buckslip.shared_config["date"] = now_isoformat

    my_normalized_name = normalize_filename(buckslip.shared_config["input_filename"])

    output_dir = os.path.join(os.getcwd(), output_dir)
    ic(output_dir)
    if not Path(output_dir).is_dir():
        os.mkdir(output_dir)

    my_workdir = str.join("-", (my_normalized_name, now_isoformat))
    my_sessiondir = os.path.join(output_dir, my_workdir)

    buckslip.shared_config["sessiondir"] = my_sessiondir
    buckslip.shared_config["session"] = my_workdir

    return buckslip


def get_shared_config(my_args: CommandlineArguments) -> dict:
    try:
        ic(my_args.config)
        with open(my_args.config, "r", encoding="utf-8-sig") as file:
            shared_config = yaml.safe_load(file)
            shared_config = shared_config["async_summarize_shared_config"]

    except (IOError, OSError) as exception:
        ic(exception)
        ic("Exit.")
        sys.exit(1)

    shared_config["api_base_url"] = my_args.url

    return shared_config


def dump_buckslip(buckslip: BuckSlip) -> None:
    change_to_sessiondir(buckslip.shared_config["sessiondir"])

    exclude_keys = {"api_key", "input_text", "sessiondir"}
    dumpslip = OrderedDict(
        sorted(
            {
                k: buckslip.shared_config[k]
                for k in set(buckslip.shared_config.keys()).difference(exclude_keys)
            }.items()
        )
    )

    dump_data_to_json_file(dumpslip, "buckslip.json", buckslip)


async def get_buckslip(my_args: CommandlineArguments) -> BuckSlip:
    # Get configuration from file
    shared_config = get_shared_config(my_args)

    # Initial creation of buck slip object
    buckslip = BuckSlip(shared_config)

    # Get API key from .env variable
    buckslip.shared_config["api_key"] = get_api_key("MY_ENV_VAR")

    # Get prompt_template from file
    prompt_template_filename = my_args.prompt
    buckslip.shared_config["prompt_templates"] = get_prompt_template(
        prompt_template_filename
    )
    buckslip.shared_config["prompt_template_filename"] = prompt_template_filename

    # Get input file and output directory
    buckslip = get_input_file(my_args.file, buckslip)
    buckslip = determine_sessiondir(buckslip, my_args.output)

    # Get Jinja2 environment
    buckslip.jinja2_env = get_jinja2_environment()

    # Get httpx client
    buckslip = get_httpx_client(buckslip)

    # Discover hf_model_id
    buckslip = await get_hf_model_id(buckslip)

    # Get OpenAI-compatible API client
    buckslip = get_api_client(buckslip)

    # Dump buck slip
    dump_buckslip(buckslip)

    return buckslip


async def main(my_args: CommandlineArguments) -> None:
    # Get buckslip
    buckslip = await get_buckslip(my_args)

    # Do some work
    await compute_pass(buckslip, "first_pass")

    # Show some intermediate results
    show_some_intermediate_results(buckslip)

    # Do some more work
    await compute_pass(buckslip, "second_pass")

    # Show final result
    if len(buckslip.shared_config["second_pass_generations"]) > 1:
        output_value = buckslip.shared_config["second_pass_generations"][
            len(buckslip.shared_config["second_pass_generations"]) - 1
        ]["completion_text"]
        print(output_value)
        with open("summary.txt", "w", encoding="utf-8") as text_file:
            print(f"{output_value}", file=text_file)

    # Close httpx client
    if buckslip.httpx_client is not None:
        await buckslip.httpx_client.aclose()

    # Delete buck slip object just for fun
    del buckslip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of OpenAI-compatible API (default: http://localhost:8000/v1).",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="configuration file (default config.yaml)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="prompt.yaml",
        help="prompt template file (default prompt.yaml)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="input.txt",
        help="input file (default: input.txt).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="output directory (default: ./output).",
    )
    parsed_args = CommandlineArguments(**vars(parser.parse_args()))
    asyncio.run(main(parsed_args))
    sys.exit(0)
