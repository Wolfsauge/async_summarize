#!/usr/bin/env poetry run python3

# Feature requests:
# - ftfy
# - read PDF

import sys
import json
from dataclasses import dataclass
from typing import Any
import argparse
from pathlib import Path
import asyncio

import os
from dotenv import load_dotenv  # type: ignore
import yaml
import jinja2
import httpx
from openai import AsyncOpenAI, types
from transformers import AutoTokenizer, LlamaTokenizerFast  # type: ignore
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from tqdm import tqdm  # type: ignore

from icecream import ic  # type: ignore


@dataclass
class CommandlineArguments:
    url: str
    prompt: str
    file: str


@dataclass
class BuckSlip:
    shared_config: dict
    jinja2_env: jinja2.environment.Environment | None = None
    httpx_client: httpx.AsyncClient | None = None
    tokenizer: LlamaTokenizerFast | None = None
    api_client: AsyncOpenAI | None = None


def get_prompt_template(prompt_template_filename: str) -> str:
    try:
        ic(prompt_template_filename)
        with open(prompt_template_filename, "r", encoding="utf-8-sig") as file:
            prompt_template = yaml.safe_load(file)
            prompt_template = prompt_template["prompt_templates"]
        # Validate prompts, otherwise error out
        ic(prompt_template)

    except (IOError, OSError) as exception:
        ic(exception)
        print("Exit.")
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


def get_tokenizer(buckslip: BuckSlip) -> BuckSlip:
    ic("Initializing Huggingface transformers tokenizer.")
    buckslip.tokenizer = AutoTokenizer.from_pretrained(
        buckslip.shared_config["hf_model_id"],
        use_fast=buckslip.shared_config["tokenizer_use_fast"],
    )
    tokenizer_is_fast = buckslip.tokenizer.is_fast
    buckslip.shared_config["tokenizer_is_fast"] = buckslip.tokenizer.is_fast
    ic(tokenizer_is_fast)

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


def get_text_splitter(
    buckslip, custom_chunk_size, custom_chunk_overlap
) -> TextSplitter:
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=buckslip.tokenizer,
        chunk_size=custom_chunk_size,
        chunk_overlap=custom_chunk_overlap,
    )

    return text_splitter


def create_custom_chunking(
    input_chunk: str, buckslip: BuckSlip, chunk_size, overlap
) -> list:
    chunking = get_text_splitter(buckslip, chunk_size, overlap).split_text(input_chunk)

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
    attempt_counter: int, prompt: str, buckslip: BuckSlip
) -> types.Completion:
    if buckslip.api_client is not None:
        completion = await buckslip.api_client.completions.create(
            model=buckslip.shared_config["hf_model_id"],
            prompt=prompt,
            max_tokens=buckslip.shared_config["max_tokens"],
            temperature=buckslip.shared_config["temperature"] + attempt_counter * 0.1,
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
        completion = await completions_create(attempt_counter, prompt, buckslip)

        completion_finish_reason = completion.choices[0].finish_reason

        if completion.usage is not None:
            last_generation_attempt = {
                "chunk_index": chunk_index,
                "completion_id": completion.id,
                "max_tokens": int(buckslip.shared_config["max_tokens"]),
                "temperature": float(
                    buckslip.shared_config["temperature"] + attempt_counter * 0.1
                ),
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


def dump_list_to_json_file(my_list: list, my_filename: str) -> None:
    with open(my_filename, "w", encoding="utf-8-sig") as json_file:
        json.dump(my_list, json_file)
    tqdm.write(f"Wrote {len(my_list)} chunks to file {my_filename}.")


def read_list_from_json_file(my_filename: str) -> list:
    with open(my_filename, "r", encoding="utf-8-sig") as json_file:
        my_list = json.load(json_file)
    tqdm.write(f"Loaded {len(my_list)} chunks from file {my_filename}.")
    return my_list


async def compute_first_pass(buckslip: BuckSlip) -> list:
    # Acquire chunks
    # chunks = create_custom_chunking(input_text, buckslip, 1024, 102)
    chunks = create_custom_chunking(
        buckslip.shared_config["input_text"], buckslip, 2048, 204
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
        dump_list_to_json_file(generations, output_filename)


def show_some_intermediate_results(buckslip: BuckSlip) -> None:
    list_of_lengths_in_tokens = []
    for element in buckslip.shared_config["first_pass_generations"]:
        list_of_lengths_in_tokens.append(str(element["completion_tokens"]))
    my_string = ", ".join(list_of_lengths_in_tokens)
    print(f"Lengths of first pass chunks in units of tokens: [{my_string}].")


def read_input_file(input_filename: str) -> str:
    ic("Reading input file.")
    ic(input_filename)

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


async def main(my_args: CommandlineArguments) -> None:
    # Default config
    shared_config = {
        "api_base_url": my_args.url,
        "hf_model_id": "",
        "tokenizer_use_fast": True,
        "tokenizer_is_fast": False,
        "max_retries": 20,
        "temperature": 0.0,
        "max_tokens": 6000,
        "max_keepalive_connections": 3,
        "max_connections": 30,
        "stop_sequence": ["ENDNOTES", "<|user|>", "\n\n\n\n"],
    }

    # Initial creation of buck slip object
    buckslip = BuckSlip(shared_config)

    # Get API key from .env variable
    buckslip.shared_config["api_key"] = get_api_key("MY_ENV_VAR")

    # Determine input file and read it
    buckslip.shared_config["input_text"] = read_input_file(my_args.file)

    # Get prompt_template from file
    prompt_template_filename = my_args.prompt
    buckslip.shared_config["prompt_templates"] = get_prompt_template(
        prompt_template_filename
    )
    buckslip.shared_config["prompt_template_filename"] = prompt_template_filename

    # Get Jinja2 environment
    buckslip.jinja2_env = get_jinja2_environment()

    # Get httpx client
    buckslip = get_httpx_client(buckslip)

    # Discover hf_model_id
    buckslip = await get_hf_model_id(buckslip)

    # Get tokenizer based on hf_model_id
    buckslip = get_tokenizer(buckslip)

    # Get OpenAI-compatible API client
    buckslip = get_api_client(buckslip)

    # Output buck slip
    ic("Buck slip")
    ic(buckslip)

    # Do some work
    await compute_pass(buckslip, "first_pass")

    # Show some intermediate results
    show_some_intermediate_results(buckslip)

    # Do some more work
    await compute_pass(buckslip, "second_pass")

    # Show final result
    print(
        buckslip.shared_config["second_pass_generations"][
            len(buckslip.shared_config["second_pass_generations"]) - 1
        ]["completion_text"]
    )

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
        help="API base url (default: http://localhost:8000/v1).",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="prompt.yaml",
        help="Jinja2 prompt template (default prompt.yaml)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="myfile.txt",
        help="Input file to summarize (default: myfile.txt).",
    )
    parsed_args = CommandlineArguments(**vars(parser.parse_args()))
    asyncio.run(main(parsed_args))
    sys.exit(0)
