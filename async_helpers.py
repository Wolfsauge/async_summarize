import sys
import asyncio
import httpx
import jinja2
import rich.progress

from icecream import ic  # type: ignore
from transformers import AutoTokenizer, LlamaTokenizerFast  # type: ignore
from openai import AsyncOpenAI

from sync_helpers import get_length_of_chunk_in_tokens


async def get_result(my_chunk: str, buck_slip: dict) -> str:
    environment = jinja2.Environment()
    template = environment.from_string(str(buck_slip["prompt_template"]))
    my_prompt = template.render(prompt=my_chunk)

    completion = await buck_slip["api_client"].completions.create(
        model=buck_slip["model_local_identifier"],
        prompt=my_prompt,
        max_tokens=buck_slip["max_tokens"],
        temperature=buck_slip["temperature"],
    )

    return completion.choices[0].text


async def get_async_chunking(my_chunk: str, buck_slip: dict) -> list:
    chunks = buck_slip["text_splitter"].split_text(my_chunk)

    return chunks


async def enter_recursion(my_chunk: str, recursion_depth: int, buck_slip: dict) -> str:
    recursion_depth += 1
    ic(recursion_depth)

    # length_of_chunk_in_chars = get_length_of_chunk_in_chars(my_chunk)
    length_of_chunk_in_tokens = get_length_of_chunk_in_tokens(my_chunk, buck_slip)

    if length_of_chunk_in_tokens >= buck_slip["chunk_size"]:
        # We need to split
        chunks = await get_async_chunking(my_chunk, buck_slip)
        ic(len(chunks))

        partial_results = []

        partial_results = await asyncio.gather(
            *[
                enter_recursion(partial_chunk, recursion_depth, buck_slip)
                for partial_chunk in chunks
            ]
        )

        my_result_string = "\n".join(partial_results)

        intermediate_result = await enter_recursion(
            my_result_string, recursion_depth, buck_slip
        )
    else:
        # We can summarize
        intermediate_result = await get_result(my_chunk, buck_slip)
        ic(len(str(intermediate_result)))

    my_result = str(intermediate_result).strip()

    return my_result


async def get_file_contents(my_filename: str, buck_slip: dict) -> str:
    with rich.progress.open(my_filename, "r", encoding="utf-8") as my_fp:
        sample_text = my_fp.read()
    buck_slip["length_of_sample_text_in_characters"] = len(sample_text)
    ic(len(sample_text))

    return sample_text


async def get_tokenizer(buck_slip: dict) -> LlamaTokenizerFast:
    if buck_slip["use_fast"] is True:
        tokenizer = AutoTokenizer.from_pretrained(
            buck_slip["model_identifier"], use_fast=True
        )
        ic(type(tokenizer))
        ic(tokenizer.is_fast)
        encoding = tokenizer(
            "My name is Sylvain and I work at Hugging Face in Brooklyn."
        )
        ic(type(encoding))
        ic(encoding.is_fast)
        if tokenizer.is_fast is not True or encoding.is_fast is not True:
            sys.exit(1)
    else:
        ic("ERROR: use_fast = False not implemented")
        sys.exit(1)

    return tokenizer


async def get_api_client(buck_slip: dict) -> AsyncOpenAI:
    my_max_keepalive_connections = int(buck_slip["httpx_max_keepalive_connections"])
    my_max_connections = int(buck_slip["httpx_max_connections"])
    limits = httpx.Limits(
        max_keepalive_connections=my_max_keepalive_connections,
        max_connections=my_max_connections,
    )
    timeout = httpx.Timeout(600.0, connect=60.0)

    api_client = AsyncOpenAI(
        api_key=buck_slip["api_key"],
        base_url=buck_slip["api_url"],
        http_client=httpx.AsyncClient(limits=limits, timeout=timeout),
    )
    ic(type(api_client))

    return api_client
