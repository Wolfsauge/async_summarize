import asyncio
import httpx

from icecream import ic  # type: ignore
from transformers import AutoTokenizer, LlamaTokenizerFast  # type: ignore
from openai import AsyncOpenAI

from sync_helpers import get_length_of_chunk_in_tokens


async def get_result(my_chunk: str, buck_slip: dict) -> str:
    #     my_prompt = f"""BEGININPUT
    # {my_chunk}
    # ENDINPUT
    # BEGININSTRUCTION
    # Summarize the input in about 130 words.
    # ENDINSTRUCTION
    # """

    #     my_prompt = f"""BEGININPUT
    # {my_chunk}
    # ENDINPUT
    # BEGININSTRUCTION
    # Summarize the input in about 130 words, focusing on characters, actions and events. Infer the appearance and personality of the characters involved in a few sentences, if they are mentioned in the text. Write confidently even if character qualities are vague or poorly-defined. Keep your response in one paragraph.
    # ENDINSTRUCTION
    # """

    my_prompt = f"""BEGININPUT
{my_chunk}
ENDINPUT
BEGININSTRUCTION
Summarize the input in about 130 words, focusing on characters, actions and events. Infer the scene description, the appearance and personality of the characters involved and write confidently and leave everything out, which is not well defined in the input. Keep your response in one paragraph.
ENDINSTRUCTION
"""

    completion = await buck_slip["api_client"].completions.create(
        model=buck_slip["model_local_identifier"],
        prompt=my_prompt,
        max_tokens=buck_slip["max_tokens"],
    )

    return completion.choices[0].text


async def get_async_chunking(my_chunk: str, buck_slip: dict) -> list:
    my_chunks = buck_slip["text_splitter"].split_text(my_chunk)
    return my_chunks


async def enter_recursion(my_chunk: str, recursion_depth: int, buck_slip: dict) -> str:
    recursion_depth += 1
    ic(recursion_depth)

    # length_of_chunk_in_chars = get_length_of_chunk_in_chars(my_chunk)
    length_of_chunk_in_tokens = get_length_of_chunk_in_tokens(my_chunk, buck_slip)

    if length_of_chunk_in_tokens >= buck_slip["chunk_size"]:
        # we need to split
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
        # we can summarize
        intermediate_result = await get_result(my_chunk, buck_slip)
        ic(len(str(intermediate_result)))

    my_result = str(intermediate_result).strip()

    return my_result


async def get_file_contents(my_filename: str) -> str:
    with open(my_filename, "r", encoding="utf-8") as my_fp:
        return my_fp.read()


async def get_tokenizer(buck_slip: dict) -> LlamaTokenizerFast:
    my_tokenizer = AutoTokenizer.from_pretrained(
        buck_slip["model_identifier"], use_fast=True
    )
    return my_tokenizer


async def get_api_client(buck_slip: dict) -> AsyncOpenAI:
    my_max_keepalive_connections = int(buck_slip["httpx_max_keepalive_connections"])
    my_max_connections = int(buck_slip["httpx_max_connections"])
    limits = httpx.Limits(
        max_keepalive_connections=my_max_keepalive_connections,
        max_connections=my_max_connections,
    )
    timeout = httpx.Timeout(600.0, connect=60.0)

    client = AsyncOpenAI(
        api_key=buck_slip["api_key"],
        base_url=buck_slip["api_url"],
        http_client=httpx.AsyncClient(limits=limits, timeout=timeout),
    )
    return client
