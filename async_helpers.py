import sys
import asyncio
import jinja2
from icecream import ic  # type: ignore
from sync_helpers import get_length_of_chunk_in_tokens


async def get_completion(my_chunk: str, buck_slip: dict) -> str:
    environment = jinja2.Environment()
    template = environment.from_string(str(buck_slip["prompt_template"]))
    my_prompt = template.render(prompt=my_chunk)

    bad_counter = 0
    attempt_counter = 0

    while attempt_counter <= buck_slip["max_completion_retries"]:
        completion = await buck_slip["api_client"].completions.create(
            model=buck_slip["model_local_identifier"],
            prompt=my_prompt,
            max_tokens=buck_slip["max_tokens"],
            temperature=buck_slip["temperature"],
        )

        attempt_counter += 1

        finish_reason = completion.choices[0].finish_reason

        if finish_reason == "stop":
            break

        bad_counter += 1

        ic(completion)
        ic(attempt_counter)
        ic(bad_counter)
        ic(finish_reason)
        ic("ERROR: finish_reason != 'stop', retrying.")

    if bad_counter >= buck_slip["max_completion_retries"]:
        ic(completion)
        ic(attempt_counter)
        ic(bad_counter)
        ic(finish_reason)
        ic("ERROR: aborting after multiple failed attempts.")
        sys.exit(1)

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
        intermediate_result = await get_completion(my_chunk, buck_slip)
        ic(len(str(intermediate_result)))

    my_result = str(intermediate_result).strip()

    return my_result
