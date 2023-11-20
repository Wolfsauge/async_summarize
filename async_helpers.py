import sys
import asyncio
import math

import jinja2
from icecream import ic  # type: ignore
from sync_helpers import (
    get_length_of_chunk_in_tokens,
    get_text_splitter,
    grouped,
    power_log,
    find_longest_element_index,
)


async def get_completion(my_chunk: str, buck_slip: dict, task: str) -> str:
    environment = jinja2.Environment()

    if task is None:
        task = "summarize"

    template = environment.from_string(buck_slip["prompt_templates"][task])
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


async def get_completion_merge(
    first_element: str, second_element: str, buck_slip: dict
) -> str:
    environment = jinja2.Environment()

    task = "merge"
    if task is None:
        task = "merge"

    template = environment.from_string(buck_slip["prompt_templates"][task])
    my_prompt = template.render(
        first_element=first_element, second_element=second_element
    )

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


async def merge_elements(
    first_element: str, second_element: str, buck_slip: dict
) -> str:
    # ic(first_element)
    # ic(second_element)
    intermediate_merge_result = await get_completion_merge(
        first_element, second_element, buck_slip
    )
    # ic(intermediate_merge_result)
    intermediate_merge_result = str(intermediate_merge_result).strip()
    ic(len(intermediate_merge_result))

    return intermediate_merge_result


async def split_further(partial_results: list, my_pos: int, buck_slip: dict) -> list:
    ic("Split further.")
    ic(my_pos)
    ic(len(partial_results))

    my_len_list = [len(_) for _ in partial_results]
    ic(my_len_list)

    my_chunk = partial_results[my_pos]
    length_of_chunk_in_tokens = get_length_of_chunk_in_tokens(my_chunk, buck_slip)
    # ic(len(my_chunk))
    # ic(length_of_chunk_in_tokens)
    # my_custom_chunk_size = math.ceil((length_of_chunk_in_tokens / 2) * 1.10)
    # my_custom_chunk_overlap = math.ceil(my_custom_chunk_size * 0.1)
    my_custom_chunk_size = length_of_chunk_in_tokens
    my_custom_chunk_overlap = 0
    # my_custom_chunk_overlap = math.ceil(my_custom_chunk_size * 0.1)

    # ic(my_custom_chunk_size)
    # ic(my_custom_chunk_overlap)

    buck_slip["text_splitter"] = get_text_splitter(
        buck_slip, my_custom_chunk_size, my_custom_chunk_overlap
    )

    chunks = await get_async_chunking(my_chunk, buck_slip)
    ic(len(chunks))
    my_len_list = [len(_) for _ in chunks]
    ic(my_len_list)

    # first_half = chunks[0]
    # second_half = chunks[1]

    # partial_results[my_pos:my_pos + 1] = (first_half, second_half)
    # partial_results[my_pos:my_pos] = chunks
    # test_list = test_list[:pos] + insert_list + test_list[pos:]
    partial_results = partial_results[: my_pos] + chunks + partial_results[my_pos + 1:]
    ic(len(partial_results))

    my_len_list = [len(_) for _ in partial_results]
    ic(my_len_list)

    return partial_results


async def enter_recursion(
    my_chunk: str, recursion_depth: int, buck_slip: dict, task: str
) -> str:
    recursion_depth += 1
    ic(recursion_depth)

    # length_of_chunk_in_chars = get_length_of_chunk_in_chars(my_chunk)
    length_of_chunk_in_tokens = get_length_of_chunk_in_tokens(my_chunk, buck_slip)

    if length_of_chunk_in_tokens >= buck_slip["chunk_size"]:
        # We need to split

        # DRAFT EVEN SPLITTING BEGIN
        ic(length_of_chunk_in_tokens)
        my_divisor = math.ceil(length_of_chunk_in_tokens / buck_slip["chunk_size"])
        ic(my_divisor)
        my_divisor = power_log(my_divisor)
        ic(my_divisor)

        # if (my_divisor % 2) != 0:
        #     my_divisor += 1
        # ic(my_divisor)
        my_custom_chunk_size = math.ceil(length_of_chunk_in_tokens / my_divisor)
        ic(my_custom_chunk_size)
        my_custom_chunk_size = math.ceil(my_custom_chunk_size * 1.10)
        ic(my_custom_chunk_size)
        my_custom_chunk_overlap = math.ceil(my_custom_chunk_size * 0.1)
        # ic(my_custom_chunk_size)
        # ic(my_custom_chunk_overlap)

        buck_slip["text_splitter"] = get_text_splitter(
            buck_slip, my_custom_chunk_size, my_custom_chunk_overlap
        )

        chunks = await get_async_chunking(my_chunk, buck_slip)
        ic(len(chunks))
        for i, chunk in enumerate(chunks):
            ic(i)
            ic(len(chunk))
            ic(get_length_of_chunk_in_tokens(chunk, buck_slip))
        # DRAFT EVEN SPLITTING END

        partial_results = []

        partial_results = await asyncio.gather(
            *[
                enter_recursion(partial_chunk, recursion_depth, buck_slip, "summarize")
                for partial_chunk in chunks
            ]
        )

        # HERE WE NEED TO CHANGE BEGIN
        # Instead of concatenating all partial_results together
        # and then summarizing the result, we want to summarize them pair-
        # wise.

        # my_result_string = "\n".join(partial_results)
        # intermediate_result = await enter_recursion(
        #     my_result_string, recursion_depth, buck_slip, "merge"
        # )

        ic("Start merging.")

        # While we do have more than two chunks to merge
        while len(partial_results) > 1:
            # While we do not have even number of elements, we need to amend
            split_further_counter = 0
            while ((len(partial_results) % 2) != 0) and split_further_counter < 10:
                # Find longest element
                my_lei = find_longest_element_index(partial_results)

                # Split longest element further
                partial_results = await split_further(
                    partial_results, my_lei, buck_slip
                )
                ic("Splitting further done.")
                split_further_counter += 1
                # ic(len(partial_results))

            # ic(len(partial_results))
            # ic("Merging.")
            partial_results = await asyncio.gather(
                *[
                    merge_elements(first_element, second_element, buck_slip)
                    for first_element, second_element in grouped(partial_results, 2)
                ]
            )
        ic(len(partial_results))

        intermediate_result = partial_results[0]
        # HERE WE NEED TO CHANGE END
    else:
        # We can summarize
        ic("Summarize.")
        intermediate_result = await get_completion(my_chunk, buck_slip, "summarize")
        ic(len(str(intermediate_result)))

    my_result = str(intermediate_result).strip()

    return my_result
