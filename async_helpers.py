import sys
import asyncio
import math
from tqdm.asyncio import tqdm  # type: ignore

from icecream import ic  # type: ignore
from sync_helpers import (
    get_length_of_chunk_in_tokens,
    get_text_splitter,
    grouped,
    power_log,
    find_longest_element_index,
)


async def get_completion(buck_slip: dict, task: str, **kwargs) -> str:
    template = buck_slip["jinja2_env"].from_string(buck_slip["prompt_templates"][task])

    if task == "summarize":
        chunk = kwargs["chunk"]
        if isinstance(chunk, str):
            my_prompt = template.render(prompt=chunk)
        else:
            ic(f"ERROR: function call error, task: {task}, kwargs={kwargs}.")
            sys.exit(1)
    elif task == "merge":
        first_element = kwargs["first_element"]
        second_element = kwargs["second_element"]
        if isinstance(first_element, str) and isinstance(second_element, str):
            my_prompt = template.render(
                first_element=first_element, second_element=second_element
            )
        else:
            ic(f"ERROR: function call error, task: {task}, kwargs={kwargs}.")
            sys.exit(1)
    else:
        ic(f"ERROR: function call error, task: {task}, kwargs={kwargs}.")
        sys.exit(1)

    bad_counter = 0
    attempt_counter = 0

    while attempt_counter <= buck_slip["max_completion_retries"]:
        my_temperature = buck_slip["temperature"] + attempt_counter * 0.1
        completion = await buck_slip["api_client"].completions.create(
            model=buck_slip["model_local_identifier"],
            prompt=my_prompt,
            max_tokens=buck_slip["max_tokens"],
            temperature=my_temperature,
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


async def merge_elements(elements, buck_slip: dict, pindex: int) -> tuple[str, int]:
    first_element, second_element = elements

    intermediate_merge_result = await get_completion(
        buck_slip, "merge", first_element=first_element, second_element=second_element
    )
    intermediate_merge_result = str(intermediate_merge_result).strip()

    return intermediate_merge_result, pindex


async def summarize_element(chunk, buck_slip: dict, pindex: int) -> tuple[str, int]:
    intermediate_merge_result = await get_completion(
        buck_slip, "summarize", chunk=chunk
    )
    intermediate_merge_result = str(intermediate_merge_result).strip()

    return intermediate_merge_result, pindex


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
    partial_results = partial_results[:my_pos] + chunks + partial_results[my_pos + 1:]
    ic(len(partial_results))

    my_len_list = [len(_) for _ in partial_results]
    ic(my_len_list)

    return partial_results


async def do_the_work(
    chunk: str, buck_slip: dict, mode: str, pindex: int
) -> tuple[str, int]:
    # recursion_depth += 1
    # ic(recursion_depth)

    if mode == "hm":
        # length_of_chunk_in_chars = get_length_of_chunk_in_chars(chunk)
        length_of_chunk_in_tokens = get_length_of_chunk_in_tokens(chunk, buck_slip)

        if length_of_chunk_in_tokens >= buck_slip["chunk_size"]:
            # We need to split

            # DRAFT EVEN SPLITTING BEGIN
            my_divisor = math.ceil(length_of_chunk_in_tokens / buck_slip["chunk_size"])
            my_divisor = power_log(my_divisor)
            my_custom_chunk_size = math.ceil(length_of_chunk_in_tokens / my_divisor)
            my_custom_chunk_size = math.ceil(my_custom_chunk_size * 1.10)
            my_custom_chunk_overlap = math.ceil(my_custom_chunk_size * 0.1)

            buck_slip["text_splitter"] = get_text_splitter(
                buck_slip, my_custom_chunk_size, my_custom_chunk_overlap
            )

            tqdm.write("Start chunking.")
            chunks = await get_async_chunking(chunk, buck_slip)
            # ic(len(chunks))
            # for i, chunk in enumerate(chunks):
            #     ic(i)
            #     ic(len(chunk))
            #     ic(get_length_of_chunk_in_tokens(chunk, buck_slip))
            # DRAFT EVEN SPLITTING END

            # ic("Before summarize:")
            # for i, partial_result in enumerate(chunks):
            #     ic(i)
            #     ic(partial_result)

            tqdm.write("Start summarizing.")
            partial_results_tasks = [
                summarize_element(partial_chunk, buck_slip, pindex)
                for pindex, partial_chunk in enumerate(chunks)
            ]
            partial_results = [None] * len(chunks)

            for my_func in tqdm(
                asyncio.as_completed(partial_results_tasks),
                total=len(partial_results_tasks),
            ):
                result, returned_index = await my_func
                tqdm.write(
                    f"Received summarize result {returned_index}, length {len(result)} characters."
                )
                partial_results[returned_index] = result

            # ic("After summarizing:")
            # for i, partial_result in enumerate(partial_results):
            #     ic(i)
            #     ic(partial_result)

            tqdm.write("Start merging.")
            my_round = 1

            # While we do have more than two chunks to merge
            while len(partial_results) > 1:
                # ic(my_round)
                tqdm.write(f"Merging round #{my_round}.")
                # While we do not have even number of elements, we need to amend
                split_further_counter = 0

                # This must be improved:
                # * splitting into 2 not possible?
                # * instead of splitting, it's possible to merge two small adjacent chunks
                # * retrying also with increase in temperature?
                while ((len(partial_results) % 2) != 0) and split_further_counter < 10:
                    # Find longest element
                    my_lei = find_longest_element_index(partial_results)

                    # Split longest element further
                    partial_results = await split_further(
                        partial_results, my_lei, buck_slip
                    )
                    ic("Splitting further done.")
                    split_further_counter += 1

                partial_results_tasks = [
                    merge_elements(elements, buck_slip, pindex)
                    for pindex, elements in enumerate(grouped(partial_results, 2))
                ]
                partial_results = [None] * math.ceil(len(partial_results) / 2)

                for my_func in tqdm(
                    asyncio.as_completed(partial_results_tasks),
                    total=len(partial_results_tasks),
                ):
                    result, returned_index = await my_func
                    tqdm.write(
                        f"Received merge result {returned_index}, length {len(result)} characters."
                    )
                    partial_results[returned_index] = result

                # ic("After merging:")
                # for i, partial_result in enumerate(partial_results):
                #     ic(i)
                #     ic(partial_result)

                my_round += 1

            intermediate_result = partial_results[0]
        else:
            intermediate_result = await summarize_element(chunk, buck_slip, pindex)
            # # We can summarize
            # # ic("Summarize.")
            # intermediate_result = await get_completion(
            #     buck_slip, "summarize", chunk=chunk
            # )
            # ic(len(str(intermediate_result)))

        my_result = str(intermediate_result).strip()

    return my_result, pindex
