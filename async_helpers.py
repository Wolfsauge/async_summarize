import sys
import asyncio
import math
from tqdm.asyncio import tqdm  # type: ignore

from icecream import ic  # type: ignore
from sync_helpers import (
    get_length_of_chunk_in_tokens,
    get_text_splitter,
    grouped,
    find_chunk_pair_with_minimal_size,
    find_longest_element_index,
    calc_custom_chunking_parameters,
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


async def do_chunking_step(my_chunk: str, buck_slip: dict) -> list:
    lock = buck_slip["lock"]
    tqdm.write(f"Acquired {lock}.")
    async with lock:
        chunks = buck_slip["text_splitter"].split_text(my_chunk)
    tqdm.write(f"Released {lock}.")

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

    lock = buck_slip["lock"]
    tqdm.write(f"Acquired {lock}.")
    async with lock:
        length_of_chunk_in_tokens = get_length_of_chunk_in_tokens(my_chunk, buck_slip)
    tqdm.write(f"Released {lock}.")

    my_custom_chunk_size = length_of_chunk_in_tokens
    my_custom_chunk_overlap = 0
    buck_slip["text_splitter"] = get_text_splitter(
        buck_slip, my_custom_chunk_size, my_custom_chunk_overlap
    )

    chunks = await do_chunking_step(my_chunk, buck_slip)
    ic(len(chunks))
    my_len_list = [len(_) for _ in chunks]
    ic(my_len_list)

    partial_results = partial_results[:my_pos] + chunks + partial_results[my_pos + 1 :]
    ic(len(partial_results))

    my_len_list = [len(_) for _ in partial_results]
    ic(my_len_list)

    return partial_results


async def do_summarizing_step(chunks: list, buck_slip: dict) -> list:
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
    return partial_results


async def do_merging_step(partial_results: list, buck_slip: dict) -> str:
    tqdm.write("Start merging.")
    merging_round = 0

    # While we do have more than two chunks to merge
    while len(partial_results) > 1:
        merging_round += 1
        tqdm.write(f"Merging round #{merging_round}.")

        # While we do not have even number of elements, we need to amend
        amend_counter = 0

        # Make sure we have an even number of chunks before we start merging
        while ((len(partial_results) % 2) != 0) and amend_counter < 10:
            # Find pair of chunks, which, concatenated together will yield the
            # smallest sum
            my_pos1, my_pos2 = find_chunk_pair_with_minimal_size(partial_results)

            lock = buck_slip["lock"]
            tqdm.write(f"Acquired {lock}.")
            async with lock:
                len_t_pos1 = get_length_of_chunk_in_tokens(
                    partial_results[my_pos1], buck_slip
                )
                len_t_pos2 = get_length_of_chunk_in_tokens(
                    partial_results[my_pos2], buck_slip
                )
            tqdm.write(f"Released {lock}.")

            # First we try to merge the smallest chunk with its adjacent chunk
            if (len_t_pos1 + len_t_pos2) <= buck_slip["chunk_size"]:
                tqdm.write(
                    f"Merging chunk #{my_pos1} ({len(partial_results[my_pos1])}/{len_t_pos1} chars/tokens) and #{my_pos2} ({len(partial_results[my_pos2])}/{len_t_pos2} chars/tokens)."
                )

                new_chunk, _ = await merge_elements(
                    (partial_results[my_pos1], partial_results[my_pos2]), buck_slip, 0
                )
                partial_results[my_pos1] = new_chunk
                del partial_results[my_pos2]
                amend_counter += 1
            else:
                # We can't reduce the number of chunks, so we try to solve the problem
                # by splitting the largest chunk again

                # Find longest element
                my_lei = find_longest_element_index(partial_results)

                tqdm.write(f"Splitting the longest element #{my_lei} further.")

                # Split longest element further
                partial_results = await split_further(
                    partial_results, my_lei, buck_slip
                )

                amend_counter += 1

        # We have an even number of chunks to merge now

        # Iterate pair-wise through the set of chunks and merge them asynchronously
        partial_results_tasks = [
            merge_elements(elements, buck_slip, pindex)
            for pindex, elements in enumerate(grouped(partial_results, 2))
        ]

        # Prepare synchronized result set
        partial_results = [None] * math.ceil(len(partial_results) / 2)

        # Await asynchronous merge tasks completion
        for my_func in tqdm(
            asyncio.as_completed(partial_results_tasks),
            total=len(partial_results_tasks),
        ):
            result, returned_index = await my_func
            tqdm.write(
                f"Received merge result {returned_index}, length {len(result)} characters."
            )
            # Once a task has completed, insert its result into the correct
            # position in the synchronized result set
            partial_results[returned_index] = result

        # We are done merging this round
    return partial_results[0]


async def do_the_work(chunk: str, buck_slip: dict, mode: str) -> str:
    if mode == "hm":
        tqdm.write("Start summarizing with the method of hierarchical merging.")

        lock = buck_slip["lock"]
        tqdm.write(f"Acquired {lock}.")
        async with lock:
            length_of_chunk_in_tokens = get_length_of_chunk_in_tokens(chunk, buck_slip)
        tqdm.write(f"Released {lock}.")

        if length_of_chunk_in_tokens >= buck_slip["chunk_size"]:
            # We need to split the input into chunks
            custom_chunk_size, custom_chunk_overlap = calc_custom_chunking_parameters(
                length_of_chunk_in_tokens, buck_slip
            )
            buck_slip["text_splitter"] = get_text_splitter(
                buck_slip, custom_chunk_size, custom_chunk_overlap
            )
            chunks = await do_chunking_step(chunk, buck_slip)
            partial_results = await do_summarizing_step(chunks, buck_slip)
            result = await do_merging_step(partial_results, buck_slip)
        else:
            # We can summarize the input directly
            result, _ = await summarize_element(chunk, buck_slip, 0)

    return result
