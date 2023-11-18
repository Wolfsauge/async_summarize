# async_summarize
An asynchronous summarization script.

```shell
$ ./main -f file.txt
```
* this script will try to summarize file.txt using the parameters defined in config.yaml
* if the input is too long, it will be split recursively, with overlap
* `httpx_max_connections` and `httpx_max_keepalive_connections` allow to control the number of simultaneous HTTP connections towards your API

```yaml
---
httpx_max_keepalive_connections: 1
httpx_max_connections: 1
use_fast: true
use_batched_tokenization: true
chunk_size: 3000
chunk_overlap: 512
# api_url: http://localhost:4999/v1
api_url: http://ultraforce:5000/v1
# api_url: http://localhost:8000/v1
api_key: empty
model_identifier: jondurbin/airoboros-m-7b-3.1.2
max_tokens: 1000
temperature: 0.2
```

# Features

## Tokenizer Parallelism
* The script enables Rust-based tokenizer parallelism by using Huggingface transformers fast tokenizers.
* To do so, the script is written with respect to the needs of tokenizer parallelism and is meant to be safe to run with `TOKENIZERS_PARALLELISM` set to `true`.

### Example
```shell
$ export TOKENIZERS_PARALLELISM=true
$ ./main -f file.txt
```

### References
* GitHub transformers Issue [Tokenizers throwing warning "The current process just got forked, Disabling parallelism to avoid deadlocks.. To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)" #5486](https://github.com/huggingface/transformers/issues/5486#issuecomment-654232343)
* Stack Overflow, [How to disable TOKENIZERS_PARALLELISM=(true | false) warning?](https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996), written Jul 10, 2022

## Batched Tokenization
* The script enables batched tokenization using Huggingface transformers fast tokenizers.
* The tokenization computations are cpu-bound and very expensive. this feature uses batching to speed up the tokenization process, which is required for the chunk length computations inside the langchain text splitter class.

### Help to get the PR running in a Python .venv quickly

* using GitHub client and poetry to build and install the langchain PR #5583 directly into your async_summarize .venv

```shell
$ gh repo clone langchain-ai/langchain
$ cd langchain
$ gh pr checkout 5583
$ cd langchain/libs/langchain
$ ln -sf <the .venv of your async_summarize> .venv
$ poetry build
$ poetry install
```

### References
* Huggingface NLP Course, [Chapter: Fast tokenizers' special powers
](https://huggingface.co/learn/nlp-course/chapter6/3#fast-tokenizers-special-powers)
* GitHub Pull Request langchain-ai [Batched length functions for text splitters #5583
](https://github.com/langchain-ai/langchain/pull/5583)
* GitHub Issue langchain-ai [Split by Tokens instead of characters: RecursiveCharacterTextSplitter #4678
](https://github.com/langchain-ai/langchain/issues/4678)

# Example Run 1 without Tensor Parallelism
* In this example Oobabooga text-generation webui was used, which does not implement tensor parallelism, so httpx connections is set to 1, in order to sequence the LLM requests.
* Tokenization/client was a macBook Pro 15-inch 2018, Sonoma 14.1.1, Intel® Core™ i9-8950HK Processor, 6 core, 32 GB RAM.
* Inference/server was a oobabooga (commit `f889302d2427c03a3c15925dd097c6213845660c`) on Windows 10, RTX 3090, Cuda 12.1.1, transformers 4.35.2.
* Sample Text [Frankenstein; Or, The Modern Prometheus by Mary Wollstonecraft Shelley](https://www.gutenberg.org/ebooks/84)

```text
$ poetry run ./main.py -c config-jondurbin_airoboros-m-7b-3.1.2.yaml -p prompt-airoboros-default-summarize-130.yaml -f pg84.txt
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
ic| buck_slip_filename: 'config-jondurbin_airoboros-m-7b-3.1.2.yaml'
Reading... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 351/351 bytes 0:00:00
ic| buck_slip: {'api_key': 'empty',
                'api_url': 'http://ultraforce:5000/v1',
                'chunk_overlap': 512,
                'chunk_size': 3000,
                'httpx_max_connections': 1,
                'httpx_max_keepalive_connections': 1,
                'max_tokens': 1000,
                'model_identifier': 'jondurbin/airoboros-m-7b-3.1.2',
                'temperature': 0.2,
                'use_batched_tokenization': True,
                'use_fast': True}
ic| prompt_template_filename: 'prompt-airoboros-default-summarize-130.yaml'
Reading... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 137/137 bytes 0:00:00
ic| prompt_template: ('BEGININPUT
                     '
                      '{{ prompt }}
                     '
                      'ENDINPUT
                     '
                      'BEGININSTRUCTION
                     '
                      'Summarize the input in about 130 words.
                     '
                      'ENDINSTRUCTION')
ic| type(tokenizer): <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>
ic| tokenizer.is_fast: True
ic| type(encoding): <class 'transformers.tokenization_utils_base.BatchEncoding'>
ic| encoding.is_fast: True
ic| type(text_splitter): <class 'langchain.text_splitter.RecursiveCharacterTextSplitter'>
ic| batched_tokenization: True
ic| type(api_client): <class 'openai.AsyncOpenAI'>
ic| input_filename: 'pg84.txt'
Reading... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 449.0/449.0 kB 0:00:00
ic| len(sample_text): 438838
ic| output_filename: 'pg84-analysis-jondurbin_airoboros-m-7b-3.1.2-0000.json'
ic| 'Init complete.'
ic| recursion_depth: 1
ic| len(chunks): 47
ic| recursion_depth: 2
ic| recursion_depth: 2
ic| recursion_depth: 2
[...]
```

* The resulting JSON looks like this:

```shell
$ jq . < pg84-analysis-jondurbin_airoboros-m-7b-3.1.2.json
{
  "summary": "Robert Walton, a man with extraordinary imagination embarks on a journey to the North Pole. Edward, a young man of remarkable scientific mind sails with Walton. Along the journey, they encounter harsh weather conditions the include extreme cold which causes the ship's sails to freeze. In desolation, they hear a man calling out for help and they meet a monster that is actually Frankenstein's creature. The creature tells his story, which includes his inability to find companionship and loneliness, which eventually drove him to kill. When Walton reaches the Pole he sends his remaining letters to his sister.",
  "duration": "448.05 seconds",
  "model_identifier": "jondurbin/airoboros-m-7b-3.1.2",
  "chunk_size": 3000,
  "chunk_overlap": 512,
  "max_tokens": 1000,
  "api_url": "http://ultraforce:5000/v1",
  "length_of_sample_text_in_characters": 438838
}
```

## Summary Result

A more readable version of this output is this:

> Robert Walton, a man with extraordinary imagination embarks on a journey to the North Pole. Edward, a young man of remarkable scientific mind sails with Walton. Along the journey, they encounter harsh weather conditions the include extreme cold which causes the ship's sails to freeze. In desolation, they hear a man calling out for help and they meet a monster that is actually Frankenstein's creature. The creature tells his story, which includes his inability to find companionship and loneliness, which eventually drove him to kill. When Walton reaches the Pole he sends his remaining letters to his sister.


# Example Run 2 with Tensor Parallelism

tbd.

* same model as above
* bigger scale:
* bigger model
    * airoboros 3.1.2 70b
* this requires 2x A100 (in other words >= 80 GB VRAM)
* vllm OpenAI compatible API endpoint
