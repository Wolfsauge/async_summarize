<img align="right" src="images/async_summarize.png">

# async_summarize
An asynchronous summarization script.

This script summarizes the input file using a large language model API. If the input exceeds the context of the LLM, it will be split using the [LangChain](https://www.langchain.com) [RecursiveTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter) class.

The LangChain text splitter class uses a Huggingface transformers tokenizer.

# Note

The documentation needs rework ...
