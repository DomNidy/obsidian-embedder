# obsidian-embedder

_(This README is a W.I.P)_

This repository contains scripts used to make managing your notes easier. Currently, there are two distinct features: **document embedding & similarity binning** and **documenting chunking with LLM generated chunk summaries**.

To install dependencies easily, you can create a virtual environment, then install everything in `requirements.txt`:

```sh
python -m venv venv # create the virtual environment
source venv/bin/activate # for windows, run: 'venv\Scripts\activate'
pip install -r requirements.txt # install the dependencies
```

**Note:** It seems that the `torch` version specified in `requirements.txt` was not compiled with CUDA support. As such, similarity binning may be a bit slow.

## Document embedding

Generate document embeddings for your Obsidian vault and organize them into groups (bins) based on semantic similarity. This utility is designed to help manage large Obsidian vaults with many documents, making it easier to identify redundant or poorly categorized documents.

The similarity search/document binning is powered by a Sentence-BERT model.

### Step 1: Pre-processing your Obsidian vault

To prepare your documents to be embedded, you can run the `preprocess.py` script and point it at the root directory of your Obsidian vault. The output of this script will be a single text file where each line contains the contents of a single document chunk.

_**Note:** Currently, creating multiple chunks from a single document is not supported. Each chunk in the output text file will be the contents of an entire document._

**Example:**

```sh
python preprocess.py C:/vault -mlr 0.30 -mdl 150 -bw tags review srdue srease excalidraw --include-document-title -o preprocessed_vault.txt
```

### Step 2: Embed and bin documents

_TODO_

**Example:**

```sh
python bin_documents.py preprocessed_vault.txt bins.txt
```

## Document chunking and summarizing

This allows you to split a single document into many chunks and then create a summary for each of them individually.

**Note:** To run this locally, you will need to have a [LM Studio](https://lmstudio.ai/) server hosting an LLM. This means you probably will need a GPU to run this efficiently. Personally, I have an RTX 2070 Super (8GB VRAM) and am able to run the instruction tuned Llama 3.2 1b and 3b with 8 bit quantization just fine.

**Example:**

```sh
python tools/document_summarizer/main.py some_document.txt --chunk_size 500 --temperature 0.15 --model llama-3.2-1b-instruct --output_dir ./output
```

This will produce two output files:

- **`./output/some_document_chunk_summary_comparison.txt`:** This file contains each original chunk and its associated summary, along with the lengths of the two. Primarily used to inspect how good the LLM is doing at summarizing.

- **`./output/some_document_summaries_only.txt`:** This file contains each chunk summary separated by newlines.

### Pipeline for chunking, summarizing, and preparing for RAG systems

1. **Document Source:** Create (or find an existing) a directory of documents (.txt, .md) files that you would like to perform RAG on. Example: `/my_documents`

2. **Chunking & Summarizing:** Split documents into chunks and create summaries of each chunk. This extracts the key concepts, compresses the token count, and generally speaking, makes the data more suitable for RAG. This step can be performed with the document summarizer tool, example: `python tools/document_summarizer/main.py --document_dir "/my_documents" --output_dir "./summaries"`

3. **Combining Summaries:** The output of the summarizer tool is a large list of .txt files, one for each document input to the tool. To combine these summaries into a single structured XML file, you can run the `tools/document_summarizer/combine_summaries.sh` script, example: `tools/document_summarizer/combine_summaries.sh ./summaries combined_summaries.xml`

4. **Setup Vector DB:** _TODO: Finish this_
