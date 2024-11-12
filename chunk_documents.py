import os
from typing import List
from math import ceil
import requests
import json
import sys
import textwrap
import tiktoken


# The endpoint we send our requests to (the server is hosted locally with LM studio)
llm_endpoint = "http://127.0.0.1:1234/v1/chat/completions"

# The model on the LM studio server we send our summary requests to
model = "llama-3.2-3b-instruct"  # llama-3.2-1b-instruct

# The system prompt used to guide the LLM into creating good chunk summaries
system_prompt = """
You will be provided with a chunk from a larger document. Summarize its purpose and topics in the following format:

The purpose of this chunk appears to be [Purpose of document], potentially covering [general topic]. It addresses the following key points: [Main purpose or subject]. [Briefly mention each key topic discussed, providing a concise context without unnecessary elaboration].

Make sure to respond strictly in paragraph format. Do not use bullet points, lists, or any other formatting. Your response should be continuous and clear. 

Avoid including pleasantries or additional text outside of the summary.

Here is the content:
"""


def chunk_document(document_content: str, chunk_size: int = 2000) -> List["str"]:
    """Return an array of chunks of a document, where each chunk is `chunk_size` in length."""
    document_length = len(document_content)

    chunks = []
    for i in range(0, document_length, chunk_size):
        chunks.append(document_content[i : i + chunk_size])

    return chunks


def write_multiple(output_file: str, chunks: List["str"]):
    """Write multiple document chunks or summaries to a single file, separated by newlines"""
    with open(output_file, "w+") as f:
        for chunk in chunks:
            f.write(f"{chunk}\n")


def request_chunk_summary(chunk_content: str, model: str) -> str:
    """Request a summary for a chunk."""
    response = requests.post(
        "http://127.0.0.1:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk_content},
                ],
                "temperature": 0.2,
                "max_tokens": -1,
                "stream": "false",
            }
        ),
    ).json()

    summary = response["choices"][0]["message"]["content"]

    return summary


def write_chunk_summary_comparison(
    output_file: str,
    chunks: List["str"],
    summaries: List["str"],
    max_line_width: int = 100,
):
    """Write each chunk and its summary version to a text file, nicely formatted."""
    assert len(chunks) == len(
        summaries
    ), "len(chunks) and len(summaries) must be equal. (Each chunk needs a summary)"

    with open(output_file, "w") as f:
        for chunk, summary in zip(chunks, summaries):
            wrapped_chunk = textwrap.fill(chunk, max_line_width)
            wrapped_summary = textwrap.fill(summary, max_line_width)
            f.write("==========")
            f.write("\n\nChunk Content: \n\n")
            f.write(wrapped_chunk)
            f.write("\n----------")
            f.write("\n\nChunk Summary: \n\n")
            f.write(wrapped_summary)
            f.write("\n----------")
            f.write(f"\nOriginal length: {len(chunk)}\n")
            f.write(f"\nSummary length: {len(summary)}\n\n")


def get_token_length(chunk: str, tokenizer="gpt-4o") -> int:
    """Return the length of a chunk in tokens."""
    enc = tiktoken.encoding_for_model(tokenizer)
    return len(enc.encode(chunk))


def get_total_character_length(chunks: List["str"]) -> int:
    """Return the total length of all chunks (in characters)"""
    return sum([len(chunk) for chunk in chunks])


def get_total_token_length(chunks: List["str"], tokenizer="gpt-4o") -> int:
    """Return the total length of all chunks (in tokens)"""

    total_len = 0
    for i in range(len(chunks)):
        total_len += get_token_length(chunks[i], tokenizer)

    return total_len


def get_file_content(file_path: str) -> str:
    """Reads the contents of the specified file and returns it as a string."""
    with open(file_path, "r") as f:
        content = f.readlines()
        return " ".join(content)


if __name__ == "__main__":
    document_path = sys.argv[1] if len(sys.argv) > 1 else exit("No document provided")
    # Read the contents of the document we want to chunk & summarize into a string
    input_document = get_file_content(document_path)
    os.path
    # Split document into multiple chunks
    chunks = chunk_document(input_document, chunk_size=2000)
    # Request a summary for each chunk
    summaries = [request_chunk_summary(chunk, model) for chunk in chunks]

    # Print out compression ratio
    print(f"Total chunk token count: {get_total_token_length(chunks)}")
    print(f"Total summary token count: {get_total_token_length(summaries)}")
    print(
        f"Token compression ratio: {get_total_token_length(chunks)/get_total_token_length(summaries):.4f}x\n"
    )
    print(f"Total chunk character count: {get_total_character_length(chunks)}")
    print(f"Total summary character count: {get_total_character_length(summaries)}")
    print(
        f"Character compression ratio: {get_total_character_length(chunks)/get_total_character_length(summaries):.4f}x"
    )

    # Save the created summaries to a single text file
    write_multiple(f"{document_path}_summaries_only.txt", summaries)
    # Create and write a chunk-summary pair comparison to a single text file
    write_chunk_summary_comparison(
        f"{document_path}_chunk_summary_comparison.txt", chunks, summaries
    )
