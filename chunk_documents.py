import argparse
from time import time
from typing import List
from math import ceil
import requests
import json
import sys
import textwrap
import tiktoken


# The endpoint we send our requests to (the server is hosted locally with LM studio)
llm_endpoint = "http://127.0.0.1:1234/v1/chat/completions"

# The system prompt used to guide the LLM into creating good chunk summaries
system_prompt = """
You will be provided with a chunk from a larger document. Summarize its purpose and topics in the following format:

This chunk's purpose is [purpose of document], potentially covering [general topic]. It addresses the following key points: [Main purpose or subject]. [Briefly mention each key topic discussed, providing a concise context without unnecessary elaboration].

Make sure to respond strictly in paragraph format. Do not use bullet points, lists, or any other formatting. Your response should be continuous and clear. 

Avoid including pleasantries or additional text outside of the summary.

Here is the content:
"""


def chunk_document(
    document_content: str, chunk_size_tokens: int = 1000, tokenizer: str = "gpt-4o"
) -> List["str"]:
    """Return an array of chunks of a document, where each chunk is `chunk_size` tokens in length."""
    enc = tiktoken.encoding_for_model(tokenizer)
    # Create an array of tokens for the content
    tokenized_document = enc.encode(document_content)
    # Total number of tokens present in the content
    num_tokens_in_document = len(tokenized_document)

    chunks = []
    for i in range(0, num_tokens_in_document, chunk_size_tokens):
        # Decode the tokens for a particular chunk (i.e. turn it back to the original content)
        chunks.append(enc.decode(tokenized_document[i : i + chunk_size_tokens]))

    return chunks


def write_multiple(output_file: str, chunks: List["str"]):
    """Write multiple document chunks or summaries to a single file, separated by newlines"""
    with open(output_file, "w+", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(f"{chunk}\n")


def request_chunk_summary(
    chunk_content: str, model: str, temperature: float = 0.25, max_new_tokens: int = -1
) -> str:
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
                "temperature": temperature,
                "max_tokens": max_new_tokens,
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
    tokenizer_used: str = "gpt-4o",
    max_line_width: int = 100,
):
    """Write each chunk and its summary version to a text file, nicely formatted."""
    assert len(chunks) == len(
        summaries
    ), "len(chunks) and len(summaries) must be equal. (Each chunk needs a summary)"

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk, summary in zip(chunks, summaries):
            wrapped_chunk = textwrap.fill(chunk, max_line_width)
            wrapped_summary = textwrap.fill(summary, max_line_width)
            # Get lengths in chars
            char_length_chunk = len(chunk)
            char_length_summary = len(summary)
            # Get lengths in tokens
            token_length_chunk = get_token_length(chunk, tokenizer_used)
            token_length_summary = get_token_length(summary, tokenizer_used)

            f.write("==========")
            f.write("\n\nChunk Content: \n\n")
            f.write(wrapped_chunk)
            f.write("\n----------")
            f.write("\n\nChunk Summary: \n\n")
            f.write(wrapped_summary)
            f.write("\n----------")
            f.write(f"\nOriginal length (chars): {char_length_chunk}")
            f.write(f"\nSummary length (chars): {char_length_summary}")
            f.write(
                f"\nCompression ratio (chars): {char_length_chunk/char_length_summary:.4f}x\n"
            )
            f.write(f"\nOriginal length (tokens): {token_length_chunk}")
            f.write(f"\nSummary length (tokens): {token_length_summary}")
            f.write(
                f"\nCompression ratio (tokens): {token_length_chunk/token_length_summary:.4f}x\n"
            )


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


def validate_arguments(args, parser):
    """Validates command-line arguments."""
    if args.chunk_size <= 0:
        parser.error("Chunk size must be a positive integer")

    if not (0 <= args.temperature <= 1):
        parser.error("Temperature must be in the interval [0,1]")


def print_configuration(args):
    """Prints the current configuration settings to the console."""
    print("\nCurrent Configuration:")
    print("=" * 30)
    print(f"Model             : {args.model}")
    print(f"Document Path     : {args.document_path}")
    print(f"Chunk Size        : {args.chunk_size} tokens")
    print(f"Tokenizer         : {args.tokenizer}")
    print(f"Temperature       : {args.temperature}")
    print(
        f"Max New Tokens    : {args.max_new_tokens if args.max_new_tokens != -1 else 'No limit'}"
    )
    print("=" * 30 + "\n")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Split a document into chunks, then create summaries for each chunk."
    )
    parser.add_argument(
        "document_path", type=str, help="Path to the document to summarize"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Size of each chunk in tokens (default: 100)",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt-4o",
        help="Tokenizer used to judge chunk size in tokens (default: gpt-4o)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.2-3b-instruct",
        help="Name of the model used to generate summaries, this is included in our requests to the LM Studio server. (default: llama-3.2-3b-instruct)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.25,
        help="Model temperature (a value in the interval [0,1], lower = more precision, higher = more creativity, default: 0.25)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=-1,
        help="Maximum number of tokens generated for each summary. -1 indicates no limit (default: -1)",
    )
    args = parser.parse_args()

    # Validate arguments
    validate_arguments(args, parser)

    # Print the current configuration
    print_configuration(args)

    document_path = args.document_path
    # Read the contents of the document we want to chunk & summarize into a string
    input_document = get_file_content(document_path)
    # Split document into multiple chunks
    chunks = chunk_document(
        input_document, chunk_size_tokens=args.chunk_size, tokenizer=args.tokenizer
    )

    # Request a summary for each chunk, while printing out the progress
    summaries = []
    start_summarizing = time()
    for i in range(len(chunks)):
        _start = time()
        summaries.append(
            request_chunk_summary(
                chunks[i],
                args.model,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        )
        _end = time()
        print(
            f"Progress: {i + 1}/{len(chunks)} ({(i + 1) / len(chunks) * 100:.2f}%) - "
            f"Last summary duration: {_end - _start:.2f} seconds",
            end="\r",
        )
    end_summarizing = time()
    # Clear the last progress line
    print(" " * 80, end="\r")
    # Print completion time
    print(f"Finished summaries in {end_summarizing - start_summarizing:.2f} seconds")

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


if __name__ == "__main__":
    main()
