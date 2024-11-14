import argparse
import os
import requests
import json
import textwrap
import tiktoken
from time import time
from io import TextIOWrapper
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# The endpoint we send our requests to (the server is hosted locally with LM studio)
llm_endpoint = "http://127.0.0.1:1234/v1/chat/completions"

# The system prompt used to guide the LLM into creating good chunk summaries
system_prompt = """
You will be provided with a chunk from a larger document, typically drawn from a user's personal notes. Summarize the chunk's purpose and topics, ensuring that your summary is significantly shorter than the original text and does not restate information multiple times. Use the following format:

"The purpose of this chunk appears to be [purpose of document], potentially touching on [general topic]. Key points include: [Main purpose or subject]. [Briefly list each significant topic with concise context, without additional detail]."

Respond in a continuous, clear paragraph without bullet points or lists.

Guidelines:

1. If the chunk contains irrelevant symbols, random characters, multiple URLs, or nonsensical phrases that do not contribute to meaningful content, respond with: "Chunk contains no understandable content."

2. If URLs appear central to the content, mention them briefly, such as, "This document includes URLs, likely relevant to [related topic if determinable]."

3. When encountering data formats like JSON or code snippets, summarize the general structure and its likely use, such as, "This JSON format may store user information," or "This code appears to perform [general function]."

4. Avoid providing any answers if the content includes questions; only summarize the information presented.

5. Conclude by relating briefly how the content might connect to the user's potential interests if evident.

Here is the content:
"""


def get_token_length(content: str, tokenizer="gpt-4o") -> int:
    """Return the length of text content in tokens."""
    enc = tiktoken.encoding_for_model(tokenizer)
    return len(enc.encode(content))


@dataclass
class Timer:
    task_label: str
    after_msg: Optional[str] = None
    before_msg: Optional[str] = None
    print_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __enter__(self):
        self.started_at_seconds = time()
        if self.before_msg:
            self._print(f"[{self.task_label}]: {self.before_msg}")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.started_at_seconds is None:
            raise RuntimeError("Timer was not started. __enter__ was not called.")

        elapsed_time = time() - self.started_at_seconds
        finishing_msg = (
            f"{self.after_msg} (duration: {elapsed_time:.2f} seconds)"
            if self.after_msg
            else f"Completed task (duration: {elapsed_time:.2f} seconds)"
        )

        self._print(f"[{self.task_label}]: {finishing_msg}")

    def _print(self, message: str):
        print(message, **self.print_kwargs)


class DocumentChunk:
    """Object containing metadata about a chunk of text taken out of a larger document."""

    def __init__(self, content: str, offset_begin: int, offset_end: int) -> None:

        assert (
            offset_begin <= offset_end
        ), f"`offset_begin` ({offset_begin}) must be less than or equal to `offset_end` ({offset_end})"

        self.content = content
        self.offset_begin = offset_begin
        self.offset_end = offset_end

    def __str__(self):
        return str(self.content)

    def __len__(self):
        return len(self.content)


class DocumentChunkSummary:
    """Object containing summarized content and metadata about a summary created from a `DocumentChunk`."""

    def __init__(self, content: str, original: "DocumentChunk") -> None:
        self.original = original
        self.content = content

    def __str__(self):
        return str(self.content)

    def __len__(self):
        return len(self.content)


def request_chunk_summary(
    chunk: DocumentChunk,
    model: str,
    temperature: float = 0.25,
    max_new_tokens: int = -1,
) -> "DocumentChunkSummary":
    """Request a summary for a `DocumentChunk`"""
    response = requests.post(
        "http://127.0.0.1:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk.content},
                ],
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "stream": "false",
            }
        ),
    ).json()

    return DocumentChunkSummary(
        content=response["choices"][0]["message"]["content"], original=chunk
    )


def chunk_content(
    content: str, chunk_size_tokens: int = 1000, tokenizer: str = "gpt-4o"
) -> List["DocumentChunk"]:
    """
    Return a `DocumentChunk` list from text content, where each chunk's content is `chunk_size` tokens in length.
    Each `DocumentChunk` will record it's byte offset (relative to the start of the provided `content`)
    """
    enc = tiktoken.encoding_for_model(tokenizer)
    # Create an array of tokens for the content
    tokenized_content = enc.encode(content)
    # Total number of tokens present in the content
    num_tokens_in_document = len(tokenized_content)

    chunks = []
    # Keep track of the text content offset in bytes
    offset = 0

    for i in range(0, num_tokens_in_document, chunk_size_tokens):
        # Decode the tokens for a particular chunk (i.e. turn it back to the original content)
        text_chunk = enc.decode(tokenized_content[i : i + chunk_size_tokens])

        # We encode with utf-8 then take the length to ensure we're using byte offsets, and not
        # simply taking the length of the text string
        chunk_size_bytes = len(text_chunk.encode("utf-8"))

        byte_offset_begin = offset
        byte_offset_end = byte_offset_begin + chunk_size_bytes

        # For each text chunk, create a DocumentChunk object containing it's metadata
        document_chunk = DocumentChunk(
            text_chunk,
            offset_begin=byte_offset_begin,
            offset_end=byte_offset_end,
        )
        chunks.append(document_chunk)

        offset = byte_offset_end

    return chunks



def write_chunk_summary_comparison(
    output_file: str,
    chunks: List["DocumentChunk"],
    summaries: List["DocumentChunkSummary"],
    tokenizer_used: str = "gpt-4o",
    max_line_width: int = 100,
):
    """Write each chunk and its summary version to a text file, nicely formatted."""
    assert len(chunks) == len(
        summaries
    ), "len(chunks) and len(summaries) must be equal. (Each chunk needs a summary)"

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk, summary in zip(chunks, summaries):
            wrapped_chunk = textwrap.fill(str(chunk), max_line_width)
            wrapped_summary = textwrap.fill(str(summary), max_line_width)
            # Get lengths in chars
            char_length_chunk = len(chunk)
            char_length_summary = len(summary)
            # Get lengths in tokens
            token_length_chunk = get_token_length(chunk.content, tokenizer_used)
            token_length_summary = get_token_length(summary.content, tokenizer_used)

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
                f"\nCompression ratio (chars): {char_length_chunk/max(char_length_summary,1):.4f}x\n"
            )
            f.write(f"\nOriginal length (tokens): {token_length_chunk}")
            f.write(f"\nSummary length (tokens): {token_length_summary}")
            f.write(
                f"\nCompression ratio (tokens): {token_length_chunk/max(token_length_summary,1):.4f}x\n"
            )


def validate_arguments(args, parser):
    """Validates command-line arguments."""
    if args.chunk_size <= 0:
        parser.error("Chunk size must be a positive integer")

    if not (0 <= args.temperature <= 1):
        parser.error("Temperature must be in the interval [0,1]")

    if not os.path.exists(args.output_dir):
        print(f"Output directory '{args.output_dir}' does not exist, creating it.")
        os.mkdir(args.output_dir)


def print_config(args):
    """Prints the current configuration settings to the console."""
    print("\nCurrent Configuration:")
    print("=" * 30)
    print(f"Model             : {args.model}")
    print(
        f"Document Path(s)   : {list(map(lambda path: path.name, args.document_paths))}"
    )
    print(f"Chunk Size        : {args.chunk_size} tokens")
    print(f"Tokenizer         : {args.tokenizer}")
    print(f"Temperature       : {args.temperature}")
    print(
        f"Max New Tokens    : {args.max_new_tokens if args.max_new_tokens != -1 else 'No limit'}"
    )
    print("=" * 30 + "\n")


def print_summary_outcome(
    file_name: str,
    total_token_len_original_chunks: int,
    total_token_len_summaries: int,
    total_char_len_original_chunks: int,
    total_char_len_summaries,
):
    """Logs information about a summary request to the console, such as the ratio by which its characters and tokens were compressed"""
    file_name = f'"{file_name}"'
    # Print out compression ratio
    print(f"{file_name} total chunk token count: {total_token_len_original_chunks}")
    print(f"{file_name} total summary token count: {total_token_len_summaries}")
    print(
        f"{file_name} token compression ratio: {total_token_len_original_chunks/max(total_token_len_summaries,1):.4f}x\n"
    )
    print(f"{file_name} total chunk character count: {total_char_len_original_chunks}")
    print(f"{file_name} total summary character count: {total_char_len_summaries}")
    print(
        f"{file_name} total character compression ratio: {total_char_len_original_chunks/max(total_char_len_summaries,1):.4f}x\n"
    )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Split a document into chunks, then create summaries for each chunk."
    )
    parser.add_argument(
        "document_paths",
        nargs="*",
        type=argparse.FileType("r", encoding="utf-8"),
        help="Path(s) to the document(s) to summarize",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="What directory should we write the summaries to? Defaults to the current directory.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="The original document text will be split into chunks, each chunk will be this many tokens (default: 500)",
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
        default=0.15,
        help="Model temperature (a value in the interval [0,1], lower = more precision, higher = more creativity, default: 0.15)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=750,
        help="Maximum number of tokens generated for each summary. -1 indicates no limit (default: 750)",
    )
    args = parser.parse_args()
    # Validate arguments
    validate_arguments(args, parser)

    # Print the current configuration
    print_config(args)

    # Type annotate args.document_paths as TextIOWrapper list because argparse
    # automatically validates the paths and loads the files into TextIOWrapper objects)
    documents: List["TextIOWrapper"] = args.document_paths

    # Process and summarize each document
    for document in documents:
        file_name_without_ext = os.path.splitext(os.path.basename(document.name))[0]

        with Timer(
            f"Summarize {file_name_without_ext}",
            before_msg="Began summarizing file",
            after_msg="Completed summarizing file",
        ):

            # Read the contents of the file we want to chunk & summarize into a string
            document_content = document.read()
            # Split file into multiple chunks
            chunks = chunk_content(
                content=document_content,
                chunk_size_tokens=args.chunk_size,
                tokenizer=args.tokenizer,
            )

            # Request a summary for each chunk, while printing out the progress
            summaries = []
            for i in range(len(chunks)):
                with Timer(
                    f"Summarize chunk {i} in {file_name_without_ext}",
                    print_kwargs={"end": "\r"},
                ):
                    summaries.append(
                        request_chunk_summary(
                            chunks[i], args.model, args.temperature, args.max_new_tokens
                        )
                    )

        # Compute token and char lengths of the original and summaries, then print them out
        total_token_len_chunks = sum(
            get_token_length(chunk.content) for chunk in chunks
        )
        total_token_len_summaries = sum(
            get_token_length(summary.content) for summary in summaries
        )
        total_char_len_chunks = sum(len(chunk) for chunk in chunks)
        total_char_len_summaries = sum(len(summary) for summary in summaries)
        print("")
        print_summary_outcome(
            file_name_without_ext,
            total_token_len_chunks,
            total_token_len_summaries,
            total_char_len_chunks,
            total_char_len_summaries,
        )

        # Where to write the summaries of this document to
        output_summary_path = os.path.join(
            args.output_dir, f"{file_name_without_ext}_summaries_only.txt"
        )

        # Where to save the chunk summary comparison to
        output_comparison_path = os.path.join(
            args.output_dir, f"{file_name_without_ext}_chunk_summary_comparison.txt"
        )

        # Save the created summaries to a single text file
        with open(output_summary_path, "w+") as f:
            for summary in summaries:
                f.write(f"{summary}\n")

        # Create and write a chunk-summary pair comparison to a single text file
        write_chunk_summary_comparison(
            output_comparison_path,
            chunks,
            summaries,
        )


if __name__ == "__main__":
    main()
