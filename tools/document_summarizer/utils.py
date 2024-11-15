from dataclasses import dataclass, field
from io import TextIOWrapper
import os
from time import time
from typing import Any, Dict, List, Optional

from config import SummarizerConfig, SummaryOutputConfig


def validate_arguments(args, parser):
    """Validates command-line arguments."""
    if args.chunk_size <= 0:
        parser.error("Chunk size must be a positive integer")

    if not (0 <= args.temperature <= 1):
        parser.error("Temperature must be in the interval [0,1]")

    if not os.path.exists(args.output_dir):
        print(f"Output directory '{args.output_dir}' does not exist, creating it.")
        os.mkdir(args.output_dir)


def print_configs(
    summarizer_config: SummarizerConfig,
    summary_output_config: SummaryOutputConfig,
    documents: List[TextIOWrapper],
):
    """Prints the current configuration settings to the console."""
    print("\nCurrent Configuration:")
    print("=" * 30)
    print(f"Model                  : {summarizer_config.model}")
    print(f"Tokenizer              : {summarizer_config.tokenizer}")
    print(f"Temperature            : {summarizer_config.temperature}")
    print(
        f"Max New Tokens         : {summarizer_config.max_new_tokens if summarizer_config.max_new_tokens != -1 else 'No limit'}"
    )
    print(f"Chunk Size             : {summarizer_config.chunk_size} tokens")
    print(f"Document Path(s)       : {list(map(lambda path: path.name, documents))}")
    print(f"Output Directory       : {summary_output_config.output_dir}")
    print(
        f"Create Comparison File : {summary_output_config.create_chunk_summary_comparisons}"
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
