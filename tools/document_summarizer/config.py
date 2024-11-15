from dataclasses import dataclass
import os
import textwrap
from typing import List, Optional
from chunk_document import (
    DocumentChunk,
    DocumentChunkSummary,
    get_token_length,
)


@dataclass
class SummarizerConfig:
    model: str
    chunk_size: int
    tokenizer: str
    temperature: float
    max_new_tokens: int


@dataclass
class SummaryOutputConfig:
    """
    Controls how document summaries are saved to disk.

    output_dir: The directory that all output files will be saved to.
    create_chunk_summary_comparisons: If true, an additional file will be output for each document that
    compares the original content with the summarized content. (useful for debugging LLM outputs)
    """

    output_dir: str
    create_chunk_summary_comparisons: bool


class SummaryOutputHandler:
    """
    Responsible for writing and saving the summaries to disk
    """

    def __init__(self, config: SummaryOutputConfig) -> None:
        self.config = config

    def save(
        self,
        file_name: str,
        summaries: List["DocumentChunkSummary"],
        chunks: Optional[List["DocumentChunk"]],
    ):
        self._save_summaries(file_name, summaries)
        if self.config.create_chunk_summary_comparisons == True and chunks is not None:
            self._save_chunk_summary_comparison(file_name, chunks, summaries)

    def _save_summaries(self, file_name: str, summaries: List["DocumentChunkSummary"]):
        output_summary_file = os.path.join(
            self.config.output_dir, f"{file_name}_summaries_only.txt"
        )
        # Save the created summaries to a single text file
        with open(output_summary_file, "w+", encoding="utf-8") as f:
            try:
                for summary in summaries:
                    f.write(f"{summary}\n")
            except Exception as e:
                print(f"Error occured while writing summary: {str(e)}")

    def _save_chunk_summary_comparison(
        self,
        file_name: str,
        chunks: List["DocumentChunk"],
        summaries: List["DocumentChunkSummary"],
    ):
        """Write each chunk and its summary version to a text file, nicely formatted."""
        output_comparison_file = os.path.join(
            self.config.output_dir, f"{file_name}_chunk_summary_comparison.txt"
        )

        # TODO: Find a good way to read these from parameters instead of hardcoding
        max_line_width = 125
        tokenizer_used = "gpt-4o"

        assert len(chunks) == len(
            summaries
        ), "len(chunks) and len(summaries) must be equal. (Each chunk needs a summary)"

        with open(output_comparison_file, "w", encoding="utf-8") as f:
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
