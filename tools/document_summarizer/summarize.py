from io import TextIOWrapper
import json
import os

import requests

from prompt import PromptChain, PromptChainBuilder
from chunk_document import DocumentChunkSummary, chunk_content, get_token_length
from config import SummarizerConfig, SummaryOutputHandler
from utils import Timer, print_summary_outcome


def request_chunk_summary(
    prompt_chain: PromptChain,
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
                    {"role": "system", "content": prompt_chain.system_prompt},
                    {"role": "user", "content": prompt_chain.user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "stream": "false",
            }
        ),
    ).json()

    return DocumentChunkSummary(
        content=response["choices"][0]["message"]["content"],
        original=prompt_chain.original_chunk_content,
    )


def process_document(
    document: TextIOWrapper,
    summarizer_config: SummarizerConfig,
    summary_output_handler: SummaryOutputHandler,
):
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
            chunk_size_tokens=summarizer_config.chunk_size,
            tokenizer=summarizer_config.tokenizer,
        )

        # Request a summary for each chunk, while printing out the progress
        summaries = []
        for i in range(len(chunks)):
            with Timer(
                f"Summarize chunk {i} in {file_name_without_ext}",
                print_kwargs={"end": "\r"},
            ):

                # Create a "prompt chain" to simplify including previous "ambient context"
                # from prior document chunk summaries. This updates the system and user prompt accordingly
                prompt_chain_builder = PromptChainBuilder().set_chunk_content(
                    chunks[i].content
                )

                # If we've previously created summaries for this document, include it in the ambient context
                if summaries:
                    prev_chunk: DocumentChunkSummary = summaries[-1]
                    prompt_chain_builder.set_ambient_context(prev_chunk.content)

                # Create the final prompts that will be sent to model
                prompt_chain = prompt_chain_builder.build()

                chunk_summary = request_chunk_summary(
                    prompt_chain,
                    summarizer_config.model,
                    summarizer_config.temperature,
                    summarizer_config.max_new_tokens,
                )

                summaries.append(chunk_summary)

    # Compute token and char lengths of the original and summaries, then print them out
    total_token_len_chunks = sum(get_token_length(chunk.content) for chunk in chunks)
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

    summary_output_handler.save(file_name_without_ext, summaries, chunks)
