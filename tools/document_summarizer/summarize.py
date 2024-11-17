from io import TextIOWrapper
import json
import logging
import os

import requests

from prompt import PromptChain, PromptChainBuilder
from chunk_document import DocumentChunkSummary, chunk_content
from config import SummarizerConfig, SummaryOutputHandler
from utils import Timer, calculate_summary_statistics, print_summary_outcome


LLM_API_ENDPOINT = "http://127.0.0.1:1234/v1/chat/completions"


def _build_request_body(
    prompt_chain: PromptChain,
    model: str,
    temperature: float,
    max_new_tokens: int,
) -> dict:
    """
    Creates JSON request body that will be sent to the LLM api.
    The schema defined below is specifically tailored to an LM studio server.
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt_chain.system_prompt},
            {"role": "user", "content": prompt_chain.user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "stream": False,  # Use boolean instead of string
    }


def _validate_llm_response(response: str):
    """
    Checks for undesired tokens/structure in the LLM summary response.
    For example, we don't want linebreaks.
    """
    response = response.replace("\n", " ").replace("  ", " ")
    return response


class LLMAPIError(Exception):
    """Exception for LLM API request related errors"""

    pass


def request_chunk_summary(
    prompt_chain: PromptChain,
    model: str,
    temperature: float = 0.25,
    max_new_tokens: int = -1,
) -> "DocumentChunkSummary":
    """
    Request a summary for a DocumentChunk from the LLM API endpoint.

    Args:
        `prompt_chain`: PromptChain object containing system and user prompts
        `model`: Name of the model to use
        `temperature`: Float between 0 and 1 controlling randomness
        `max_new_tokens`: Maximum number of tokens to generate (-1 for unlimited)

    Returns:
        DocumentChunkSummary containing the generated summary

    Raises:
        LLMAPIError: If API request fails
    """

    request_body = _build_request_body(prompt_chain, model, temperature, max_new_tokens)

    try:
        response = requests.post(
            LLM_API_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=request_body,
        )
        response.raise_for_status()

        response_data = response.json()
        summary = _validate_llm_response(
            response_data["choices"][0]["message"]["content"]
        )

        return DocumentChunkSummary(
            content=summary,
            original=prompt_chain.original_chunk_content,
        )

    except Exception as e:
        logging.error(f"Failed to generate summary: {str(e)}")
        raise LLMAPIError(f"Failed to generate summary: {str(e)}")


def process_document(
    document: TextIOWrapper,
    summarizer_config: SummarizerConfig,
    summary_output_handler: SummaryOutputHandler,
):
    """
    Process a document by chunking it and generating summaries.

    Args:
        `document`: The input document to process
        `summarizer_config`: Configuration for the summarization process
        `summary_output_handler`: Handler for saving summaries

    Raises:
        `ValueError`: If document is invalid
        `RuntimeError`: If processing fails
    """
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
    stats = calculate_summary_statistics(chunks, summaries)

    print("")
    print_summary_outcome(
        file_name_without_ext,
        stats["token_len_chunks"],
        stats["token_len_summaries"],
        stats["char_len_chunks"],
        stats["char_len_summaries"],
    )

    summary_output_handler.save(file_name_without_ext, summaries, chunks)
