import argparse
from io import TextIOWrapper
from typing import List
from config import SummarizerConfig, SummaryOutputConfig, SummaryOutputHandler
from summarize import process_document
from utils import print_configs, validate_arguments


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
        "--comparisons",
        default=False,
        action="store_true",
        help="Whether or not to create the chunk summary comparison file. If argument is present, they will be created.",
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
    validate_arguments(args, parser)

    # Type annotate args.document_paths as TextIOWrapper list
    documents: List["TextIOWrapper"] = args.document_paths

    # Configuration setup
    summarizer_config = SummarizerConfig(
        args.model,
        args.chunk_size,
        args.tokenizer,
        args.temperature,
        args.max_new_tokens,
    )
    summary_output_config = SummaryOutputConfig(
        output_dir=args.output_dir,
        create_chunk_summary_comparisons=args.comparisons,
    )
    summary_output_handler = SummaryOutputHandler(config=summary_output_config)

    # Print the current configuration
    print_configs(summarizer_config, summary_output_config, documents)

    # Process and summarize each document
    for document in documents:
        process_document(
            document=document,
            summarizer_config=summarizer_config,
            summary_output_handler=summary_output_handler,
        )


if __name__ == "__main__":
    main()
