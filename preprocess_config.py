from argparse import ArgumentParser
from dataclasses import dataclass
import os
from typing import List, Set

# Default values for arguments
_default_processes = 1
_default_minimum_document_length = 100
_default_legibility_ratio = 0.35
_default_output = f"preprocessed_vault.txt"

parser = ArgumentParser(
    prog="preprocess",
    description="Preprocess Obsidian vault into document chunks.",
)

# Input and Output Options
parser.add_argument(
    "directory",
    help="Directory to your Obsidian vault (which contains .md files)",
)
parser.add_argument(
    "-o",
    "--output",
    help="Where to save the processsed documents?",
    default=_default_output,
)
parser.add_argument(
    "-p",
    "--processes",
    type=int,
    default=_default_processes,
    help="The number of processes to run in parallel (higher may be faster depending on number of documents)",
)

# Text Preprocessing Options
parser.add_argument(
    "--lemmatize-words",
    dest="lemmatize_words",
    action="store_true",
    help="Enable lemmatization: reduce words to their base form (e.g., 'cats' -> 'cat', 'running' -> 'run').",
)
parser.add_argument(
    "--no-lemmatize-words",
    dest="lemmatize_words",
    action="store_false",
    help="Disable lemmatization.",
)
parser.set_defaults(lemmatize_words=False)

parser.add_argument(
    "--remove-special-chars-and-numbers",
    dest="remove_special_chars_and_numbers",
    action="store_true",
    help="Remove special characters and numbers from the text.",
)
parser.add_argument(
    "--no-remove-special-chars-and-numbers",
    dest="remove_special_chars_and_numbers",
    action="store_false",
    help="Keep special characters and numbers in the text.",
)
parser.set_defaults(remove_special_chars_and_numbers=True)

parser.add_argument(
    "--remove-urls",
    dest="remove_urls",
    action="store_true",
    help="Remove URLs from the text.",
)
parser.add_argument(
    "--no-remove-urls",
    dest="remove_urls",
    action="store_false",
    help="Keep URLs in the text.",
)
parser.set_defaults(remove_urls=True)

parser.add_argument(
    "--include-document-title",
    dest="include_document_title",
    action="store_true",
    help="Include the document title as a special token in the output text chunks.",
)
parser.add_argument(
    "--no-include-document-title",
    dest="include_document_title",
    action="store_false",
    help="Exclude the document title from the output text chunks.",
)
parser.set_defaults(include_document_title=False)

# Quality Filtering Options
parser.add_argument(
    "-mdl",
    "--minimum-document-length",
    type=int,
    default=_default_minimum_document_length,
    help="Minimum number of characters for a document to be kept",
)

parser.add_argument(
    "-bw",
    "--blacklist-words",
    nargs="+",
    default=[],
    dest="blacklisted_words",
    help="List of words to blacklist and remove from the text. Pass words separated by spaces. Example: --blacklist-words word1 word2. Note: The blacklisted words are matched AFTER the tokens have been filtered",
)

parser.add_argument(
    "-mlr",
    "--minimum-legibility-ratio",
    type=float,
    default=_default_legibility_ratio,
    help="Documents where the ratio of legible words to illegible words is less than mlr, are discarded.",
)


@dataclass
class PreprocessConfig:
    """
    Configuration options for the document preprocessing step.
    """

    # Path to Obsidian vault
    directory: str
    # Path to the output file
    output: str
    # Number of processes to run in parallel
    processes: int
    lemmatize_words: bool
    remove_special_chars_and_numbers: bool
    remove_urls: bool
    # Should the title of the document (file name) be included in its chunks?
    include_document_title: bool
    # Remove documents with less than `minimum_document_length`` characters
    minimum_document_length: int
    minimum_legibility_ratio: float
    # Remove these words from documents during preprocessing
    # Note: The blacklisted words are matched AFTER the tokens have been filtered
    blacklisted_words: Set["str"]


def get_preprocesser_config() -> PreprocessConfig:
    """
    Parses the command line arguments, validates them, and returns a config object.
    """
    args = parser.parse_args()

    # Get Obsidian vault directory & worker count
    directory = args.directory
    if not _validate_obsidian_vault_path(directory):
        parser.error("Invalid obsidian vault path")

    output = args.output
    if not _validate_output_path(output):
        parser.error("Invalid output path")

    processes = args.processes

    # Parse text pre-processing options
    lemmatize_words = args.lemmatize_words
    remove_special_chars_and_numbers = args.remove_special_chars_and_numbers
    remove_urls = args.remove_urls
    remove_urls = _handle_conflicting_url_removal(
        remove_special_chars_and_numbers, remove_urls
    )

    include_document_title = args.include_document_title

    # Parse quality filter options with potential warnings
    minimum_document_length = _validate_positive_int(
        args.minimum_document_length,
        "minimum_document_length",
        default_value=_default_minimum_document_length,
    )
    minimum_legibility_ratio = _validate_ratio_range(
        args.minimum_legibility_ratio,
        "minimum_legibility_ratio",
        default_value=_default_legibility_ratio,
    )
    blacklisted_words = set([word.lower() for word in args.blacklisted_words])

    # Create and return the config object
    return PreprocessConfig(
        directory=directory,
        output=output,
        processes=processes,
        lemmatize_words=lemmatize_words,
        remove_special_chars_and_numbers=remove_special_chars_and_numbers,
        remove_urls=remove_urls,
        include_document_title=include_document_title,
        minimum_document_length=minimum_document_length,
        minimum_legibility_ratio=minimum_legibility_ratio,
        blacklisted_words=blacklisted_words,
    )


def print_config(config: PreprocessConfig):
    """Print the PreprocessConfig to the console."""
    print("Preprocessing Configuration:")
    print(f"  Directory: {config.directory}")
    print(f"  Output File: {config.output}")
    print(f"  Processes: {config.processes}")
    print(f"  Lemmatize Words: {config.lemmatize_words}")
    print(
        f"  Remove Special Characters and Numbers: {config.remove_special_chars_and_numbers}"
    )
    print(f"  Remove URLs: {config.remove_urls}")
    print(f"  Include Document Title: {config.include_document_title}")
    print(f"  Minimum Document Length: {config.minimum_document_length}")
    print(f"  Minimum Legibility Ratio: {config.minimum_legibility_ratio}")
    print(
        f"  Blacklisted Words: {config.blacklisted_words if config.blacklisted_words else '{}'}"
    )


def _handle_conflicting_url_removal(
    remove_special_chars: bool, remove_urls: bool
) -> bool:
    if remove_special_chars and not remove_urls:
        print(
            "WARNING: --remove-special-chars-and-numbers is set to True, but --remove-urls is False. "
            "This will result in broken URLs. For consistent behavior, --remove-urls has been set to True."
        )
        return True
    return remove_urls


def _validate_obsidian_vault_path(obsidian_vault_path: str) -> bool:
    if not os.path.isdir(obsidian_vault_path):
        print(
            f"ERROR: The path to your Obsidian vault '{obsidian_vault_path}' is invalid or does not exist."
        )
        return False
    return True


def _validate_output_path(output_path: str) -> bool:
    """Validate that output_path is valid (if we can create a file at that path)"""
    output_dir = os.path.dirname(output_path) or "."

    # Ensure that the parent directory of output_path exists
    if not os.path.isdir(output_dir):
        print(
            f"ERROR: The parent directory for the output path '{output_dir}' does not exist."
        )
        return False

    # To test if we can write to the file, try opening it in append mode
    # this will create the file if it doesn't already exist
    try:
        with open(output_path, "a"):
            pass
    except IOError:
        print(
            f"ERROR: The output path '{output_path}' is not a valid file path or is not writable."
        )
        return False

    return True


def _validate_positive_int(value: int, arg_name: str, default_value: int) -> int:
    if value <= 0:
        print(
            f"WARNING: --{arg_name} was set to {value}, but must be a positive integer. "
            f"The {arg_name} has been set to {default_value}."
        )
        return default_value
    return value


def _validate_ratio_range(value: float, arg_name: str, default_value: float) -> float:
    if value < 0 or value > 1:
        print(
            f"WARNING: --{arg_name} was set to {value}, but must be between 0 and 1. "
            f"The {arg_name} has been set to {default_value}."
        )
        return default_value
    return value
