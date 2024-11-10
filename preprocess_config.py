from argparse import ArgumentParser
from dataclasses import dataclass

# Default values for arguments
_default_processes = 1
_default_minimum_document_length = 100
_default_legibility_ratio = 0.35

parser = ArgumentParser(
    prog="preprocess",
    description="Preprocess Obsidian vault into document chunks.",
)

# Input and Output Options
parser.add_argument(
    "directory", help="Directory to your Obsidian vault (which contains .md files)"
)
parser.add_argument(
    "-p",
    "--processes",
    type=int,
    default=_default_processes,
    help="The number of processes to run in parallel (more = faster)",
)

# Text Preprocessing Options
parser.add_argument(
    "-lw",
    "--lemmatize-words",
    action="store_true",
    default=False,
    help="Lemmatize words: Reduce words to their base form (e.g., 'cats' -> 'cat', 'running' -> 'run')",
)
parser.add_argument(
    "-rsc",
    "--remove-special-chars-and-numbers",
    action="store_true",
    default=True,
    help="Remove special characters and numbers from the text (Note: Setting this to true will corrupt URLs, irrespective of the --remove-urls option.)",
)
parser.add_argument(
    "-ru",
    "--remove-urls",
    default=True,
    action="store_true",
    help="Remove URLs from the text",
)
parser.add_argument(
    "-idt",
    "--include-document-title",
    default=False,
    action="store_true",
    help="Include the document title as a special token in the output text chunks",
)

# Quality Filtering Options
parser.add_argument(
    "-mdl",
    "--minimum-document-length",
    type=int,
    default=_default_minimum_document_length,
    help="Minimum number of characters for a document to be kept",
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

    # The obsidian vault
    directory: str
    processes: int
    lemmatize_words: bool
    remove_special_chars_and_numbers: bool
    remove_urls: bool
    include_document_title: bool
    minimum_document_length: int
    minimum_legibility_ratio: float


def get_preprocesser_config() -> PreprocessConfig:
    """
    Parses the command line arguments, validates them, and returns a config object.
    """
    args = parser.parse_args()

    # Get Obsidian vault directory & worker count
    directory = args.directory
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

    # Create and return the config object
    return PreprocessConfig(
        directory=directory,
        processes=processes,
        lemmatize_words=lemmatize_words,
        remove_special_chars_and_numbers=remove_special_chars_and_numbers,
        remove_urls=remove_urls,
        include_document_title=include_document_title,
        minimum_document_length=minimum_document_length,
        minimum_legibility_ratio=minimum_legibility_ratio,
    )


def print_config(config: PreprocessConfig):
    """Print the PreprocessConfig to the console."""
    print("Preprocessing Configuration:")
    print(f"  Directory: {config.directory}")
    print(f"  Processes: {config.processes}")
    print(f"  Lemmatize Words: {config.lemmatize_words}")
    print(
        f"  Remove Special Characters and Numbers: {config.remove_special_chars_and_numbers}"
    )
    print(f"  Remove URLs: {config.remove_urls}")
    print(f"  Include Document Title: {config.include_document_title}")
    print(f"  Minimum Document Length: {config.minimum_document_length}")
    print(f"  Minimum Legibility Ratio: {config.minimum_legibility_ratio}")


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
