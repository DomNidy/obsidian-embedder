import os
import re
from time import time
import nltk
from nltk import corpus
from typing import List, Optional, Set, Tuple
from multiprocessing import Pool
from preprocess_config import get_preprocesser_config, print_config, PreprocessConfig

# This is forced to re-run in each process
nltk.download("words", quiet=True)

ENGLISH_WORDS = set(corpus.words.words())
DOC_TITLE_SPECIAL_TOKEN = (
    "[DOCUMENT_TITLE="  # Special token used to prevent filtering out document title
)
LEMMATIZER = None  # Global variable to hold the lemmatizer in **each process**


def initialize_worker_deps(lemmatize_words: bool):
    """Initialize the lemmatizer and corpus word set for each worker process."""
    global LEMMATIZER
    if lemmatize_words:
        LEMMATIZER = nltk.WordNetLemmatizer()


def load_document_paths(root_dir: str, allow_extensions=[".md", ".txt"]):
    """Return array of (document_name, document_path). These tuples are paths to documents."""
    document_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(tuple(allow_extensions)):
                path = os.path.join(dirpath, filename)
                document_paths.append((filename, path))
    return document_paths


def tokenize_text(text: str) -> list[str]:
    """Tokenize input text into individual words."""
    return nltk.word_tokenize(text)


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize tokens to their base form."""
    return [LEMMATIZER.lemmatize(token) for token in tokens]


def remove_special_chars_and_numbers(text: str) -> str:
    """Remove special characters and numbers from the input text."""
    return re.sub(r"[^a-zA-Z\s]", "", text)


def remove_urls(text: str) -> str:
    """Remove URLs from the input text."""
    return re.sub(
        r"(https?://\S+|www\.\S+|\b\w+\.(com|org|net|io)\b)",
        "",
        text,
        flags=re.IGNORECASE,
    )


def remove_blacklisted_words(
    tokens: List[str], blacklisted_words: Set[str]
) -> List[str]:
    """Returns a new list of tokens, excluding all words in `blacklisted_words`"""
    return [word for word in tokens if word not in blacklisted_words]


def preprocess_text(
    document_title: Optional[str],
    text: str,
    should_remove_urls: bool = True,
    should_remove_special_chars_and_numbers: bool = True,
    should_lemmatize_words: bool = False,
    blacklisted_words: Optional[Set[str]] = None,
) -> str:
    """Preprocess text by removing special chars and numbers, tokenizing, lemmatizing, and removing stopwords."""
    tokens = text

    # Create special document title token if desired
    if document_title is not None:
        document_special_token = f"{DOC_TITLE_SPECIAL_TOKEN}" + document_title + "]"

    # Filter by removing urls and special chars if desireed
    if should_remove_urls:
        tokens = remove_urls(tokens)
    if should_remove_special_chars_and_numbers:
        tokens = remove_special_chars_and_numbers(tokens)

    # Split text into list of tokens
    tokens = tokenize_text(tokens)
    if should_lemmatize_words:
        tokens = lemmatize_tokens(tokens)

    # Lowercase all tokens
    tokens = [token.lower() for token in tokens]

    # Remove blacklisted words if any were specified
    tokens = (
        remove_blacklisted_words(tokens, blacklisted_words)
        if blacklisted_words
        else tokens
    )

    # If requested, prepend the document title special token
    if document_title is not None:
        tokens.insert(0, document_special_token)

    return tokens


def process_document(
    document: Tuple[str, str],
    include_document_title: bool = False,
    should_remove_urls: bool = True,
    should_remove_special_chars_and_numbers: bool = True,
    should_lemmatize_words: bool = True,
    blacklisted_words: Optional[Set[str]] = None,
) -> List[str] | None:
    """
    Processes a single document by tokenizing it, performing a sequence of filtering operations on it and
    then enriching it with optional special tokens. Returns a list of tokens.

    The filtering operations performed depend on the config (parameters), and are as follows:
    - remove urls
    - remove numbers and special characters
    - lemmatize words

    'special tokens' include:
    - document title (prepend a token containing the file name from which the tokens were produced)
    """
    title, path = document

    with open(path, "r", encoding="utf-8") as doc:
        doc_tokens = preprocess_text(
            document_title=title if include_document_title else None,
            text=doc.read(),
            should_remove_urls=should_remove_urls,
            should_remove_special_chars_and_numbers=should_remove_special_chars_and_numbers,
            should_lemmatize_words=should_lemmatize_words,
            blacklisted_words=blacklisted_words,
        )
    return doc_tokens


def filter_low_quality_documents(
    preprocessed_documents: List[str], legibility_ratio=0.25, min_len=100
) -> List[str]:
    """
    Returns a new array containg only high quality documents
    Documents which contain less than `legibility_ratio` legible english words are removed.
    Documents which have less than `min_len` characters are removed
    """
    filtered_documents = []
    total_legibility = 0
    for document in preprocessed_documents:

        if len(document) < min_len:
            continue

        doc_words = set(document)
        legible_words = ENGLISH_WORDS.intersection(doc_words)
        doc_legibility = len(legible_words) / max(len(doc_words), 1)
        if doc_legibility < legibility_ratio:
            continue
        else:
            total_legibility += doc_legibility
            filtered_documents.append(document)

    print(
        f"Avg document legibility after filtering: {total_legibility/len(filtered_documents)}"
    )

    return filtered_documents


def write_processed_documents(
    preprocessed_documents: List[List[str]], output_file: str
):
    with open(output_file, "w+", encoding="utf-8") as file:
        for document in preprocessed_documents:
            doc_str = " ".join(document)
            file.write(doc_str + "\n")


def multi_process_extract_and_preprocess_documents(
    document_paths: List[Tuple[str, str]], config: PreprocessConfig
) -> List[List[str]]:
    """
    Use multiple worker processes to process multiple documents concurrently
    """

    # Create tuples of (document_path, include_document_title) for each document path
    # This ensures the .map method correctly receives both arguments
    process_document_args = [
        (
            doc_path,
            config.include_document_title,
            config.remove_urls,
            config.remove_special_chars_and_numbers,
            config.lemmatize_words,
            config.blacklisted_words,
        )
        for doc_path in document_paths
    ]

    # Use a pool with initializer to create the lemmatizer only once per worker
    with Pool(
        processes=config.processes,
        initializer=initialize_worker_deps,
        initargs=(config.lemmatize_words,),
    ) as pool:
        preprocessed_documents = pool.starmap(
            process_document,
            process_document_args,
        )

    # Filter out low quality documents
    before_filter = len(preprocessed_documents)
    preprocessed_documents = filter_low_quality_documents(
        preprocessed_documents,
        config.minimum_legibility_ratio,
        config.minimum_document_length,
    )
    after_filter = len(preprocessed_documents)

    print(f"Filtered out {before_filter-after_filter} low quality documents.")

    return preprocessed_documents


if __name__ == "__main__":
    config = get_preprocesser_config()
    print_config(config)

    # load documents from obsidian vault
    document_paths = load_document_paths(config.directory)
    for _, filepath in document_paths:
        print(filepath)
    num_processes = config.processes

    start = time()
    preprocessed_documents = multi_process_extract_and_preprocess_documents(
        document_paths, config=config
    )
    end = time()

    print(f"Completed in {end - start:.7f} seconds with {num_processes} process(es)")
    write_processed_documents(preprocessed_documents, config.output)
