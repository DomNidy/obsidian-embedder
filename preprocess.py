import os
import re
from time import time
import nltk
from nltk import corpus
from typing import List, Tuple
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


def preprocess_text(
    document_title: None,
    text: str,
    should_remove_urls: bool = True,
    should_remove_special_chars_and_numbers: bool = True,
    should_lemmatize_words: bool = False,
) -> str:
    """Preprocess text by removing special chars and numbers, tokenizing, lemmatizing, and removing stopwords."""

    # include special document token if desired
    if document_title is not None:
        document_special_token = f"{DOC_TITLE_SPECIAL_TOKEN}" + document_title + "]"

    tokens = text
    if should_remove_urls:
        tokens = remove_urls(tokens)
    if should_remove_special_chars_and_numbers:
        tokens = remove_special_chars_and_numbers(tokens)

    tokens = tokenize_text(tokens)
    if should_lemmatize_words:
        tokens = lemmatize_tokens(tokens)

    tokens = [token.lower() for token in tokens]

    # todo: find a better way to prepend the document title here, this is O(n) operation (pretty sure)
    final_tokens = []
    if document_title is not None:
        final_tokens.append(document_special_token)
    final_tokens.extend(tokens)
    return final_tokens


def process_document(
    document: Tuple[str, str],
    include_document_title: bool = False,
    should_remove_urls: bool = True,
    should_remove_special_chars_and_numbers: bool = True,
    should_lemmatize_words: bool = True,
) -> List[str] | None:
    """Process a single document and return preprocessed tokens."""
    title, path = document

    with open(path, "r", encoding="utf-8") as doc:
        doc_tokens = preprocess_text(
            document_title=title if include_document_title else None,
            text=doc.read(),
            should_remove_urls=should_remove_urls,
            should_remove_special_chars_and_numbers=should_remove_special_chars_and_numbers,
            should_lemmatize_words=should_lemmatize_words,
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
    num_processes = config.processes

    start = time()
    preprocessed_documents = multi_process_extract_and_preprocess_documents(
        document_paths, config=config
    )
    end = time()

    print(f"Completed in {end - start:.7f} seconds with {num_processes} process(es)")
    write_processed_documents(preprocessed_documents, config.output)
