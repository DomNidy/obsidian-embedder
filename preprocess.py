import os
import re
import nltk
from nltk import corpus
from typing import List, Tuple
from multiprocessing import Pool
import time
import sys

# This is forced to re-run in each process
# but we need to create multiple processes because CPython GIL
nltk.download("words")

ENGLISH_WORDS = set(corpus.words.words())
LEMMATIZER = None  # Global variable to hold the lemmatizer in each worker

# Quality filters
MINIMUM_DOCUMENT_LENGTH = 100  # do not keep documents with less than 100 chars
MINIMUM_LEGIBILITY_RATIO = 0.35  # do not keep documents where more than this many words (as a percentage) are illegible


def initialize_worker_deps():
    """Initialize the lemmatizer and corpus word set for each worker process."""
    global LEMMATIZER
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


def preprocess_text(text: str) -> str:
    """Preprocess text by removing special chars and numbers, tokenizing, lemmatizing, and removing stopwords."""
    tokens = remove_urls(text)
    tokens = remove_special_chars_and_numbers(tokens)
    tokens = tokenize_text(tokens)
    tokens = lemmatize_tokens(tokens)
    tokens = [token.lower() for token in tokens]
    return tokens


def process_document(document: Tuple[str, str]) -> List[str] | None:
    """Process a single document and return preprocessed tokens."""
    title, path = document
    with open(path, "r", encoding="utf-8") as doc:
        doc_tokens = preprocess_text(doc.read())
    return doc_tokens


def filter_low_quality_documents(
    preprocessed_documents: List[str], legibility_ratio=0.25, min_len=100
) -> List[str]:
    """Returns a new array containg only high quality documents
    Documents which contain less than `legibility_ratio` legible englishwords are removed.
    Documents which have less than `min_len` characters are removed"""
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


def multi_process_extract_and_preprocess_documents(
    document_paths: List[Tuple[str, str]], num_workers: int = 4
) -> List[List[str]]:
    # Use a pool with initializer to create the lemmatizer only once per worker
    with Pool(processes=num_workers, initializer=initialize_worker_deps) as pool:
        # Process all documents in parallel
        preprocessed_documents = pool.map(process_document, document_paths)

    # Filter our documents that were too short to be processed
    before_filter = len(preprocessed_documents)
    preprocessed_documents = filter_low_quality_documents(
        preprocessed_documents, MINIMUM_LEGIBILITY_RATIO, MINIMUM_DOCUMENT_LENGTH
    )
    after_filter = len(preprocessed_documents)

    print(f"Filtered out {before_filter-after_filter} low quality documents.")

    return preprocessed_documents


def write_processed_documents(
    preprocessed_documents: List[List[str]], output_file: str
):
    with open(output_file, "w+") as file:
        for document in preprocessed_documents:
            doc_str = " ".join(document)
            file.write(doc_str + "\n")


if __name__ == "__main__":
    document_paths = load_document_paths(
        "C:\\vault"
    )  # load documents from obsidian vault
    num_processes = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    start = time.time()
    preprocessed_documents = multi_process_extract_and_preprocess_documents(
        document_paths, num_processes
    )
    end = time.time()

    print(f"Completed in {end - start:.7f} seconds with {num_processes} process(es)")
    write_processed_documents(preprocessed_documents, "preprocessed_docs.txt")
