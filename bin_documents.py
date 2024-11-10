from typing import List, Optional, Set
from sentence_transformers import SentenceTransformer
import torch as t
import sys
import textwrap  # For pretty printing
import re

device = "cuda" if t.cuda.is_available() else "cpu"

# Load a pretrained Sentence Transformer model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME, device=device)

# Parameters
MINIMUM_SIMILARITY_TO_BIN = 0.5
TOP_K = 25  # Number of most similar documents to include in each bin (MAX BIN SIZE)
MAX_LINE_WIDTH = 155  # Line width for wrapping text in output


def load_sentences(filepath: str) -> list:
    """Load preprocessed dataset from a text file."""
    with open(filepath, "r") as file:
        return file.read().splitlines()


def compute_similarity_matrix(embeddings) -> t.Tensor:
    """Compute similarity matrix and set diagonal to a low value to ignore self-similarities."""
    similarities = model.similarity(embeddings, embeddings)
    similarities.fill_diagonal_(-100)  # Arbitrarily low to ignore self-similarity
    return similarities


def bin_documents(similarities: t.Tensor) -> list:
    """Create bins of similar documents based on similarity scores."""
    bins = []
    already_binned_document_indices = set()

    for i, similarity_scores in enumerate(similarities):
        if i in already_binned_document_indices:
            continue  # Skip already binned documents

        # Initialize a new bin with the current document
        curr_doc_bin = {i}
        already_binned_document_indices.add(i)

        # Get top k most similar documents
        top_k_indices = t.topk(similarity_scores, TOP_K).indices

        # Filter based on minimum similarity and add to bin
        for j in top_k_indices:
            similarity_value = similarity_scores[j].item()
            if (
                similarity_value > MINIMUM_SIMILARITY_TO_BIN
                and j not in already_binned_document_indices
            ):
                curr_doc_bin.add(int(j))
                already_binned_document_indices.add(int(j))

        # Only add bins with more than one document
        if len(curr_doc_bin) > 1:
            bins.append(curr_doc_bin)

    return bins


def get_document_titles_from_bin(
    bin: Set[int], sentences: List[str]
) -> List[Optional[str]]:
    """
    Extracts document titles from a single bin using the DOCUMENT_TITLE special token.

    This function attempts to match document title special token with the pattern '[DOCUMENT_TITLE=*]'
    in each document within the bin. Note: this only works if the preprocessor config used to create
    the document chunks set the 'include document title' special token option to true.

    Parameters:
        bin (Set[int]): A Set of document indices within a bin.
        sentences (List[str]): List of all processed document chunks, we will index into these to retrieve the document text.

    Returns:
        List[Optional[str]]: A list of document titles if found, or None if a title is missing.
    """
    document_titles = []
    document_title_pattern = r"\[DOCUMENT_TITLE=(.*?)\]"

    for doc_idx in bin:
        sentence = sentences[doc_idx]
        match = re.match(document_title_pattern, sentence)

        if match:
            document_titles.append(
                match.group(1)
            )  # Extract the title within the parentheses
        else:
            document_titles.append(None)  # No title found for this document index

    if all(title is None for title in document_titles):
        print(
            "No document titles found. Ensure the preprocessor uses the 'include document title' special token option."
        )

    return document_titles


def write_binned_output(bins: list, sentences: list, output_file: str):
    """Write binned output to a file with formatted document content."""
    with open(output_file, "w") as file:
        for bin_index, bin in enumerate(bins, start=1):
            file.write(f"Bin {bin_index} (size={len(bin)}): {bin}\n")

            for document_index in bin:
                file.write(f"\tDocument {document_index}:\n")
                file.write(
                    "\n".join(
                        textwrap.wrap(
                            sentences[document_index],
                            width=MAX_LINE_WIDTH,
                            initial_indent="\t\t",
                            subsequent_indent="\t\t",
                        )
                    )
                )
                file.write("\n\n")
    print(f"Found {len(bins)} unique categories.")


if __name__ == "__main__":
    # Path to the preprocessed document text file containing all the document chunks
    document_chunks_file_path = str(sys.argv[1]) if len(sys.argv) > 1 else None
    # Path to write the binned chunks to
    output_path = str(sys.argv[2]) if len(sys.argv) > 2 else "binned.txt"

    if not document_chunks_file_path:
        raise FileNotFoundError(
            "Invalid path to document chunk file:", document_chunks_file_path
        )

    sentences = load_sentences(document_chunks_file_path)
    embeddings = model.encode(sentences)
    similarities = compute_similarity_matrix(embeddings)
    bins = bin_documents(similarities)

    for bin in bins:
        print(f"Bin: {bin}")
        for doc_title in get_document_titles_from_bin(bin, sentences):
            print(f'\t"{doc_title}"')

    write_binned_output(bins, sentences, output_path)
