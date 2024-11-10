from sentence_transformers import SentenceTransformer
import torch as t
import textwrap  # For pretty printing

device = "cuda" if t.cuda.is_available() else "cpu"

# Load a pretrained Sentence Transformer model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME, device=device)

# Parameters
MINIMUM_SIMILARITY_TO_BIN = 0.5
TOP_K = 25  # Number of most similar documents to include in each bin
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
    sentences = load_sentences("preprocessed_docs.txt")
    embeddings = model.encode(sentences)
    similarities = compute_similarity_matrix(embeddings)
    bins = bin_documents(similarities)
    write_binned_output(bins, sentences, "binned.txt")
