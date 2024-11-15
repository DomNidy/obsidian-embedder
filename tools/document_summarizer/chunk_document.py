from typing import List
import tiktoken


def get_token_length(content: str, tokenizer="gpt-4o") -> int:
    """Return the length of text content in tokens."""
    enc = tiktoken.encoding_for_model(tokenizer)
    return len(enc.encode(content))


def chunk_content(
    content: str, chunk_size_tokens: int = 1000, tokenizer: str = "gpt-4o"
) -> List["DocumentChunk"]:
    """
    Return a `DocumentChunk` list from text content, where each chunk's content is `chunk_size` tokens in length.
    Each `DocumentChunk` will record it's byte offset (relative to the start of the provided `content`)
    """
    enc = tiktoken.encoding_for_model(tokenizer)
    # Create an array of tokens for the content
    tokenized_content = enc.encode(content)
    # Total number of tokens present in the content
    num_tokens_in_document = len(tokenized_content)

    chunks = []
    # Keep track of the text content offset in bytes
    offset = 0

    for i in range(0, num_tokens_in_document, chunk_size_tokens):
        # Decode the tokens for a particular chunk (i.e. turn it back to the original content)
        text_chunk = enc.decode(tokenized_content[i : i + chunk_size_tokens])

        # We encode with utf-8 then take the length to ensure we're using byte offsets, and not
        # simply taking the length of the text string
        chunk_size_bytes = len(text_chunk.encode("utf-8"))

        byte_offset_begin = offset
        byte_offset_end = byte_offset_begin + chunk_size_bytes

        # For each text chunk, create a DocumentChunk object containing it's metadata
        document_chunk = DocumentChunk(
            text_chunk,
            offset_begin=byte_offset_begin,
            offset_end=byte_offset_end,
        )
        chunks.append(document_chunk)

        offset = byte_offset_end

    return chunks


class DocumentChunk:
    """Object containing metadata about a chunk of text taken out of a larger document."""

    def __init__(self, content: str, offset_begin: int, offset_end: int) -> None:

        assert (
            offset_begin <= offset_end
        ), f"`offset_begin` ({offset_begin}) must be less than or equal to `offset_end` ({offset_end})"

        self.content = content
        self.offset_begin = offset_begin
        self.offset_end = offset_end

    def __str__(self):
        return str(self.content)

    def __len__(self):
        return len(self.content)


class DocumentChunkSummary:
    """Object containing summarized content and metadata about a summary created from a `DocumentChunk`."""

    def __init__(self, content: str, original: "DocumentChunk") -> None:
        self.original = original
        self.content = content

    def __str__(self):
        return str(self.content)

    def __len__(self):
        return len(self.content)
