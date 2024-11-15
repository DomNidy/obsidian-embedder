from dataclasses import dataclass

system_prompt_default = """
You will be provided with a chunk from a larger document, typically drawn from a user's personal notes. Summarize the chunk's purpose and topics, ensuring that your summary is significantly shorter than the original text and does not restate information multiple times. Use the following format:

"The purpose of this chunk appears to be [purpose of document], potentially touching on [general topic]. Key points include: [Main purpose or subject]. [Briefly list each significant topic with concise context, without additional detail]."

Respond in a continuous, clear paragraph without bullet points or lists.

Guidelines:

1. If the chunk contains irrelevant symbols, random characters, multiple URLs, or nonsensical phrases that do not contribute to meaningful content, respond with: "Chunk contains no understandable content."

2. If URLs appear central to the content, mention them briefly, such as, "This document includes URLs, likely relevant to [related topic if determinable]."

3. When encountering data formats like JSON or code snippets, summarize the general structure and its likely use, such as, "This JSON format may store user information," or "This code appears to perform [general function]."

4. Avoid providing any answers if the content includes questions; only summarize the information presented.

5. Conclude by relating briefly how the content might connect to the user's potential interests if evident.

Here is the content:
"""

system_prompt_with_ambient_context = """
You will process two text chunks to create a summary for the second chunk using context from the first. 

Guidelines:
1. Do not reference the first chunk explicitly.
2. Your summary must be concise and shorter than the second chunk.
3. Avoid repeating information or including irrelevant content.

Response Format:
The purpose of this chunk is: [Purpose].
Key points include: [Point 1], [Point 2], [Point 3].

Edge Cases:
- If the chunk has irrelevant symbols or nonsensical phrases: respond with "Chunk contains no understandable content."
- If URLs dominate: summarize as, "This document contains URLs related to [topic]."

Example:
First Chunk: "Global warming causes include CO2 emissions."
Second Chunk: "Renewable energy reduces CO2 and mitigates warming."
Output: "Renewable energy reduces CO2, helping mitigate warming."

"""


@dataclass
class PromptChain:
    system_prompt: str
    user_prompt: str
    original_chunk_content: str


class PromptChainBuilder:
    """Dynamically update system prompt and user prompt with 'ambient context' from previous chunks and summaries"""

    def __init__(self) -> None:
        self._ambient_context = None
        self._chunk_content = None

        # if using ambient context, this will need to include the proper structure specified in that system prompt
        # if not using ambient context, this is simply the content of the chunk
        self._user_prompt = ""

    def set_chunk_content(self, content: str) -> "PromptChainBuilder":
        """Set the content that the LLM should create a summary for"""
        self._chunk_content = content
        # If we previously set ambient context, include it before the current chunk
        if self._ambient_context is not None:
            self._user_prompt = f"Previous Chunk: {self._ambient_context}\nCurrent Chunk: {self._chunk_content}"
        else:
            self._user_prompt = self._chunk_content
        return self

    def set_ambient_context(self, ambient_context: str) -> "PromptChainBuilder":
        """Sets the ambient context that the LLM should use to gain greater insight into the document"""
        self._ambient_context = ambient_context
        # If we set ambient context BEFORE setting the chunk content, don't update the user prompt yet
        # (as ambient context requires some chunk content to be relevant or useful to the model)

        if self._chunk_content is not None:
            self._user_prompt = f"Previous Chunk: {self._ambient_context}\nCurrent Chunk: {self._chunk_content}"
        return self

    def build(self) -> "PromptChain":
        if self._ambient_context is not None:
            return PromptChain(
                system_prompt_with_ambient_context,
                self._user_prompt,
                self._chunk_content,
            )
        return PromptChain(
            system_prompt_default, self._user_prompt, self._chunk_content
        )
