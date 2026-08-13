"""Answer synthesis with a Claude backend and a dependency-free fallback.

If ``ANTHROPIC_API_KEY`` is set the service synthesises answers with Claude,
grounded in the retrieved passages. If it is not, it returns an extractive
answer stitched from the highest-scoring chunks. The fallback exists so the
repository is runnable by anyone who clones it -- retrieval quality, the
benchmark, and the API surface can all be evaluated without a key.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.agents.orchestrator import build_context
from app.generation.prompts import SYSTEM_PROMPT, build_user_message
from app.retrieval.store import SearchHit

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

NO_CONTEXT_MESSAGE = (
    "No indexed content matched this question. Ingest documents with POST /ingest, "
    "then try again."
)


@dataclass
class Answer:
    text: str
    mode: str  # "llm" | "extractive" | "none"


class Answerer(ABC):
    mode: str

    @abstractmethod
    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        ...


class ExtractiveAnswerer(Answerer):
    """No-LLM fallback: return the most relevant sentences, with citations.

    Not a summariser -- it selects. Sentences are scored by overlap with the
    question's content words, which keeps the output faithful to the source at
    the cost of reading less fluently than a synthesised answer.
    """

    mode = "extractive"

    def __init__(self, max_sentences: int = 6):
        self.max_sentences = max_sentences

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer(text=NO_CONTEXT_MESSAGE, mode="none")

        keywords = _content_words(question)
        scored: list[tuple[float, int, str]] = []

        for index, hit in enumerate(hits, start=1):
            for raw_sentence in _SENTENCE_SPLIT.split(hit.text.strip()):
                sentence = raw_sentence.strip()
                if len(sentence) < 30:
                    continue
                overlap = len(keywords & _content_words(sentence))
                if overlap == 0 and index > 1:
                    continue
                # Blend lexical overlap with the chunk's vector score so a
                # strong chunk still contributes when wording differs.
                score = overlap + hit.score
                scored.append((score, index, sentence))

        if not scored:
            top = hits[0]
            return Answer(text=f"{top.text.strip()[:600]} [1]", mode="extractive")

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: self.max_sentences]
        # Restore reading order by passage so the answer isn't a jumble.
        selected.sort(key=lambda item: item[1])

        lines = [f"{sentence} [{index}]" for _, index, sentence in selected]
        preamble = (
            "Answer assembled directly from the retrieved passages "
            "(set ANTHROPIC_API_KEY for synthesised answers):\n\n"
        )
        return Answer(text=preamble + " ".join(lines), mode="extractive")


class ClaudeAnswerer(Answerer):
    """Grounded synthesis via the Anthropic Messages API."""

    mode = "llm"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-5",
        max_tokens: int = 1024,
        effort: str = "low",
        thinking: str = "disabled",
        fallback: Answerer | None = None,
    ):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._anthropic = anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        self._fallback = fallback or ExtractiveAnswerer()

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer(text=NO_CONTEXT_MESSAGE, mode="none")

        context = build_context(hits)
        request: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_user_message(question, context)}],
            "output_config": {"effort": self.effort},
        }
        # Thinking is on by default on current models and counts against
        # max_tokens. For grounded extraction the retrieval has already done
        # the reasoning, so disabling it buys latency at no quality cost.
        # (Disabling is only permitted at effort "high" or below.)
        if self.thinking == "disabled" and self.effort in {"low", "medium", "high"}:
            request["thinking"] = {"type": "disabled"}
        else:
            request["thinking"] = {"type": "adaptive"}

        try:
            response = self._client.messages.create(**request)
        except self._anthropic.APIError as exc:
            logger.warning("Claude call failed (%s); falling back to extractive", exc)
            return self._fallback.answer(question, hits)

        # Always check stop_reason before reading content: a refusal returns
        # HTTP 200 with an empty content list.
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            logger.warning("Model declined the request (category=%s)", category)
            return Answer(
                text="The model declined to answer this question. Retrieved passages are "
                "listed below as citations.",
                mode="llm",
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        if not text:
            logger.warning(
                "Empty completion (stop_reason=%s); using fallback", response.stop_reason
            )
            return self._fallback.answer(question, hits)

        if response.stop_reason == "max_tokens":
            text += "\n\n[Answer truncated at the token limit; raise RAG_LLM_MAX_TOKENS.]"

        return Answer(text=text, mode="llm")


def _content_words(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with", "is",
        "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those",
        "as", "at", "by", "from", "what", "how", "why", "when", "where", "who", "which",
        "do", "does", "did", "can", "could", "would", "should", "about",
    }
    words = re.findall(r"[a-z0-9]{3,}", text.lower())
    return {w for w in words if w not in stop}


def build_answerer(
    model: str,
    max_tokens: int,
    effort: str,
    thinking: str = "disabled",
    api_key: str | None = None,
) -> Answerer:
    """Return the Claude answerer when a key is available, else the fallback."""
    key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
    if not key.strip():
        logger.info("ANTHROPIC_API_KEY not set -- using extractive answers")
        return ExtractiveAnswerer()

    try:
        answerer = ClaudeAnswerer(
            api_key=key, model=model, max_tokens=max_tokens, effort=effort, thinking=thinking
        )
        logger.info("Answer generation: Claude (%s, effort=%s)", model, effort)
        return answerer
    except ImportError:
        logger.warning("anthropic package not installed -- using extractive answers")
        return ExtractiveAnswerer()
