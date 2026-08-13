"""Prompts for grounded answer synthesis."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You answer questions strictly from the numbered context passages you are given.

Grounding rules:
- Use only the provided passages. Do not add facts from your own knowledge.
- Cite the passage number in square brackets after each claim, like [2]. Cite \
every claim; a sentence drawing on two passages gets both, like [1][3].
- If the passages do not contain the answer, say so plainly in one sentence and \
stop. Do not guess, and do not pad the response with what the passages do cover \
unless it directly bears on the question.
- If the passages conflict, say so and cite each side.

Style:
- Lead with the answer. Supporting detail comes after.
- Be concise: cover the substance without filler, preamble, or a restatement of \
the question.
- Plain prose. Use a short list only when the answer is genuinely enumerable.
- Do not include internal or system XML tags in your response.\
"""

USER_TEMPLATE = """\
<context>
{context}
</context>

Question: {question}

Answer the question using only the context above, citing passage numbers."""


def build_user_message(question: str, context: str) -> str:
    return USER_TEMPLATE.format(context=context, question=question)
