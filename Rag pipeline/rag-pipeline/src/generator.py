"""
generator.py
Builds a grounded prompt from retrieved chunks and calls the configured
LLM backend (Anthropic or OpenAI). Refuses to answer when retrieval
confidence is too low, instead of letting the model hallucinate.
"""

from __future__ import annotations

import os

from src.retriever import RetrievedChunk

LOW_CONFIDENCE_MESSAGE = (
    "I don't have enough relevant information in the indexed documents "
    "to answer that confidently. Try rephrasing, or add a document that "
    "covers this topic."
)

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's \
question using ONLY the provided context excerpts. Every claim must be \
traceable to the context. If the context does not contain the answer, say \
so explicitly rather than guessing. Cite sources inline using [source: <id>] \
after each claim that depends on that source."""


def build_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    context_block = "\n\n".join(
        f"[source: {r.chunk.chunk_id}]\n{r.chunk.text}" for r in retrieved
    )
    return (
        f"Context excerpts:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above, with inline [source: ...] citations."
    )


class Generator:
    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        min_retrieval_score: float | None = None,
    ):
        self.backend = backend or os.getenv("LLM_BACKEND", "anthropic")
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-6")
        self.min_retrieval_score = (
            min_retrieval_score
            if min_retrieval_score is not None
            else float(os.getenv("MIN_RETRIEVAL_SCORE", "0.35"))
        )

    def _passes_confidence_gate(self, retrieved: list[RetrievedChunk]) -> bool:
        if not retrieved:
            return False
        return max(r.score for r in retrieved) >= self.min_retrieval_score

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> dict:
        if not self._passes_confidence_gate(retrieved):
            return {
                "answer": LOW_CONFIDENCE_MESSAGE,
                "grounded": False,
                "sources": [],
                "top_score": max((r.score for r in retrieved), default=0.0),
            }

        prompt = build_prompt(query, retrieved)

        if self.backend == "anthropic":
            answer = self._call_anthropic(prompt)
        elif self.backend == "openai":
            answer = self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {self.backend}")

        return {
            "answer": answer,
            "grounded": True,
            "sources": [r.chunk.chunk_id for r in retrieved],
            "top_score": max(r.score for r in retrieved),
        }
