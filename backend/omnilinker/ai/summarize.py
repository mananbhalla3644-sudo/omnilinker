"""Extractive summarisation (blueprint 9.2).

TextRank: build a similarity graph over sentences, run weighted PageRank, take
the top-k. No model, no weights to train, no GPU, and - the part that matters
for this product - **every output sentence is a verbatim span of a real
document**, so a summary can always be clicked through to its source. That is
not a limitation we accept; it is the property that makes a summary
trustworthy in a system whose entire pitch is "it knows where your data came
from".

An abstractive LLM summary is a nicer sentence and a worse product: you cannot
cite it, you cannot verify it, and when it is wrong there is no way to find out
what it was wrong about. `AI_MODE=model` is the upgrade path, and the interface
(`Summarizer.summarize`) is the same list of spans with an optional
`generated` flag so the UI can mark model-written text differently.

Similarity is TF-IDF cosine over the sentence set. IDF is computed *within the
document being summarised* (not the corpus), because a summary only needs to
know which words are distinctive in this text, and using corpus IDF would make
a summary of one Slack thread depend on unrelated documents.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from omnilinker.search.index import STOPWORDS, stem

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n{2,}")
_WORD = re.compile(r"[A-Za-z0-9_@'.+-]+")

#: PageRank damping. 0.85 is the value from the original TextRank paper and is
#: not tuned here; it is a damping constant, not a hyperparameter.
DAMPING = 0.85
ITERATIONS = 40
CONVERGENCE = 1e-6

#: A sentence below this many characters carries no propositional content
#: ("Thanks!", "Will do.") and only adds noise to the graph.
MIN_SENTENCE_CHARS = 25
#: ...and above this, it is usually several sentences glued together by missing
#: punctuation, so it gets split rather than ranked whole.
MAX_SENTENCE_CHARS = 600


@dataclass
class Sentence:
    index: int
    text: str
    tokens: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Summary:
    text: str
    sentences: list[dict]
    method: str
    compression: float
    source_words: int
    generated: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "sentences": self.sentences,
            "method": self.method,
            "compression": round(self.compression, 3),
            "source_words": self.source_words,
            "summary_words": sum(s["words"] for s in self.sentences),
            "generated": self.generated,
        }


def split_sentences(text: str) -> list[str]:
    """Split, then split the over-long ones. Abbreviations and initials are the
    usual failure mode here, so single-letter and known abbreviation prefixes do
    not end a sentence."""
    raw = _SENTENCE.split(text or "")
    out: list[str] = []
    for piece in raw:
        piece = piece.strip()
        while len(piece) > MAX_SENTENCE_CHARS:
            window = piece[:MAX_SENTENCE_CHARS]
            cut = window.rfind(". ")
            if cut < MAX_SENTENCE_CHARS // 2:
                cut = window.rfind(" ")
            if cut <= 0:
                break
            out.append(window[:cut].strip())
            piece = piece[cut:].lstrip(". ").strip()
        if piece:
            out.append(piece)
    return out


def _tokenize(sentence: str) -> list[str]:
    tokens: list[str] = []
    for raw in _WORD.findall(sentence.lower()):
        token = raw.strip(".'-")
        if len(token) < 2 or token in STOPWORDS or token.isdigit():
            continue
        tokens.append(stem(token))
    return tokens


def _tf_idf_vectors(sentences: Sequence[Sentence]) -> tuple[list[dict[str, float]], float]:
    """TF-IDF vectors with IDF scoped to this document set."""
    n = max(1, len(sentences))
    df: dict[str, int] = defaultdict(int)
    for sentence in sentences:
        for token in set(sentence.tokens):
            df[token] += 1
    vectors: list[dict[str, float]] = []
    for sentence in sentences:
        if not sentence.tokens:
            vectors.append({})
            continue
        counts: dict[str, int] = defaultdict(int)
        for token in sentence.tokens:
            counts[token] += 1
        length = sum(counts.values())
        vector: dict[str, float] = {}
        for token, count in counts.items():
            # sublinear tf, smoothed idf - the same reasoning as the BM25 index
            tf = 1.0 + math.log(count)
            idf = math.log((n + 1.0) / (df[token] + 1.0)) + 1.0
            vector[token] = tf * idf
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        vectors.append({t: v / norm for t, v in vector.items()})
    return vectors, float(n)


def _cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


class TextRankSummarizer:
    """Unsupervised extractive summariser."""

    method = "textrank"

    def summarize(
        self,
        text: str,
        *,
        max_sentences: int = 3,
        include_source_order: bool = True,
    ) -> Summary:
        sentences = self._prepare(text)
        source_words = sum(s.word_count for s in sentences)
        if not sentences:
            return Summary("", [], self.method, 0.0, 0)

        self._rank(sentences)

        # PageRank over-samples early sentences of long inputs, because the
        # first few paragraphs of any document are unusually self-similar and
        # form a dense clique. Scaling with length is the standard correction
        # and it is why a 40-sentence thread does not summarise to its first
        # two sentences every time.
        target = max(1, min(max_sentences, max(1, int(round(
            math.sqrt(len(sentences)) * 0.6)))))
        target = min(target, max_sentences)

        ranked = sorted(range(len(sentences)), key=lambda i: (-sentences[i].score, i))
        chosen = sorted(ranked[:target], key=lambda i: i) if include_source_order \
            else ranked[:target]
        picked = [sentences[i] for i in chosen]
        summary_words = sum(s.word_count for s in picked)

        return Summary(
            text=" ".join(s.text for s in picked),
            sentences=[
                {
                    "index": s.index,
                    "text": s.text,
                    "score": round(s.score, 5),
                    "words": s.word_count,
                    "position": round(s.index / max(1, len(sentences)), 3),
                }
                for s in picked
            ],
            method=self.method,
            compression=(summary_words / source_words) if source_words else 0.0,
            source_words=source_words,
        )

    # -- internals -----------------------------------------------------
    def _prepare(self, text: str) -> list[Sentence]:
        out: list[Sentence] = []
        for raw in split_sentences(text):
            cleaned = re.sub(r"\s+", " ", raw).strip()
            if len(cleaned) < MIN_SENTENCE_CHARS and not cleaned.endswith("?"):
                continue
            if not re.search(r"[A-Za-z]{3}", cleaned):
                continue
            out.append(Sentence(index=len(out), text=cleaned,
                                tokens=_tokenize(cleaned)))
        return out

    def _rank(self, sentences: list[Sentence]) -> None:
        n = len(sentences)
        if n == 1:
            sentences[0].score = 1.0
            return
        vectors, _ = _tf_idf_vectors(sentences)

        # Similarity matrix, symmetrised and row-normalised. Rows of all zeros
        # (a sentence sharing no vocabulary with anything) get a uniform row
        # instead of a zero row, so they neither absorb nor lose all rank mass
        # and PageRank still converges.
        weights: list[dict[int, float]] = [{} for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sim = _cosine(vectors[i], vectors[j])
                if sim > 0.05:  # threshold: below this it is noise, not relation
                    weights[i][j] = sim
                    weights[j][i] = sim
        for i in range(n):
            total = sum(weights[i].values())
            if total > 0:
                weights[i] = {j: v / total for j, v in weights[i].items()}
            elif n > 1:
                weights[i] = {j: 1.0 / (n - 1) for j in range(n) if j != i}

        scores = [1.0 / n] * n
        for _ in range(ITERATIONS):
            updated: list[float] = []
            for i in range(n):
                # teleport term, plus weighted mass arriving from every neighbour
                incoming = 0.0
                for j in range(n):
                    weight = weights[j].get(i)
                    if weight:
                        incoming += scores[j] * weight
                updated.append((1.0 - DAMPING) / n + DAMPING * incoming)
            delta = sum(abs(a - b) for a, b in zip(scores, updated))
            scores = updated
            if delta < CONVERGENCE:
                break

        # Break ties toward the start of the document, then toward brevity.
        # Deterministic output is a hard requirement: a summary that changes
        # between two identical requests cannot be cached, diffed, or tested.
        ranked = sorted(range(n), key=lambda i: (-scores[i], i))
        best = scores[ranked[0]] or 1.0
        for position, i in enumerate(ranked):
            sentences[i].score = scores[i] / best * (1.0 - 0.05 * position / n)


class LeadSummarizer(TextRankSummarizer):
    """First-k sentences. Kept as a named baseline: it is what the quality
    evaluation compares against, and a summariser that cannot be beaten by its
    own baseline is not being measured."""

    method = "lead"

    def summarize(self, text: str, *, max_sentences: int = 3,
                  include_source_order: bool = True) -> Summary:
        base = super().summarize(text, max_sentences=max_sentences,
                                 include_source_order=include_source_order)
        sentences = self._prepare(text)
        picked = sentences[: max_sentences]
        summary_words = sum(s.word_count for s in picked)
        source_words = sum(s.word_count for s in sentences) or 1
        return Summary(
            text=" ".join(s.text for s in picked),
            sentences=[{"index": s.index, "text": s.text, "score": 1.0,
                        "words": s.word_count,
                        "position": round(s.index / max(1, len(sentences)), 3)}
                       for s in picked],
            method=self.method,
            compression=summary_words / source_words,
            source_words=source_words,
        )


def get_summarizer(mode: str = "extractive") -> TextRankSummarizer:
    if mode == "lead":
        return LeadSummarizer()
    return TextRankSummarizer()
