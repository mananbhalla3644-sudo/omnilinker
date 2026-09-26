# ADR-024: deterministic AI, with the model as an upgrade

**Status:** accepted · **Area:** ai · **Reversible:** yes, configuration

## The blueprint specifies

LLM summarisation, LLM prediction, LLM entity extraction.

## The default is

TextRank extraction, rule-based prediction, rule-based entity and date
extraction. All offline, all deterministic, no model.

## Why

**Every output must be clickable to its source.** This is the product's whole
claim — it knows where your data came from — and a generated summary is a nicer
sentence and a worse product. You cannot cite it, you cannot verify it, and when
it is wrong there is no way to find out what it was wrong about. An extractive
summary is a set of verbatim spans; `test_ai.py::test_summary_sentences_come_from_the_source`
asserts every one of them appears in the source text.

**A linear importance model can explain itself.** The dashboard shows *why* a
document is ranked high — owner, extracted deadline, decision language,
substance, recency, each with its weight. A user looking at a "high importance"
badge has to get an answer, and a linear model over named features is the only
kind that can produce one.

**Rule-based prediction carries its evidence.** A deadline names the sentence it
came from and the cue that made it a commitment. An unanswered question names the
thread and the count of messages that followed it. When the system is wrong, you
can see the input that made it wrong.

**Determinism makes it testable.** Byte-identical output for identical input is
what lets the summary be cached, diffed, and asserted on.

## Residual risk

**Extractive summaries cannot merge facts across documents.** A five-document
thread with one sentence per document summarises to a list, not to a synthesis.
That is the real cost, and it is the gap a model closes.

**Rule-based extraction misses implicit entities.** "Can you look at the deck
Alice sent?" yields a file reference and no person, because the name is not
adjacent to a known handle. A model would catch it.

**Both are measurable, and that is the point.** The same evaluation set scores
either implementation, so the upgrade is a decision made on evidence rather than
on taste.

## The upgrade path

Interfaces do not change, so the escalation is configuration:

- `AI_MODE=local|cloud` routes to a model gateway.
- `embeddings.embed()` is already a model-shaped interface.
- `NL2QueryCompiler(llm=...)` already accepts a rewriter — and it is constrained
  to return a *plan fragment*, validated field by field, with anything
  unsupported reported in `unsupported` rather than dropped. A hallucinated
  filter cannot reach the store, because the model proposes and the schema
  disposes.

The deterministic path stays available as the baseline the model is measured
against. `LeadSummarizer` exists for the same reason: a summariser that cannot
be beaten by its own baseline is not being measured.
