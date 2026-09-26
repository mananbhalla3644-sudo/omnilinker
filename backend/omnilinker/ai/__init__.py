"""AI layer: summarisation, prediction, enrichment.

All three are deterministic and offline by default. That is a deliberate
product decision, not a placeholder:

* **Extractable** summarisation (TextRank) means every sentence can be clicked
  through to its source. A generated summary cannot be cited, and a summary you
  cannot cite is a summary you cannot trust in a system whose entire claim is
  provenance.
* **Rule-based** prediction means every prediction carries the sentence it came
  from.
* **Linear** importance scoring means "why is this ranked high" has an answer.

`AI_MODE` selects a model-backed implementation where one is configured. The
interfaces do not change, so the escalation path is a configuration change, not
a rewrite - and the deterministic path stays available as the baseline the
model is measured against.
"""

from omnilinker.ai.enrich import (
    assign_thread_subjects,
    derive_subject,
    enrich_documents,
    score_importance,
)
from omnilinker.ai.predict import (
    Prediction,
    PredictionResult,
    predict_deadlines,
    predict_follow_ups,
    predict_resurfacing,
    run_predictions,
)
from omnilinker.ai.summarize import (
    LeadSummarizer,
    Summary,
    TextRankSummarizer,
    get_summarizer,
    split_sentences,
)

__all__ = [
    "TextRankSummarizer",
    "LeadSummarizer",
    "Summary",
    "get_summarizer",
    "split_sentences",
    "Prediction",
    "PredictionResult",
    "run_predictions",
    "predict_deadlines",
    "predict_follow_ups",
    "predict_resurfacing",
    "score_importance",
    "derive_subject",
    "assign_thread_subjects",
    "enrich_documents",
]
