"""Definition-Specific Familiarity (DSF) — minimal reference implementation.

DSF measures how closely a model's internal concept of a task aligns with the
dataset's task definition, via cosine similarity of sentence embeddings.

Reference:
    Casanova, Kocielnik, Alvarez. "On the Limits of LLM Adaptability:
    Impact of Model-Internalized Priors on Annotation Task Performance."
    ICML 2026 (Spotlight). https://openreview.net/forum?id=oTv2bKG5Qg

Typical usage:

    from dsf import compute_dsf, consensus_dsf, MODEL_EXPLANATION_PROMPT

    # 1. Ask your LLM to describe its own understanding of the concept.
    prompt = MODEL_EXPLANATION_PROMPT.format(concept_name="toxicity")
    model_explanation = your_llm(prompt, temperature=0.2, max_tokens=200)

    # 2. Score alignment against the dataset's definition.
    dataset_definition = "A rude, disrespectful, or unreasonable comment ..."
    score = compute_dsf(dataset_definition, model_explanation)

The paper's headline partial r = +0.41 uses consensus DSF (mean across six
encoders); see consensus_dsf() below.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_EXPLANATION_PROMPT = (
    "You are documenting your internal understanding of a labeling concept.\n"
    "Concept: {concept_name}\n"
    "Describe, in your own words, what content qualifies as this concept. "
    "Write 2-3 sentences, be concrete.\n"
    "Answer:\n"
)

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# Encoders averaged for the paper's headline consensus DSF metric.
# OpenAI text-embedding-3-small uses a different API and is omitted here;
# add it yourself if you want to reproduce the full six-encoder consensus.
CONSENSUS_ENCODERS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-large-v2",
    "hkunlp/instructor-large",
)


def compute_dsf(
    definition: str,
    model_explanation: str,
    encoder: str = DEFAULT_ENCODER,
) -> float:
    """Cosine similarity between embeddings of two definitions, clipped to [0, 1]."""
    model = SentenceTransformer(encoder)
    emb_a, emb_b = model.encode(
        [definition.strip(), model_explanation.strip()],
        normalize_embeddings=True,
    )
    return float(np.clip(np.dot(emb_a, emb_b), 0.0, 1.0))


def consensus_dsf(
    definition: str,
    model_explanation: str,
    encoders: tuple[str, ...] = CONSENSUS_ENCODERS,
) -> float:
    """Mean DSF across multiple sentence encoders (paper's headline metric)."""
    scores = [compute_dsf(definition, model_explanation, enc) for enc in encoders]
    return float(np.mean(scores))
