# RQ1 — Familiarity and Definition-Specific Familiarity (DSF)

**Research question.** Does an LLM's familiarity with a task definition predict its annotation performance, and if so, *what kind* of familiarity matters: text-level memorization, or definition-level alignment?

**Headline finding.** After controlling for dataset-level confounds, **Definition-Specific Familiarity (DSF)** is positively associated with annotation accuracy (partial *r* = +0.41, *N* = 54 model-dataset pairs). Three text-memorization metrics — ROUGE-L, BERTScore, and embedding cosine similarity — all fail to show a positive association.

## What's in this directory

* [`dsf.py`](dsf.py) — a minimal reference implementation of DSF (~75 lines).

The full research code (multi-prompt variants, calibration, mixed-effects analysis) is not yet public. This directory ships only the core metric so others can drop it into their own pipelines.

## What DSF does

1. Prompt the model to describe its own understanding of a concept (e.g. "toxicity").
2. Embed both the model's response and the dataset's definition with a sentence encoder.
3. Take the cosine similarity → a score in [0, 1].

The exact prompt used in the paper is exported from `dsf.py` as `MODEL_EXPLANATION_PROMPT`.

## Usage

```bash
pip install sentence-transformers numpy
```

```python
from dsf import compute_dsf, consensus_dsf, MODEL_EXPLANATION_PROMPT

# 1. Ask your LLM to describe its own understanding of the concept.
prompt = MODEL_EXPLANATION_PROMPT.format(concept_name="toxicity")
model_explanation = your_llm(prompt, temperature=0.2, max_tokens=200)

# 2. Score alignment against the dataset's definition.
dataset_definition = (
    "A rude, disrespectful, or unreasonable comment that is likely to make "
    "people leave a discussion."
)

# Fast single-encoder DSF (MiniLM by default).
score = compute_dsf(dataset_definition, model_explanation)

# Paper's headline metric: mean across five encoders.
score = consensus_dsf(dataset_definition, model_explanation)
```

## Notes

* **Consensus DSF.** The paper's reported partial *r* = +0.41 uses the mean across six encoders: MiniLM, MPNet, BGE-large, E5-large, Instructor-large, and OpenAI `text-embedding-3-small`. `consensus_dsf()` here averages the five open ones; add OpenAI yourself if you need the full six.
* **No LLM client bundled.** `dsf.py` does not call any LLM — you supply the model explanation string. This keeps the dependency surface to `sentence-transformers` and `numpy`.
* **What this is not.** This is a single-pair score for one (model, concept). Reproducing the paper's *r* = +0.41 requires running DSF across all (model, dataset) pairs and computing partial correlations against accuracy — not in scope for this minimal release.
