# RQ3 — Susceptibility to Misaligned Definitions

**Research question.** What happens when an LLM is given a task definition that contradicts conventional usage (e.g. a definition labeling "polite criticism" as toxic)? Does the model push back, lose confidence, or simply comply?

**Headline finding.** LLMs **follow misaligned definitions while keeping their confidence levels indistinguishable from the aligned condition.** This is a critical calibration failure: confidence provides no reliable signal for detecting when a model is applying an incorrect definition.

Practical implication for annotation pipelines: you cannot rely on confidence scores to catch poorly-specified task definitions. Definition alignment must be validated upstream — for example, with the DSF metric (see [`../rq1_familiarity/`](../rq1_familiarity)).

## What's in this directory

A minimal reference implementation for the misalignment protocol (definition perturbation + paired-condition evaluation) is planned for a future release. For now this directory describes the experiment and points to the paper for the full methodology.
