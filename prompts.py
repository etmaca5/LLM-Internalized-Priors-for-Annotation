"""Prompt templates used in the paper.

Reference:
    Casanova, Kocielnik, Alvarez. "On the Limits of LLM Adaptability:
    Impact of Model-Internalized Priors on Annotation Task Performance."
    ICML 2026 (Spotlight). https://openreview.net/forum?id=oTv2bKG5Qg

Every prompt template from the paper is collected here as a Python constant,
with a comment linking it to the section of the paper it appears in. Use
``.format(...)`` to fill in the placeholders. The DSF concept-elicitation
prompt is the one exception: it lives next to its reference implementation
in ``rq1_familiarity/dsf.py`` as ``MODEL_EXPLANATION_PROMPT``.

Placeholder conventions:
    {label}         dataset-specific positive class, e.g. "toxic", "hateful",
                    "offensive", "hate speech"
    {neg_label}     negative class; defaults to "not {label}"
    {definition}    the dataset's aligned definition, or one of the six
                    misaligned definitions (see RQ3 / Table 7 in the paper)
    {message}       the text to be classified
    {examples}      a pre-formatted block of 4 balanced labeled examples,
                    one per stanza:
                        Example N:
                        Message: ...
                        Label: ...

Two implementation details from the experiment code that are not visible in
the templates below:

    1. For the four definition-bearing conditions (aligned, misaligned,
       few-shot + aligned, and DSPy aligned), the experiments prepended a
       short per-dataset context line; the five strings are in
       ``CONTEXT_PREAMBLES`` at the bottom of this file. The other
       conditions (zero-shot, plain few-shot, multi-turn rescue) use the
       templates below unchanged.
    2. The few-shot examples block is built by formatting each labeled
       example with the stanza shown above and joining with blank lines.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Confidence elicitation suffix — Section 3 (Methods), Figure 1.
#
# Appended (joined with a blank line) to every classification prompt below.
# Identical across all models and conditions; only {label} / {neg_label}
# vary by dataset.
# ---------------------------------------------------------------------------

CONFIDENCE_SUFFIX = (
    "Respond in exactly this format:\n"
    "PREDICTION: [{label}/{neg_label}]\n"
    "CONFIDENCE: [0-100]"
)


# ---------------------------------------------------------------------------
# The five manually-specified prompting conditions — Section 4 (Methods)
# and the "Prompts" appendix.
#
# Each base prompt asks the classification question, presents the message,
# and ends with "Answer:". The CONFIDENCE_SUFFIX is then appended.
#
# Note the question casing: it appears as "Is this message ..." when it
# stands alone (zero-shot, few-shot) and as "is this message ..." when it
# follows the "Given the definition: '...'," clause (aligned, misaligned,
# few-shot + aligned).
# ---------------------------------------------------------------------------

ZERO_SHOT = """Is this message {label} or {neg_label}?

Message: {message}

Answer:"""


ALIGNED_DEFINITION = """Given the definition: '{definition}', is this message {label} or {neg_label}?

Message: {message}

Answer:"""


# Same template as ALIGNED_DEFINITION — only the {definition} string changes.
# The paper evaluates six misaligned definitions per dataset: twitter_hate,
# foxnews_hate, olid_offensive, gametox_toxicity, general_toxicity (Perspective
# API), and gaming_toxicity (Table 7).
MISALIGNED_DEFINITION = ALIGNED_DEFINITION


FEW_SHOT = """Is this message {label} or {neg_label}? Here are some examples:

{examples}

Now classify this message:
Message: {message}

Answer:"""


FEW_SHOT_ALIGNED_DEF = """Given the definition: '{definition}', is this message {label} or {neg_label}? Here are some examples:

{examples}

Now classify this message:
Message: {message}

Answer:"""


# ---------------------------------------------------------------------------
# Text-familiarity continuation prompt — "Continuation Prompt" subsection
# of the Prompts appendix.
#
# Used to compute the ROUGE-L, BERTScore, and embedding-cosine memorization
# baselines that DSF is compared against. The {prefix} is the first 40% (by
# word count) of a held-out text; the model's continuation is scored against
# the remaining 60%. Decoding is greedy (temperature 0.0).
# ---------------------------------------------------------------------------

CONTINUATION = "Continue the following passage faithfully: {prefix}"


# ---------------------------------------------------------------------------
# Multi-turn rescue prompts — "Multi-Turn Rescue Experiment" appendix.
#
# The supplementary three-turn experiment that tests whether iterative,
# history-aware prompting can break decision stickiness. Each later turn
# shows the model its own prior answers and confidences. Across all six
# open-weights models this lifts rescue from 7.5% at Turn 1 to 18.7% at
# Turn 3, and high-confidence rescue to just 8.5%.
#
# Placeholders:
#     {t0_pred}, {t0_conf}   zero-shot prediction and confidence
#     {t1_pred}, {t1_conf}   Turn 1 prediction and confidence
#     {t2_pred}, {t2_conf}   Turn 2 prediction and confidence
# The CONFIDENCE_SUFFIX is appended at each turn just as in the single-turn
# conditions.
# ---------------------------------------------------------------------------

RESCUE_TURN_1 = """You previously answered this classification task.

Message: {message}
Your previous answer: {t0_pred}
Your previous confidence: {t0_conf}

Here are labeled examples from this task:
{examples}

Now answer the same task again for the message above."""


RESCUE_TURN_2 = """You previously answered this classification task.

Message: {message}
Previous answers so far:
- Turn 0: {t0_pred} ({t0_conf})
- Turn 1: {t1_pred} ({t1_conf})

Task definition:
{definition}

Here are labeled examples from this task:
{examples}

Using the definition and examples above, answer the same task again."""


RESCUE_TURN_3 = """You previously answered this classification task multiple times.

Message: {message}
Previous answers so far:
- Turn 0: {t0_pred} ({t0_conf})
- Turn 1: {t1_pred} ({t1_conf})
- Turn 2: {t2_pred} ({t2_conf})

Task definition:
{definition}

Here are labeled examples from this task:
{examples}

Final check: are you sure your current answer is correct under this definition?
If needed, revise it one last time."""


# ---------------------------------------------------------------------------
# Per-dataset context preambles.
#
# Prepended (with a trailing blank line) to the four definition-bearing
# conditions only: ALIGNED_DEFINITION, MISALIGNED_DEFINITION,
# FEW_SHOT_ALIGNED_DEF, and the DSPy aligned variant. Not used for
# zero-shot, plain few-shot, the continuation prompt, or any rescue turn.
# ---------------------------------------------------------------------------

CONTEXT_PREAMBLES = {
    "jigsaw":  "The following comment is from an online discussion platform such as Wikipedia talk pages or a public comment section.",
    "gametox": "The following message is from online gaming chat.",
    "foxnews": "The following comment is from a Fox News article comment section.",
    "twitter": "The following message is from Twitter (now X).",
    "olid":    "The following message is from Twitter.",
}
