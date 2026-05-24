# RQ2 — Decision Stickiness and Rescue Rate

**Research question.** When an LLM gets a zero-shot annotation wrong, can prompting (aligned definitions, few-shot examples, iterative re-asking) actually fix it?

**Headline findings.**

* **Overall rescue rate is 36.4%** — nearly two-thirds of zero-shot errors resist correction by any single prompting strategy we tested.
* **High-confidence errors are the most resistant.** The more confident the initial wrong prediction, the less likely additional prompting is to fix it.
* **Iterating doesn't help much.** A three-turn rescue sequence (few-shot → aligned definition → explicit reconsideration) lifts rescue from 7.5% at Turn 1 to only 18.7% at Turn 3, and high-confidence rescue to just 8.5%.

These results suggest prompting primarily *stabilizes* predictions the model already gets right, rather than serving as a reliable mechanism for error recovery.

## Definitions

**Rescue Rate** — `P(Correct | Prompted, Zero-Shot Wrong)`. Among items the model initially got wrong under zero-shot, the fraction the prompted condition corrects.

## What's in this directory

A minimal reference implementation for computing rescue rate and reproducing the confidence-bucketed analysis is planned for a future release. For now this directory describes the metric and points to the paper for the full methodology.
