# Stage 3 Deterministic Evaluation Report

## Status

The deterministic Stage 3 implementation is complete and validated. It includes
evidence availability, primary failure attribution, paired paper-clustered
bootstrap comparisons, ordered eligibility intersections, and the SciDocs
whole-document BM25 control.

Five of the six frozen QASPER validation runs have complete metric artifacts.
The remaining Qwen3-4B complete-paper run has 127 unique first attempts
persisted. It cannot continue until macOS restarts because the local MLX
processes entered uninterruptible GPU-driver states after a rejected
multi-server throughput experiment.

The held-out QASPER test split has not been inspected.

## Completed QASPER validation results

| Model | Track | Cases | Valid rate | Token F1 | Exact match | Citation F1 | Request errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | Oracle evidence | 853 | 0.9437 | 0.2778 | 0.0035 | 0.9229 | 0 |
| Qwen3-4B | Dense top-5 | 930 | 0.9677 | 0.1852 | 0.0366 | 0.3486 | 0 |
| GPT-5 | Oracle evidence | 853 | 0.8406 | 0.2618 | 0.0234 | 0.9123 | 135 |
| GPT-5 | Dense top-5 | 930 | 0.7129 | 0.1485 | 0.0344 | 0.4470 | 267 |
| GPT-5 | Complete paper | 930 | 0.5452 | 0.1541 | 0.0237 | 0.6220 | 415 |

The GPT-5 API quota was exhausted during the full runs. The persisted request
errors are valid operational reliability outcomes, but they confound answer
quality, citation, abstention, calibration, and cross-model comparisons. These
runs therefore cannot select a model on quality. A clean GPT-5 rerun is required
if GPT-5 remains a candidate.

## Dense retrieved-context diagnosis

The frozen dense retrieval context was identical across models:

- evidence Hit@5: `0.5205`
- best-reference Recall@5: `0.4156`
- best-reference Precision@5: `0.1273`
- best-reference MRR@5: `0.2831`
- best-reference NDCG@5: `0.2922`
- complete-reference-set@5: `0.3388`

For Qwen3-4B, the primary deterministic failure counts were:

- retrieval miss: `547`
- answer failure despite sufficient evidence: `269`
- format failure: `28`
- false answer: `26`
- evidence unavailable for attribution: `17`
- false abstention: `7`
- correct answer: `4`
- citation failure: `3`
- correct abstention: `29`

Retrieval noise was also present as a secondary flag in `289` cases. Retrieval
miss is the dominant failure under the strict rule that at least one complete
human reference evidence set must be present.

The paired dense comparison contains 930 matched cases and 10,000
paper-clustered resamples. GPT-5 minus Qwen3-4B token F1 was `-0.0367`, with a
95 percent interval of `[-0.0547, -0.0185]`. GPT-5 minus Qwen3-4B valid-response
rate was `-0.2548`, with a 95 percent interval of
`[-0.3161, -0.1959]`. These differences are quota-confounded and are not model
selection evidence.

## SciDocs whole-document control

The full-corpus whole-document BM25 diagnostic used 25,657 documents, 1,000
queries, and 4,928 positive judgments:

- Recall@5: `0.10618`
- MRR@5: `0.24757`
- Precision@5: `0.10480`
- NDCG@5: `0.12116`
- p95 query latency: `32.4975 ms`

The best registered chunked primary result remains fixed-chunk dense retrieval,
with Recall@5 `0.15753`, MRR@5 `0.33928`, and p95 latency `4.8559 ms`.

## Release decision

Stage 3 supplies deterministic diagnostic evidence, but numeric promotion
thresholds remain intentionally unset. Thresholds require explicit product
requirements, the completed Qwen3-4B complete-paper baseline, and a clean GPT-5
rerun if GPT-5 is still under consideration. The held-out test remains sealed
until those thresholds are frozen.
