# Stage 3 Deterministic Evaluation Report

## Revised execution order

Stage 3 now selects retrieval before running either generation model:

1. Freeze BM25, dense, and hybrid contexts for the same 930 validation cases.
2. Compare each retrieved top 5 against the oracle evidence ceiling.
3. Select one retrieval setting without using generation metrics.
4. Run Qwen3-4B on the selected setting.
5. Run a GPT-5 pilot, estimate full-run cost, and obtain explicit budget
   approval.
6. Run GPT-5 on the same selected contexts and compare matched cases.

The held-out QASPER test split has not been inspected.

## Retrieval selection

| Retrieval | Complete evidence | Recall@5 | Hit@5 | MRR@5 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid MiniLM plus BM25 RRF | 0.4033 | 0.4774 | 0.5826 | 0.3310 | 0.3396 |
| Dense MiniLM | 0.3388 | 0.4156 | 0.5205 | 0.2831 | 0.2922 |
| BM25 | 0.3118 | 0.3697 | 0.4666 | 0.2587 | 0.2613 |
| Oracle evidence | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9973 |

Hybrid MiniLM plus BM25 reciprocal-rank fusion is selected. All generation runs
after this point must use
`hybrid-minilm-paper-top5-validation-v1.json`.

## Qwen3-4B on selected hybrid retrieval

The full 930-case Qwen3-4B run completed with:

- valid-response rate: `0.9677`
- token F1: `0.1938`
- normalized exact match: `0.0312`
- citation F1: `0.3655`
- answerability accuracy: `0.8419`
- false-abstention rate: `0.0943`
- p95 latency: `4.91 seconds`
- request errors: `0`

The primary deterministic attribution contains 492 retrieval misses and 318
answer failures despite sufficient evidence. The strict retrieval-miss count is
lower than the previous dense run because hybrid retrieval increased complete
evidence availability from `0.3388` to `0.4033`.

## GPT-5 cost preflight

The hybrid GPT-5 pilot supplied 25 exact counted input prompts, but all API
requests failed with `429 insufficient_quota`. The prior dense GPT-5 artifact
supplies 663 successful output-usage records as a provisional proxy. For a
930-case hybrid run, current GPT-5 pricing and these token observations project:

- expected cost: `$4.07`
- cost using observed p95 output tokens: `$6.57`
- cost if every case reaches the 1,024-token ceiling: `$10.74`
- ceiling with one fully billed retry for every case: `$21.48`

Pricing was verified on July 29, 2026. A `$10` budget is approved with a
GPT-specific 768-token output cap and zero retries. The conservative full-run
ceiling is `$8.36`. The revised 25-case pilot also returned only
`429 insufficient_quota`, so a clean pilot is still required after API billing
or credits are restored.

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

## Previous dense retrieved-context diagnosis

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
requirements and matched Qwen3-4B and GPT-5 results on the selected hybrid
retrieval setting. The held-out test remains sealed until those thresholds are
frozen.
