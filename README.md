# Local RAG Retrieval Eval

A small evaluation harness inspired by [BEIR](https://github.com/beir-cellar/beir).
It downloads a standard BEIR dataset, runs retrieval, compares ranked documents
with qrels, and records reproducible experiment results.

The initial scope is retrieval evaluation. It deliberately separates retrieval
quality from LLM answer quality so you can diagnose the first stage before adding
generation and judge-based metrics.

## What it measures

- Precision@K
- Recall@K
- MRR@K
- nDCG@K with graded relevance
- Total retrieval latency and latency per query

Supported retrieval methods:

- `bm25`: dependency-free lexical baseline
- `dense`: exact cosine search with a sentence-transformers model
- `hybrid`: BM25 plus dense search using reciprocal rank fusion

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Small, fast BM25 learning run
rag-eval \
  --dataset scifact \
  --retriever bm25 \
  --max-documents 2000 \
  --max-queries 100 \
  --k 1,3,5,10
```

The first run downloads the public BEIR archive into `data/`. Reports are written
to `results/<experiment-id>.json`, and each run is appended to
`results/experiments.csv`.

To compare dense and hybrid retrieval:

```bash
pip install -e ".[dense]"

rag-eval --dataset scifact --retriever dense --max-documents 2000 --max-queries 100
rag-eval --dataset scifact --retriever hybrid --max-documents 2000 --max-queries 100
```

The default dense model is `sentence-transformers/all-MiniLM-L6-v2`. Override it
with `--model`.

## SciDocs chunking benchmark

Install the benchmark dependencies and download SciDocs:

```bash
pip install -e ".[benchmark]"
rag-benchmark --download-only
```

Run a small smoke experiment before the full matrix:

```bash
rag-benchmark --max-documents 2000 --max-queries 25 --repetitions 1 --k 1,3,5,10
```

Run the reportable fixed-versus-recursive matrix on the full corpus:

```bash
rag-benchmark
```

The runner builds fixed and recursive LangChain chunks, evaluates BM25, dense,
and hybrid RRF retrieval, collapses chunk scores to SciDocs parent documents, and
writes chunk manifests, rankings, per-run metrics, timing, and a CSV summary under
`results/scidocs/<session-id>/`.

## QASPER generation benchmark

Install the generation dependency and prepare a small validation smoke sample:

```bash
pip install -e ".[generation]"
rag-generation prepare \
  --split validation \
  --limit-papers 5 \
  --output-dir data/generation/qasper-v1
```

Remove `--limit-papers` to normalize the complete validation split. The command
creates one case per QASPER question, preserves all reference annotations, maps
annotated evidence to stable passage IDs, and writes a checksummed manifest.
Downloaded data and normalized cases remain under the ignored `data/` directory.
The pinned validation Parquet has SHA-256
`089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222`.
The normalized 1,005-case validation file has SHA-256
`e0172f79d2b17435b5c8c0aaa1ce9db76de0f6619772979a85f5a8c926f38c93`.
The frozen machine-readable validation policy is in
`generation_protocol_v1.json`.

To use the OpenAI API provider, install its optional dependencies:

```bash
pip install -e ".[generation,openai]"
cp .env.example .env
```

Set the key only in the ignored `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key
```

Do not add the key to commands, source files, logs, or committed configuration.

To use OpenRouter, install its optional dependencies and set its key in the same
ignored `.env` file:

```bash
pip install -e ".[generation,openrouter]"
```

```dotenv
OPENROUTER_API_KEY=your-openrouter-api-key
```

OpenRouter accepts a free-form model slug through `--openrouter-model`. The
benchmark requests strict JSON-schema output and requires OpenRouter to select an
endpoint that supports it. This preserves the response contract when models are
changed. By default, Luna Pro falls back to Qwen3.7 Plus and then DeepSeek V4
Flash:

```bash
rag-generation generate-retrieved \
  --provider openrouter \
  --openrouter-model openai/gpt-5.6-luna-pro \
  --openrouter-fallback-model qwen/qwen3.7-plus \
  --openrouter-fallback-model deepseek/deepseek-v4-flash \
  --env-file .env \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --context-manifest data/generation/qasper-v1/retrieval/hybrid-minilm-paper-top5-validation-v1.json \
  --output-file results/generation/qasper-v1/predictions/gpt-5.6-luna-pro-openrouter.jsonl \
  --max-cases 25
```

The two fallback flags above are shown for clarity and are already the defaults.
Use `--no-openrouter-fallbacks` to run only the primary model. OpenRouter records
the model that ultimately served the response, and the runner writes it as
`resolved_model_id` alongside the configured fallback chain.

The optional `--openrouter-http-referer` and `--openrouter-app-title` flags set
OpenRouter attribution headers. The app title defaults to `Project Local RAG`.
OpenRouter token counting before a request uses a stable `o200k_base` estimate
because model slugs can span tokenizer families. Each completed result records
the authoritative prompt and completion token counts returned by OpenRouter.

The generation experiment runner has three diagnostic tracks:

- `oracle-evidence` is the primary controlled generator benchmark.
- `complete-paper` is the long-context answerability and calibration diagnostic.
- `retrieved-context` consumes a frozen top-K retrieval manifest, so retrieval is
  never rerun during generation and every model receives identical context.

Execution walkthrough: [generation smoke-test execution flow](generation_smoke_test_execution.md).

For the Apple Silicon baseline, serve the 4-bit MLX model in one terminal:

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit
```

The server listens on `http://127.0.0.1:8080` by default. It is intended only as
a local benchmark endpoint.

The same generation commands accept `--provider openai`. The OpenAI provider uses
the Responses API, structured JSON output, `store=false`, and the API key loaded
from `.env`. For an oracle-evidence run:

```bash
rag-generation generate-oracle \
  --provider openai \
  --openai-model gpt-5 \
  --env-file .env \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --output-file results/generation/qasper-v1/predictions/gpt-5-oracle-v3.jsonl \
  --max-cases 25 \
  --max-output-tokens 1024
```

`gpt-5` is the default OpenAI model. The allowed model IDs are:

- `gpt-5`
- `gpt-5.6-sol`
- `gpt-5.6-luna`

Evaluate that run with:

```bash
rag-generation metrics \
  --track oracle-evidence \
  --model gpt-5 \
  --predictions-file results/generation/qasper-v1/predictions/gpt-5-oracle-v3.jsonl \
  --output-dir results/generation/qasper-v1/metrics/gpt-5-oracle-v3
```

### Oracle generator comparison

The 25-case QASPER oracle runs produced the following aggregate metrics:

| Metric | Qwen3 4B local | GPT-5 API | GPT-5 minus Qwen3 |
| --- | ---: | ---: | ---: |
| Valid response rate | 1.0000 | 1.0000 | 0.0000 |
| Normalized exact match | 0.0000 | 0.0400 | +0.0400 |
| Answer token F1 | 0.3150 | 0.3733 | +0.0583 |
| Citation precision | 0.8659 | 0.9058 | +0.0399 |
| Citation recall | 0.9167 | 0.8551 | -0.0616 |
| Citation F1 | 0.8717 | 0.8467 | -0.0251 |
| Mean latency, seconds | 2.9146 | 6.7700 | +3.8554 |
| Mean total tokens | 922.52 | 1226.52 | +304.00 |

GPT-5 had higher answer token F1 on 15 matched cases, Qwen3 on 8, with 2
ties. The paper-clustered 95% intervals were `[0.2142, 0.4157]` for Qwen3
answer token F1 and `[0.2644, 0.4793]` for GPT-5. Citation F1 intervals were
`[0.8160, 0.9427]` for Qwen3 and `[0.7809, 0.9214]` for GPT-5. These are
individual-run intervals, not a paired interval for the difference, so the
comparison does not establish statistical significance.

Interpret the efficiency comparison cautiously. Local and API latency depend on
different hardware and network paths, and provider token counts use different
tokenizers. The evaluator also does not currently calculate OpenAI API cost, so
its zero-cost field must not be interpreted as an actual API cost.

This is not a strictly controlled model-only comparison. The Qwen3 artifact uses
prompt `qasper-generation-v2`, while GPT-5 uses the citation-format correction
in `qasper-generation-v3`. Both runs use the same 25 ordered case IDs and oracle
evidence track.

### Freeze retrieval separately from generation

`freeze-context` is the only fixed-context command that performs retrieval.
Choose the scope and method there, then give the resulting immutable manifest to
one or more `generate-retrieved` runs.

Paper-scoped BM25, the default:

```bash
rag-generation freeze-context \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --eligibility-file results/generation/qasper-v1/predictions/qwen3-4b-track-b-v2.eligibility.json \
  --retrieval-scope paper \
  --retriever bm25 \
  --top-k 5 \
  --output-file data/generation/qasper-v1/retrieval/bm25-paper-top5.json
```

Corpus-scoped BM25:

```bash
rag-generation freeze-context \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --eligibility-file results/generation/qasper-v1/predictions/qwen3-4b-track-b-v2.eligibility.json \
  --retrieval-scope corpus \
  --retriever bm25 \
  --top-k 5 \
  --output-file data/generation/qasper-v1/retrieval/bm25-corpus-top5.json
```

Dense model 1, the faster MiniLM baseline:

```bash
pip install -e ".[generation,dense]"

rag-generation freeze-context \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --eligibility-file results/generation/qasper-v1/predictions/qwen3-4b-track-b-v2.eligibility.json \
  --retrieval-scope paper \
  --retriever dense \
  --dense-model sentence-transformers/all-MiniLM-L6-v2 \
  --top-k 5 \
  --output-file data/generation/qasper-v1/retrieval/dense-minilm-paper-top5.json
```

Dense model 2, the larger MPNet baseline:

```bash
rag-generation freeze-context \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --eligibility-file results/generation/qasper-v1/predictions/qwen3-4b-track-b-v2.eligibility.json \
  --retrieval-scope paper \
  --retriever dense \
  --dense-model sentence-transformers/all-mpnet-base-v2 \
  --top-k 5 \
  --output-file data/generation/qasper-v1/retrieval/dense-mpnet-paper-top5.json
```

Hybrid BM25 plus dense retrieval with reciprocal-rank fusion:

```bash
rag-generation freeze-context \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --eligibility-file results/generation/qasper-v1/predictions/qwen3-4b-track-b-v2.eligibility.json \
  --retrieval-scope paper \
  --retriever hybrid \
  --dense-model sentence-transformers/all-MiniLM-L6-v2 \
  --hybrid-candidate-k 100 \
  --hybrid-rrf-k 60 \
  --top-k 5 \
  --output-file data/generation/qasper-v1/retrieval/hybrid-minilm-paper-top5.json
```

The manifest records schema version 2, retrieval scope, method, dense model,
batch size, BM25 parameters, hybrid parameters, source checksums, ranked passage
IDs, and scores. Existing schema-version-1 manifests remain readable.

Generation never accepts a retrieval method or scope. Replay any frozen manifest
by changing only `--context-manifest` and the prediction filename:

```bash
rag-generation generate-retrieved \
  --provider local \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --context-manifest data/generation/qasper-v1/retrieval/dense-minilm-paper-top5.json \
  --output-file results/generation/qasper-v1/predictions/qwen3-4b-dense-minilm-paper-top5-v3.jsonl
```

This boundary lets different generators consume byte-identical retrieved
contexts and lets one generator compare BM25, either dense model, and hybrid
retrieval without retrieval running during generation.

### Legacy corpus-scoped retrieved-context generator comparison

Both generators were run with prompt `qasper-generation-v3`, a 1024-token output
limit, and the same schema-version-1 corpus-scoped BM25 top-5 manifest:

```bash
rag-generation generate-retrieved \
  --provider local \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --context-manifest data/generation/qasper-v1/retrieval/bm25-top5-track-b-v2.json \
  --output-file results/generation/qasper-v1/predictions/qwen3-4b-retrieved-bm25-top5-v3.jsonl \
  --max-output-tokens 1024 \
  --no-resume

rag-generation generate-retrieved \
  --provider openai \
  --env-file .env \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --context-manifest data/generation/qasper-v1/retrieval/bm25-top5-track-b-v2.json \
  --output-file results/generation/qasper-v1/predictions/gpt-5-retrieved-bm25-top5-v3.jsonl \
  --max-output-tokens 1024 \
  --no-resume
```

Qwen3 initially completed 24 cases and produced one invalid-schema response that
omitted the required `abstain` key. A historical diagnostic retry reproduced the
same deterministic error. Current resume behavior skips every persisted attempt,
including invalid responses, so an interrupted primary run cannot silently
become a format-retry experiment. GPT-5 completed all 25 cases without an error.

Generate the metrics with:

```bash
rag-generation metrics \
  --track retrieved-context \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --predictions-file results/generation/qasper-v1/predictions/qwen3-4b-retrieved-bm25-top5-v3.jsonl \
  --output-dir results/generation/qasper-v1/metrics/qwen3-4b-retrieved-bm25-top5-v3

rag-generation metrics \
  --track retrieved-context \
  --model gpt-5 \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --predictions-file results/generation/qasper-v1/predictions/gpt-5-retrieved-bm25-top5-v3.jsonl \
  --output-dir results/generation/qasper-v1/metrics/gpt-5-retrieved-bm25-top5-v3
```

| Metric | Qwen3 4B local | GPT-5 API | GPT-5 minus Qwen3 |
| --- | ---: | ---: | ---: |
| Valid response rate | 0.9600 | 1.0000 | +0.0400 |
| Answerability accuracy | 0.8800 | 0.6800 | -0.2000 |
| False abstention rate | 0.0800 | 0.3200 | +0.2400 |
| Normalized exact match | 0.0000 | 0.0000 | 0.0000 |
| Answer token F1 | 0.1218 | 0.0858 | -0.0360 |
| Citation precision | 0.1023 | 0.1588 | +0.0566 |
| Citation recall | 0.1250 | 0.1618 | +0.0368 |
| Citation F1 | 0.1023 | 0.1503 | +0.0481 |
| Expected calibration error | 0.7085 | 0.4202 | -0.2883 |
| Area under risk-coverage curve | 0.8505 | 0.8733 | +0.0228 |
| Mean latency, seconds | 2.8957 | 3.9132 | +1.0175 |
| Mean total tokens | 1041.12 | 1278.24 | +237.12 |

Lower is better for false abstention rate, expected calibration error, and area
under the risk-coverage curve. Qwen3 had higher answer token F1 on 12 matched
cases, GPT-5 on 6, with 7 ties. Paper-clustered 95% intervals for answer token F1
were `[0.0698, 0.1791]` for Qwen3 and `[0.0336, 0.1632]` for GPT-5. Citation F1
intervals were `[0.0000, 0.2269]` and `[0.0000, 0.3389]`, respectively. These are
individual-run intervals, not paired intervals for the model differences.

All 25 source cases are labeled answerable. GPT-5 abstained on 8, while Qwen3
abstained on 2 and had one invalid response. The low answer and citation scores
show that the frozen BM25 contexts often do not contain sufficient reference
evidence. As with the oracle comparison, provider token counts and latency are
not directly comparable, and the evaluator does not calculate OpenAI API cost.

### Three generation tasks

Use `generate-oracle` to measure generation with gold evidence and no retrieval
failure:

```bash
rag-generation generate-oracle \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --output-file results/generation/qasper-v1/predictions/qwen3-4b-oracle-validation-v1.jsonl \
  --all-cases \
  --max-context-tokens 32768 \
  --max-output-tokens 1024 \
  --no-resume
```

Omit `--all-cases` for the default 25-case smoke run.
Use a new run ID for the full-validation output. Do not reuse a 25-case smoke
sidecar because its frozen `max_cases` value is intentionally incompatible.

Use `generate-retrieved` to replay a previously frozen context manifest. This
task does not invoke retrieval, so different generators receive identical
passages:

```bash
rag-generation generate-retrieved \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --context-manifest data/generation/qasper-v1/retrieval/bm25-top5-track-b-v2.json \
  --output-file results/generation/qasper-v1/predictions/qwen3-4b-retrieved-bm25-top5-v1.jsonl \
  --max-context-tokens 32768 \
  --max-output-tokens 1024
```

Use `generate-end-to-end` as a convenience wrapper around configurable retrieval
and generation.
The retrieval result is frozen before the first model call, so the complete run
can be replayed later with `generate-retrieved`:

```bash
rag-generation generate-end-to-end \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --eligibility-file results/generation/qasper-v1/predictions/qwen3-4b-track-b-v2.eligibility.json \
  --retrieval-scope paper \
  --retriever bm25 \
  --top-k 5 \
  --context-manifest data/generation/qasper-v1/retrieval/end-to-end-bm25-top5-v1.json \
  --output-file results/generation/qasper-v1/predictions/qwen3-4b-end-to-end-bm25-top5-v1.jsonl \
  --max-context-tokens 32768 \
  --max-output-tokens 1024
```

Prompt contract `qasper-generation-v3` declares exact JSON field types, requires
numeric confidence, repeats the contract after long contexts, and limits answer
length and citation count. The runner calls the server's OpenAI-compatible chat
endpoint sequentially,
counts tokens with the original Qwen tokenizer, rejects prompts that exceed the
shared context limit, validates the strict JSON response, and appends predictions
to `results/generation/qasper-v1/predictions/qwen3-4b-track-a-v2.jsonl`.
Repeating the command resumes from every persisted case, track, model, and
prompt-version attempt, regardless of response validity. It also writes
`qwen3-4b-track-a-v2.eligibility.json`,
which freezes the eligible case IDs used as the metric denominator. Use
`--no-resume` to overwrite the output.

Use explicit, track-specific output paths when running a comparison. The prompt-v2
25-case baseline uses `qwen3-4b-track-a-v2.jsonl` for Track A and
`qwen3-4b-track-b-v2.jsonl` for Track B. The observed local runs used identical
eligible case IDs. Track A produced 25 valid responses, while Track B improved
from 11 valid responses with prompt v1 to 16 with prompt v2.

Validate the matched response-status change with:

```bash
rag-generation compare-responses \
  --baseline-predictions-file results/generation/qasper-v1/predictions/qwen3-4b-track-b-v1.jsonl \
  --baseline-eligibility-file results/generation/qasper-v1/predictions/qwen3-4b-track-b-v1.eligibility.json \
  --candidate-predictions-file results/generation/qasper-v1/predictions/qwen3-4b-track-b-v2.jsonl \
  --candidate-eligibility-file results/generation/qasper-v1/predictions/qwen3-4b-track-b-v2.eligibility.json \
  --output-file results/generation/qasper-v1/comparisons/track-b-prompt-v1-v2.json
```

The lower-level equivalent of the retrieved workflow remains available:

```bash
rag-generation freeze-context \
  --eligibility-file results/generation/qasper-v1/predictions/qwen3-4b-track-b-v2.eligibility.json \
  --retrieval-scope paper \
  --retriever bm25 \
  --output-file data/generation/qasper-v1/retrieval/bm25-paper-top5.json \
  --top-k 5

rag-generation generate-retrieved \
  --context-manifest data/generation/qasper-v1/retrieval/bm25-paper-top5.json \
  --output-file results/generation/qasper-v1/predictions/qwen3-4b-retrieved-bm25-top5-v1.jsonl
```

After the generation run, calculate Stage 0 to 6 metrics:

```bash
rag-generation metrics
```

The command joins cases and predictions by `case_id`, scores every eligible case,
and writes:

```text
results/generation/qasper-v1/metrics/qwen3-4b-track-a-v2/
  evaluation_records.jsonl
  per_case_metrics.jsonl
  summary.json
```

The summary also reports deterministic evidence availability and a primary
failure attribution for every case. Retrieved-context evidence metrics include
Hit@K, best-reference Recall@K, Precision@K, MRR@K, NDCG@K, and complete
reference-evidence-set@K. The primary attribution distinguishes retrieval miss,
false abstention, answer failure despite sufficient evidence, citation failure,
format failure, request failure, correct answer, and correct abstention.
Retrieval noise is reported as a secondary flag because sufficient gold evidence
can coexist with distractors. Cases without resolvable reference evidence are
marked unavailable for attribution instead of being assigned to retrieval or
generation.

Select retrieval before generation by comparing frozen candidates directly with
oracle evidence:

```bash
rag-generation compare-retrieval \
  --context-manifest data/generation/qasper-v1/retrieval/bm25-paper-top5-validation-v1.json \
  --context-manifest data/generation/qasper-v1/retrieval/dense-minilm-paper-top5-validation-v1.json \
  --context-manifest data/generation/qasper-v1/retrieval/hybrid-minilm-paper-top5-validation-v1.json \
  --output-file results/generation/qasper-v1/comparisons/retrieval-vs-oracle-validation-v1.json
```

The command requires identical ordered case IDs and ranks candidates without
using generator output. The frozen selection order is complete evidence set,
best-reference recall, hit rate, NDCG, MRR, then precision.

Before a full GPT API run, project cost from an observed pilot and optionally
enforce a hard budget:

```bash
rag-generation estimate-cost \
  --predictions-file results/generation/qasper-v1/predictions/gpt-5-hybrid-pilot.jsonl \
  --output-usage-file results/generation/qasper-v1/predictions/gpt-5-comparable-pilot.jsonl \
  --model gpt-5 \
  --target-case-count 930 \
  --max-output-tokens 768 \
  --retries 0 \
  --budget-usd 10 \
  --budget-basis ceiling_with_retries \
  --output-file results/generation/qasper-v1/comparisons/gpt-5-hybrid-cost.json
```

The estimator makes no API calls. It reports expected, observed-p95-output,
ceiling, and ceiling-with-retries scenarios using a dated model price table. A
budget is required by the Stage 3 protocol before full GPT generation.

Compare two evaluated systems on identical ordered cases with paired
paper-clustered bootstrap differences:

```bash
rag-generation compare-metrics \
  --track oracle-evidence \
  --baseline-per-case-file results/generation/qasper-v1/metrics/qwen3-4b-oracle-validation-v1/per_case_metrics.jsonl \
  --candidate-per-case-file results/generation/qasper-v1/metrics/gpt-5-oracle-validation-v1/per_case_metrics.jsonl \
  --baseline-label qwen3-4b \
  --candidate-label gpt-5 \
  --output-file results/generation/qasper-v1/comparisons/oracle-validation-v1.json
```

The comparison uses 10,000 paired resamples of whole `paper_id` clusters,
reports a 95 percent interval for every candidate-minus-baseline difference,
and reports candidate win probability. It fails closed if case order differs.

Use `intersect-eligibility` before freezing a shared retrieval manifest when
generator context windows produce different eligible subsets:

```bash
rag-generation intersect-eligibility \
  --eligibility-file results/generation/qasper-v1/predictions/qwen.eligibility.json \
  --eligibility-file results/generation/qasper-v1/predictions/gpt.eligibility.json \
  --output-file results/generation/qasper-v1/predictions/common.eligibility.json
```

The summary includes response-status rates, retries, latency, token usage, local
inference cost, official QASPER token F1, normalized exact match, citation
precision, citation recall, citation F1, citation validity, and citation coverage.
Missing and invalid predictions remain visible in the denominator. Complete-paper
and retrieved-context runs also report answerability accuracy, abstention
precision, recall, F1, false
answers, false abstentions, no decisions, a confusion matrix, confidence
availability, expected calibration error, a risk-coverage curve, and its area.
All tracks include deterministic 95 percent percentile intervals from 10,000
paper-clustered bootstrap resamples.

## Blinded semantic evaluation

The semantic judge supplements the deterministic metrics. It extracts atomic
material claims, labels context support as `supported`, `contradicted`, or
`not_in_context`, checks whether each associated anonymous citation entails its
claim, and assigns semantic-correctness and rubric-completeness scores from 0 to
4. QASPER does not provide atomic required-fact annotations, so completeness is
not reported as exact required-fact recall.

Prepare judge inputs separately from judge execution. The prepared JSONL contains
only the question, supplied context, references, candidate answer, and anonymous
context IDs in its `judge_input` field. Generator identity and source checksums
are kept in the sidecar manifest and are never rendered into the judge prompt.

```bash
rag-semantic-judge prepare \
  --cases-file data/generation/qasper-v1/validation.cases.jsonl \
  --predictions-file results/generation/qasper-v1/predictions/gpt-5.6-luna-pro-hybrid-minilm-paper-top5-validation-v1.jsonl \
  --output-file results/generation/qasper-v1/semantic/luna-pro-hybrid-validation-v1.inputs.jsonl \
  --track retrieved-context \
  --generator-model openai/gpt-5.6-luna-pro
```

Run a one-case judge smoke test before any larger paid evaluation:

```bash
rag-semantic-judge run \
  --provider openrouter \
  --env-file .env \
  --inputs-file results/generation/qasper-v1/semantic/luna-pro-hybrid-validation-v1.inputs.jsonl \
  --output-file results/generation/qasper-v1/semantic/luna-pro-hybrid-validation-v1.judgments.jsonl \
  --judge-model anthropic/claude-sonnet-4.5 \
  --max-cases 1 \
  --retries 0
```

The runner rejects a judge model, including a configured fallback, when it
matches the generator. It records request failures separately and validates
every response against a strict JSON schema. No paid judge call is made by
preparation or summarization.

```bash
rag-semantic-judge summarize \
  --judgments-file results/generation/qasper-v1/semantic/luna-pro-hybrid-validation-v1.judgments.jsonl \
  --output-dir results/generation/qasper-v1/semantic/luna-pro-hybrid-validation-v1
```

The summary reports claim-weighted and case-macro support, fully faithful cases,
citation entailment and citation completeness, mean rubric scores, score
distributions, and evaluator failures. This custom harness is the reporting
source of truth. Ragas is not used in this implementation because the frozen
protocol requires explicit three-way claim labels and passage-level anonymous
citation judgments that its standard metrics do not directly preserve.

## Evaluation design


BEIR datasets have three important pieces:

```text
corpus.jsonl        documents to search
queries.jsonl       evaluation queries
qrels/<split>.tsv   graded query-document relevance judgments
```

When limits are supplied, the sampler selects queries first, retains every
positively judged document for those queries, and fills the remaining document
budget deterministically. This is important. Independently sampling documents
would remove ground-truth positives and make retrieval scores misleading.

Each run records:

- Dataset, split, retriever, model, cutoffs, sample limits, and random seed
- Document, query, and positive-judgment counts
- All retrieval metrics
- Timing and Python version

## Commands

```text
rag-eval [-h]
  [--dataset DATASET]
  [--split SPLIT]
  [--data-dir DATA_DIR]
  [--output-dir OUTPUT_DIR]
  [--retriever {bm25,dense,hybrid}]
  [--model MODEL]
  [--batch-size BATCH_SIZE]
  [--k 1,3,5,10]
  [--max-documents N]
  [--max-queries N]
  [--seed SEED]
```

Use `--max-documents` only as a learning shortcut. Scores from a sampled corpus
are not directly comparable with published full-corpus BEIR results.

## Test

```bash
pip install -e ".[dev]"
pytest
```

## Benchmark project plan

The benchmark should answer three different questions independently:

1. **Retrieval:** Did the system find and rank the necessary evidence?
2. **Generation:** Given sufficient evidence, did the model produce a correct,
   complete, and grounded answer?
3. **End to end:** Does the full pipeline produce a useful answer within production
   latency and cost constraints?

Keep these tracks separate during diagnosis. A poor final answer can come from
missing evidence, poor context construction, or generation failure. A single
end-to-end score cannot identify which component failed.

### Phase 0. Define the benchmark contract

- [ ] Write down the target use cases, users, corpus types, languages, and expected
  question distribution.
- [ ] Define what counts as a correct answer, a valid citation, and an acceptable
  abstention for the product.
- [ ] Choose quality, latency, cost, and storage constraints before comparing
  systems. Include target values and hard limits.
- [ ] Select primary metrics. Use `Recall@K` as the retrieval gate, `NDCG@K` or
  `MRR@K` for ranking, and grounded answer correctness as the primary end-to-end
  outcome.
- [ ] Define a fixed evaluation protocol, including dataset version, split, random
  seed, K values, hardware, dependency versions, and the number of repeated runs.
- [ ] Create a run manifest schema so every result can be reproduced from its
  dataset, configuration, code commit, prompt version, and model version.

**Deliverable:** `benchmark_spec.md` with target workloads, metrics, constraints,
and promotion thresholds.

### Phase 1. Build a representative evaluation dataset

- [ ] Inventory the production corpus and remove secrets, personal information,
  duplicates, and documents that must not enter the benchmark.
- [ ] Create representative queries across common, difficult, ambiguous,
  unanswerable, multi-hop, and freshness-sensitive cases.
- [ ] Split queries by stable IDs into development and held-out test sets. Do not
  tune on the test set.
- [ ] Label every query with its supporting passage IDs and graded relevance
  judgments. Record all valid supporting passages when more than one answer path
  exists.
- [ ] Add a reference answer, answerability label, required facts, and expected
  citations for generation evaluation.
- [ ] Have a second annotator review a sample. Resolve disagreements and document
  the labeling rubric.
- [ ] Version the corpus, queries, qrels, reference answers, and rubric together.
- [ ] Add validation that rejects missing IDs, orphaned qrels, train/test overlap,
  empty references, and answerable queries with no positive passage.

Suggested record:

```json
{
  "query_id": "q-001",
  "question": "What evidence supports the claim?",
  "answerable": true,
  "relevant_passages": {"doc-17#p3": 2, "doc-42#p1": 1},
  "reference_answer": "The claim is supported by ...",
  "required_facts": ["fact A", "fact B"],
  "expected_citations": ["doc-17#p3"],
  "slice": ["multi-hop", "difficult"]
}
```

**Deliverable:** a versioned, validated benchmark dataset with a frozen held-out
test split.

### Phase 2. Establish retrieval baselines

Detailed execution contract: [SciDocs Retrieval Benchmark Specification](benchmark_spec.md).

- [x] Implement BM25, dense, and hybrid retrieval.
- [x] Implement Precision@K, Recall@K, MRR@K, and NDCG@K.
- [x] Record total retrieval time, time per query, configuration, sample size, and
  Python version.
- [x] Run BM25 on a fixed development sample and save it as the lexical baseline.
- [x] Run dense and hybrid retrieval on the identical queries, corpus, K values,
  and seed.
- [x] Add warm-up runs and report p50, p95, and p99 latency separately from index
  construction time.
- [ ] Record embedding cost, query cost, peak memory, and on-disk index size.
- [x] Save ranked passage IDs and scores for every query, not only aggregate
  metrics.
- [ ] Report bootstrap confidence intervals and per-slice results in addition to
  macro averages.
- [ ] Inspect false negatives from the worst queries and classify causes such as
  vocabulary mismatch, bad chunk boundaries, metadata filtering, multi-hop
  evidence, or stale content.
- [x] Repeat the selected configuration on the full corpus before reporting final
  retrieval results.

**Exit gate:** the selected retriever meets the agreed `Recall@K`, tail-latency,
cost, memory, and index-size thresholds on the held-out set.

### Phase 3. Evaluate generation with fixed context

Detailed execution contract: [Phase 3 fixed-context generation specification](benchmark_spec.md#16-phase-3-purpose-and-diagnostic-boundary).

- [x] Use QASPER questions, answers, evidence, and answerability annotations.
- [x] Normalize QASPER into checksummed, versioned generation cases.
- [x] Complete a generation-only runner that bypasses retrieval and supplies the
  same gold passages to every model.
- [x] Define and implement the structured prompt and response contract.
- [x] Freeze context eligibility, decoding settings,
  model version, and output schema.
- [x] Pin an immutable QASPER Parquet revision.
- [x] Download and normalize the complete QASPER validation split.
- [ ] Run both oracle-evidence and complete-paper fixed-context tracks.
- [x] Require a structured response containing the answer, citations, and an
  abstention indicator.
- [x] Measure answer correctness against QASPER reference answers.
- [x] Measure faithfulness by checking that each material claim is supported by
  the supplied context.
- [x] Measure rubric completeness against the references and context. QASPER does
  not contain atomic required-fact annotations.
- [x] Measure citation precision and recall against annotated evidence sets.
- [x] Measure whether each citation supports its associated generated claim.
- [ ] Test appropriate abstention using both answerable and unanswerable
  questions. Report false-answer and false-abstention rates separately.
- [ ] Combine deterministic checks with a rubric-based evaluator. Blinding is
  implemented. Candidate-order randomization remains pending if pairwise judging
  is added.
- [ ] Manually audit a stratified sample and measure agreement between human and
  automated judgments.
- [x] Record input tokens, output tokens, cost, p50/p95/p99 latency, errors, and
  retries.

**Exit gate:** with gold context, the generator meets correctness, faithfulness,
completeness, citation, abstention, latency, and cost thresholds. Failures here are
generation failures, not retrieval failures.

### Phase 4. Run the end-to-end benchmark

- [ ] Connect ingestion, chunking, indexing, retrieval, optional reranking,
  context construction, generation, citations, and abstention in one runner.
- [ ] Freeze the corpus snapshot and execute the held-out test set without tuning.
- [ ] Measure final-answer correctness, faithfulness, completeness, citation
  quality, answer relevance, and abstention behavior.
- [ ] Record stage-level and total p50/p95/p99 latency, token usage, monetary cost,
  failure rate, and throughput.
- [ ] Store a trace for every query containing retrieved passages, retrieval
  scores, reranked order, constructed context, model output, citations, metrics,
  and timing.
- [ ] Produce per-slice results so regressions are not hidden by an overall
  average.
- [ ] Classify each failure as dataset, ingestion, retrieval, reranking, context,
  generation, citation, or infrastructure related.
- [ ] Compare end-to-end results with the fixed-context results to estimate the
  quality lost before generation.

**Exit gate:** the complete pipeline meets product quality and production
constraints on the frozen test set.

### Phase 5. Run controlled experiments

- [ ] Create a baseline configuration and change exactly one independent variable
  per experiment.
- [ ] When comparing embedding models, hold the corpus, chunking, indexing,
  similarity function, candidate count, reranker, context builder, generator,
  prompt, and seed constant.
- [ ] Apply the same rule to chunking, top K, filters, hybrid weights, rerankers,
  context ordering, prompts, and generation models.
- [ ] Run paired comparisons on the same queries and include confidence intervals.
- [ ] Reject a candidate that improves an average quality score but violates a
  hard latency, cost, memory, or reliability limit.
- [ ] Promote the best individual components into candidate pipelines, then
  compare those complete pipelines under the same production constraints.
- [ ] Confirm the winner once on the held-out set. Do not repeatedly tune against
  held-out results.

Use a decision table for every experiment:

| Run | Changed variable | Quality delta | p95 latency delta | Cost delta | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | none | 0 | 0 | 0 | keep |
| candidate | embedding model | TBD | TBD | TBD | promote or reject |

### Phase 6. Make the benchmark a regression suite

- [ ] Add a small deterministic smoke set to continuous integration.
- [ ] Run the full benchmark on scheduled releases or material corpus, model, or
  pipeline changes.
- [ ] Fail the quality gate when a primary metric crosses its allowed regression
  budget or a hard constraint is violated.
- [ ] Keep raw per-query outputs and aggregate reports for comparison over time.
- [ ] Track dataset and judge drift. Periodically refresh production-like queries
  without changing historical benchmark versions.
- [ ] Add online monitoring for retrieval misses, unsupported answers, citation
  failures, abstention rate, latency, cost, and user feedback.
- [ ] Feed reviewed production failures into the next dataset version, not into the
  frozen historical test set.

**Deliverable:** a repeatable benchmark command, a comparison report, and a release
gate that detects quality, latency, cost, and reliability regressions.

## Recommended execution order

1. Finish the benchmark contract and dataset schema.
2. Validate the current harness with a fixed BM25 development run.
3. Complete the retrieval baselines and retrieval error analysis.
4. Build the fixed-context generation evaluator.
5. Add the end-to-end runner and query-level traces.
6. Run controlled component experiments.
7. Select candidate pipelines and test once on the held-out split.
8. Automate the winning benchmark as a regression suite.

Generation scores should not be mixed into the current retrieval baseline until
the fixed-context evaluator is available. This preserves failure attribution and
makes each experiment easier to interpret.
