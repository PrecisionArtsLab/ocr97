# OCR97 Benchmarks

## Benchmark Policy

OCR97 grades must be tied to dated evidence, sample inputs, extracted outputs, expected fields, and field-level scoring. Do not treat the project name as a universal accuracy claim.

Public benchmark language should distinguish:

- `production_router`: the supported OCR97 grade path.
- raw fallback lanes: diagnostic compatibility baselines such as plain Tesseract.
- generated evidence: deterministic fixtures and synthetic variants.
- real-document evidence: legally safe real samples with expected fields.

## Current Release Gate

Latest verified release-gate result, 2026-07-03:

| Measure | Result |
|---|---:|
| FTP-published grade | `100/100` |
| Evidence gate | `passed` / `elite` |
| Release cases | `120` |
| Document categories | `16` |
| Worst production-router average | `93` |
| Cases below 75 | `0` |
| Average latency | `14527.69 ms` |
| P95 latency | `22502.72 ms` |

This result applies to the `production_router` view: native PDF extraction for digital PDFs plus OCR97's local image preprocessing/router for image variants.

Primary evidence files:

- `benchmarks/release_97_gate_manifest.json`
- `benchmarks/ocr97_97_grade_queue.json`
- `artifacts/release_97_gate/broad_120_case_gate/mixed_corpus_summary.json`

For a public GitHub repo, include a curated summary report rather than the full generated `artifacts/` tree.

## Guarded Fallback Stress

The July 3 hard-image stress pass targeted weak raw fallback behavior. OCR97 now preserves raw OCR as baseline evidence, then escalates to `local_image_preprocessed_best` when scoring proves a high-value miss.

Compact guarded fallback slice:

| Lane | Avg score | Below 75 | Notes |
|---|---:|---:|---|
| `native_pdf_text_broad` | `100` | `0` | baseline still perfect |
| `tesseract_broad_rotated` | `98` | `0` | 6 escalation attempts, all accepted |
| `tesseract_broad_noisy_scan` | `100` | `0` | 10 escalation attempts, all accepted |

Representative failures recovered:

- noisy invoice: raw Tesseract missed `amount_due`, score `69`; guarded fallback recovered to `100`.
- rotated line-item invoice: raw Tesseract misread `invoice_number`, score `73`; guarded fallback recovered to `96-100`.

The tradeoff is latency. This is a quality-first path and should be budgeted or queued when many hard images arrive at once.

## Baseline Comparisons

OCR97's practical advantage over plain Tesseract is above the raw OCR layer:

- preprocessing,
- multi-candidate routing,
- label-proximity field ranking,
- OCR-aware token normalization,
- consensus extraction,
- guarded fallback escalation.

Current generated strict evidence shows OCR97 beating the best available baseline by roughly `+2` to `+10` points across hard variants. Treat that as generated-field evidence, not a universal real-document claim.

Baseline comparison command:

```bash
python -m ocr97.baseline_compare \
  --manifest benchmarks/truth10_manifest.json \
  --fixture-dir artifacts/baseline_compare/truth10_scored_baselines/fixtures \
  --artifact-dir artifacts/baseline_compare/truth10_scored_baselines \
  --variant clean \
  --engines ocr97,tesseract,easyocr \
  --ocr97-engine local_image_best
```

## Reproducible Commands

Run the lightweight truth benchmark:

```bash
python -m ocr97.truth_benchmark \
  --manifest benchmarks/truth10_manifest.json \
  --output /tmp/ocr97_truth10_result.json
```

Run actual gateway extraction on generated image fixtures:

```bash
python -m ocr97.truth_runner \
  --mode image \
  --variant noisy_scan \
  --engine local_image_preprocessed_best \
  --manifest benchmarks/truth10_manifest.json \
  --fixture-dir /tmp/ocr97_truth10_image_fixtures \
  --artifact-dir /tmp/ocr97_truth10_image_artifacts \
  --output /tmp/ocr97_truth10_image_gateway.json
```

Run the mixed corpus release gate from the OCR97 repo:

```bash
python -m ocr97.mixed_corpus_benchmark \
  --manifest benchmarks/release_97_gate_manifest.json \
  --output-dir artifacts/release_97_gate/broad_120_case_gate \
  --broad-variants clean,mild_degraded,low_contrast \
  --focus-variants rotated,noisy_scan \
  --focus-limit 24
```

## Evidence Gaps

The biggest remaining public-evidence gap is a larger real-document corpus with redistributable samples. OCR97 is strong enough for a public `v0.1` release, but a research-grade claim needs:

- public real-document samples,
- baselines run in the same environment,
- ablation tests for each OCR97 layer,
- failure examples,
- latency/cost tables,
- human-review outcomes.

