# OCR97 Roadmap

## Current Position

OCR97 is ready for a public `v0.1` / research-preview style repository after cleanup. The strongest current claim is a measured 2026-07-03 release-gate result:

- `100/100` on the deterministic 120-case `production_router` release gate.
- `98-100/100` on the compact guarded fallback hard-image slice.
- zero below-75 cases in the current production-router gate and guarded fallback slice.

That does not make OCR97 a universal OCR solution. It means the local-first OCR97 routing, preprocessing, scoring, and field-recovery stack is strong enough to publish honestly with caveats.

## Priority 1 - Quality-First Guarded Fallback

The current guarded fallback fix improves correctness by escalating raw OCR only when scored evidence shows high-value field loss.

Next steps:

- Add a latency-aware budget for escalation.
- Queue heavy preprocessing when multiple documents arrive.
- Expose `guarded_fallback_escalated` as a first-class metric.
- Report raw baseline score, escalated score, reason, and accepted/rejected decision in user-facing output.

Success criteria:

- hard-image fallback keeps zero below-75 cases on a larger slice,
- average latency remains bounded under documented quality-first settings,
- no silent fallback hides a high-value field failure.

## Priority 2 - Real-Document Public Corpus

The release gate is deterministic and useful, but public credibility improves when the project includes redistributable real documents.

Next steps:

- Collect legally safe invoices, receipts, statements, forms, and notices.
- Store expected fields in a manifest.
- Include sample inputs, OCR outputs, and field-level score reports.
- Keep private/customer documents out of the repo.

Success criteria:

- at least 50 redistributable real documents,
- at least 8 document categories,
- baseline comparisons against Tesseract and any installed open-source challengers,
- failure examples retained for future improvement.

## Priority 3 - Human Review For Uncertain Fields

OCR97 should not pretend uncertainty is certainty.

Next steps:

- Add an uncertainty output format for low-confidence fields.
- Show source line, field alias, selected candidate, competing candidates, and score.
- Allow a human correction to be saved as Helix97 training data.

Success criteria:

- every uncertain field has an explanation,
- corrections produce structured JSONL examples,
- corrected examples can be replayed in tests.

## Priority 4 - Helix97 Promotion Gates

Helix97 is the improvement loop for field ranking and correction data. It should stay gated.

Promotion gates:

- captured-failure evaluation passes,
- clean manifest does not regress,
- strict matrix does not fall below best baseline,
- real-document subset does not regress,
- model/ranker version is recorded.

Success criteria:

- trained rankers improve hard failures without broad regressions,
- raw OCR recognizer training remains deferred until field/ranking/layout layers are exhausted.

## Priority 5 - Research-Grade Evaluation

OCR97 can support a technical report now. A stronger research paper requires more controlled evidence.

Required additions:

- public real-document dataset or reproducible synthetic generator,
- same-environment baseline engines,
- ablation study,
- latency/cost table,
- failure taxonomy,
- human-review correction loop,
- repeated runs across multiple days or machines.

Potential ablation matrix:

| Layer | Question |
|---|---|
| raw Tesseract | What does the baseline recover alone? |
| preprocessing | How much do image variants improve raw OCR? |
| field ranking | How much does label/candidate ranking improve extraction? |
| consensus | How much do multi-engine candidates improve reliability? |
| guarded fallback | How much quality is recovered at what latency cost? |
| Helix97 ranker | Does learned correction improve captured failures without regressions? |

## Public Release Target

For GitHub, target:

- `v0.1.0-alpha`
- local-first OCR and field extraction
- benchmarked but not universal
- portable CPU path by default
- optional GPU/remote-model paths documented but not required

