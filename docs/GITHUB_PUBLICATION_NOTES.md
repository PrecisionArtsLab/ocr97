# OCR97 GitHub Publication Notes

## Publication Verdict

OCR97 is worth publishing as an open-source Precision Arts Lab subproduct after repository cleanup.

The publishable value is not that OCR97 replaces every OCR engine. The value is the layer above raw OCR:

- local-first OCR routing,
- native PDF text extraction,
- hard-image preprocessing,
- field-aware scoring,
- consensus extraction,
- guarded fallback escalation,
- benchmark discipline,
- Helix97 captured-failure improvement loop.

OCR97 should be presented as a measured local document-extraction pipeline, not as a universal accuracy claim.

## Current Public Claim

Current verified result as of 2026-07-03:

- FTP-published grade: `100/100`
- Evidence gate: `passed` / `elite`
- Benchmark: OCR97 120-case release-gate `production_router`
- Cases: `120`
- Document categories: `16`
- Worst production-router average: `93`
- Below-75 cases: `0`
- Average latency: `14527.69 ms`
- P95 latency: `22502.72 ms`

Allowed wording:

> OCR97 passed its deterministic 120-case production-router release gate at 100/100 on the July 3, 2026 benchmark.

Avoid wording:

> OCR97 is 100% accurate OCR.

> OCR97 guarantees 97% accuracy on all documents.

## GitHub-Ready Scope

Include:

- `src/ocr97/`
- `tests/`
- `benchmarks/`
- `examples/`
- `tools/ocr97_public_release_audit.py`
- `pyproject.toml`
- `LICENSE`
- `PUBLIC_README.md` as the basis for the public `README.md`
- small curated proof reports under `docs/` if they contain no private paths or private documents

Exclude:

- large generated `artifacts/`
- `_tmp/`
- machine-local reports with absolute private paths
- private customer or employee documents
- credentials, API keys, tunnel configs, personal machine names
- screenshots or desktop automation dumps unless explicitly curated and sanitized
- internal Sky/Aegis/FTP operational outputs that are not needed to run OCR97

## Suggested Repo Shape

```text
ocr97/
  README.md
  LICENSE
  pyproject.toml
  src/ocr97/
  tests/
  benchmarks/
  examples/
  tools/
  docs/
    GITHUB_PUBLICATION_NOTES.md
    BENCHMARKS.md
    ROADMAP.md
```

## README Cleanup

Use `PUBLIC_README.md` as the public README base.

The current long `README.md` contains useful history, but it also contains stale grades and internal operational notes. Keep the current long README only as an internal history artifact or split it into:

- `docs/history/README_HISTORY_2026_Q2.md`
- `docs/BENCHMARKS.md`
- `docs/ROADMAP.md`
- `docs/GB10_OPTIONAL_LANES.md`

The public README should lead with:

1. What OCR97 does.
2. Current verified status.
3. What the grade means and does not mean.
4. Quick start.
5. Hardware scaling.
6. Benchmarks.
7. Roadmap.
8. Repo boundary.

## Next Improvements Before Public Release

Quality-first:

- Expand the real-document corpus with legally safe examples, expected fields, and sample outputs.
- Add human-review UI/workflow for uncertain fields.
- Promote Helix97 only after captured-failure, clean-manifest, and strict-matrix gates pass.
- Keep raw fallback diagnostics separate from the production-router grade.

Operational:

- Add latency-aware guarded fallback budget.
- Add a small `BENCHMARKS.md` with reproducible commands and curated links.
- Run `python tools/ocr97_public_release_audit.py`.
- Run `python -m pytest -q` or a documented smaller public release test command.

## Research / Paper Readiness

OCR97 can support a technical report now. A stronger research paper would require:

- public real-document dataset or reproducible synthetic-document generator,
- multiple baseline engines runnable in the same environment,
- controlled ablation study: raw Tesseract, preprocessing only, field ranking only, consensus only, guarded fallback,
- latency/cost table,
- failure taxonomy,
- human-review examples,
- strict separation between generated, real, and private evidence.

The current system is GitHub-worthy before it is research-paper-ready.

