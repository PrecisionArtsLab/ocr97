# OCR97

OCR97 is a portable, local-first OCR and document extraction pipeline. It combines local OCR engines, native PDF text extraction, preprocessing, scoring, and a small HTTP gateway so applications and agents can extract text and structured fields from documents without depending on a hosted OCR service.

OCR97 is the project name and release target. It is not a blanket claim that every document scores 97% accuracy. Published grades should come from benchmark reports with sample inputs, outputs, field-level scoring, and current test dates.

## Current Verified Status

As of 2026-07-03, OCR97's current release-gate result is:

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

That grade applies to the supported `production_router` path: native PDF text extraction for digital PDFs and OCR97's local image preprocessing/router for image variants. It does not mean plain Tesseract, every fallback lane, or every real-world document scores 100. Raw fallback lanes remain diagnostic evidence and are reported separately.

Latest hard-image guarded fallback proof:

- `tesseract_broad_rotated`: `98` average, `0` below-75 cases.
- `tesseract_broad_noisy_scan`: `100` average, `0` below-75 cases.
- Guarded fallback escalates weak hard-image raw OCR to `local_image_preprocessed_best` when high-value fields are missing or wrong.

## What OCR97 Does

- Extracts text from images and PDFs.
- Uses native PDF text extraction when a digital PDF already contains selectable text.
- Uses local OCR engines such as Tesseract and RapidOCR for image documents.
- Runs adaptive preprocessing for harder images, including rotated, noisy, or low-contrast scans.
- Exposes CLI and HTTP gateway entrypoints for scripts, local tools, desktop apps, and agent systems.
- Includes benchmark manifests and test harnesses so quality claims can be measured instead of guessed.

## Native Desktop And VM Use

OCR97 is not tied to an Aegis VM, noVNC desktop, container, or cloud runtime. The default package can run directly on a native desktop or workstation as long as Python and the local OCR dependencies are installed.

Supported operating patterns:

- Native desktop app helper: run `ocr97-serve` on the same Windows, Linux, or macOS machine that is using OCR97, then call `http://127.0.0.1:5521/ocr/*` from a local app, browser tool, script, or agent.
- Native workstation batch tool: call the `ocr97` CLI from scheduled jobs, PowerShell, bash, Python scripts, or document-processing pipelines without starting a VM.
- VM or container service: run the same CLI/server inside a VM or container when isolation, repeatable environments, remote access, or agent-controlled desktops are useful.
- Mixed native/VM setup: keep OCR97 on the native machine for direct access to local files and hardware, while VM agents call the OCR97 gateway over an explicitly configured local network route.

Use a VM when you want sandboxing, disposable test environments, or visual agent automation. Use native desktop mode when you want lower overhead, direct access to files/hardware, simpler debugging, or integration with normal desktop workflows. In both modes, OCR97 should be treated as the same local OCR service with the same benchmark and quality gates.

## Install

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e .[dev]
python -m pytest -q
```

Optional heavier model lanes can be installed separately:

```bash
python -m pip install .[ml,engines]
```

The default install is intended to be portable. OCR97 scales from the hardware it can actually prove is available instead of assuming a specific workstation, GPU, GB10, or remote model endpoint.

## Engine And Provider Routing

OCR97 now exposes a shared engine registry so routing is driven by capability classes rather than hardcoded model-name branches scattered across the stack.

- Canonical runtime engines remain available: `native_pdf_text`, `rapidocr`, `tesseract`, `local_image_best`, `local_image_preprocessed_best`, `gb10_qwen_ocr`, `gb10_got_ocr2`, `gb10_paddleocr_vl`, `mineru2_5`, and `olmocr2`.
- Generic aliases are also accepted anywhere an engine/model name is requested. Examples:
  - `native_text` -> `native_pdf_text`
  - `image_router` -> `local_image_best`
  - `image_preprocessor` -> `local_image_preprocessed_best`
  - `semantic_cleanup` -> `gb10_qwen_ocr`
  - `dense_scan_vision` -> `gb10_got_ocr2`
  - `layout_vision` -> `gb10_paddleocr_vl`
  - `structure_parser` -> `mineru2_5`
  - `linearization` -> `olmocr2`

This keeps older integrations working while letting new callers target OCR capabilities instead of private lane names.

## Hardware Scaling

OCR97 starts with the portable local path and enables stronger lanes only when the host proves they are available.

| Hardware profile | What OCR97 should use | Typical use |
|---|---|---|
| `cpu` | Native PDF text, Tesseract, RapidOCR, local preprocessing | Laptops, desktops, VMs, CI, and simple document workflows |
| `local-gpu` | CPU lanes plus optional local GPU/model lanes when modules and model assets are present | Workstations with CUDA-capable GPUs |
| `remote-model` | CPU lanes plus an explicitly configured remote OCR/model gateway | GB10/GX10, Ollama hosts, or another model server |
| `workstation` | CPU lanes plus local GPU and remote model options when both are configured | Larger operator-controlled document stations |
| `auto` | Detects the best safe profile from environment and installed capabilities | Default |

Set `OCR97_HARDWARE_PROFILE=cpu`, `local-gpu`, `remote-model`, `workstation`, or `auto` to override the automatic choice. Compatibility aliases such as `cpu-only`, `gpu`, `cuda`, `remote`, `ollama`, `gb10`, and `gx10` normalize to the generic profiles above.

`ocr97 doctor` and `examples/doctor_payload_demo.py` report the effective hardware profile, baseline lanes, optional lanes, and why that profile was selected. A public build should never require a specific private machine to pass its smoke checks.

## Baseline Comparison

OCR97 includes a baseline comparison runner for measuring actual OCR outputs against open-source baselines through the same field-evidence scorer.

```bash
python -m ocr97.baseline_compare --manifest benchmarks/truth10_manifest.json --fixture-dir artifacts/baseline_compare/truth10_scored_baselines/fixtures --artifact-dir artifacts/baseline_compare/truth10_scored_baselines --variant clean --engines ocr97,tesseract,easyocr --ocr97-engine local_image_best
```

The runner writes `baseline_comparison.json` and `baseline_comparison.md`. Baselines that are not installed or cannot start are marked as skipped with a reason, not counted as failed OCR quality.

Current strict generated-field evidence is tracked in `artifacts/baseline_compare/above_ocr_layer_pass_20260702/above_ocr_layer_summary.md`. On that saved-output rescore, OCR97 beats Tesseract by `+2` to `+10` points across generated hard variants after label-proximity field ranking and gateway consensus integration. Treat this as a benchmark-specific proof, not a blanket accuracy claim.

## Helix97 Improvement Loop

`Helix97` is the named OCR97 correction dataset and local training pipeline. It saves OCR97 field failures as JSONL training records, trains a local field-ranker/corrector, exports weak layout-region examples, and writes a GB10/Ollama training handoff plan for teacher rationales and synthetic expansion.

```bash
python -m ocr97.helix97 run --comparison artifacts/baseline_compare/strict_hard_matrix_20260702T131800Z/clean/baseline_comparison.json --output-dir artifacts/helix97/clean
```

Raw OCR recognizer training is intentionally gated. Helix97 only recommends considering it after field ranking, correction, and layout-region training mature and OCR97 still loses to baselines on real-document benchmark gates.

If GB10/Ollama is available, `python -m ocr97.helix97 gb10-augment ...` can add local teacher rationales, hard negatives, and synthetic variant prompts without sending documents to a cloud OCR service.

Trained Helix97 field-ranker models can be enabled with `OCR97_HELIX97_FIELD_RANKER_MODEL=/path/to/helix97_field_ranker_model.json`. Models should only be promoted after passing captured-failure, clean-manifest, and strict-matrix gates.

## Quick Start

Start the local OCR gateway:

```bash
ocr97-serve --host 127.0.0.1 --port 5521
```

Useful endpoints:

- `GET /ocr/health`
- `GET /ocr/capabilities`
- `POST /ocr/extract`

Run a lightweight diagnostics payload without loading heavy model runtimes:

```bash
python examples/doctor_payload_demo.py
```

## Command Line Tools

The package exposes these entrypoints:

- `ocr97`
- `ocr97-serve`
- `ocr97-bootstrap`
- `ocr97-truth-benchmark`
- `ocr97-truth-runner`
- `ocr97-mixed-corpus-benchmark`
- `ocr97-release-grade`
- `ocr97-capability-audit`

## Quality And Benchmarks

OCR97 should be judged by measured benchmark evidence:

- sample input documents,
- extracted output,
- expected truth data,
- field-level scoring,
- pass/fail gates,
- latency and fallback behavior,
- current run date.

The repository includes benchmark manifests under `benchmarks/` and tests under `tests/`. Generated run artifacts should not be committed unless they are curated release evidence.

For the current benchmark summary, commands, and caveats, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Improvement Roadmap

The next improvements should stay quality-first:

- Add a latency-aware escalation budget for guarded fallback so heavy preprocessing is queued or capped without hiding field failures.
- Build a larger legally safe real-document corpus with expected fields, sample outputs, field-level scores, and failure examples.
- Continue Helix97 captured-failure training, but only promote trained rankers after captured-failure, clean-manifest, and strict-matrix gates pass.
- Add a human-review workflow for uncertain fields with source lines, competing candidates, and suggested corrections.
- Keep public grades separated by evidence type: release-gate production router, raw fallback diagnostics, generated baseline comparisons, and real-document proof.

For the fuller quality-first roadmap, see [docs/ROADMAP.md](docs/ROADMAP.md).

## Public Repo Boundary

This public candidate is intended to include:

- package source under `src/ocr97`,
- tests,
- benchmark manifests,
- examples,
- public documentation,
- packaging metadata,
- release audit tooling.

It should not include private customer documents, machine-local paths, credentials, generated runtime artifacts, temporary outputs, desktop automation screenshots, or unpublished internal account workflows.

## Release Audit

Before publishing a standalone repository, run:

```bash
python tools/ocr97_public_release_audit.py
```

The audit checks for required public files, local machine path leakage, generated artifact directories, credential-like assignments, and misleading OCR97 accuracy wording.

## License

MIT. See `LICENSE`.
