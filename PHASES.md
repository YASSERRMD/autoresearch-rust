# Autoresearch Rust Migration Plan

This repository ports [karpathy/autoresearch](https://github.com/karpathy/autoresearch) to a production-grade Rust implementation with Python FFI support.

## Phase 1 - Intake and Setup (`codex/phase_1`)

Goal: establish source context and repository hygiene.

Atomic tasks:
1. Clone upstream reference into `tmp/autoresearch`.
2. Add `tmp/` to `.gitignore`.
3. Read `README.md`, `prepare.py`, `train.py`, and `program.md` end-to-end.
4. Write this phased implementation breakdown.

Deliverables:
- Isolated upstream source clone in `tmp/`.
- Clean workspace settings.
- Execution plan for phased delivery.

## Phase 2 - Rust Core Foundations (`codex/phase_2`)

Goal: build high-performance, modular Rust core suitable for both CLI and Python embedding.

Atomic tasks:
1. Create Rust workspace (`autoresearch-core`, `autoresearch-cli`, `pyautoresearch`).
2. Implement shared configuration, cache paths, constants, and typed results.
3. Implement robust shard downloader with retries and concurrent workers.
4. Implement parquet text ingestion and shard splitting (train/val pinned shard).
5. Implement tokenizer training/loading and token-byte lookup generation.
6. Add unit tests for config/path/tokenizer serialization.

Deliverables:
- Compileable core library with data preparation pipeline.
- CLI command surface for `prepare`.

## Phase 3 - Training Engine and Performance (`codex/phase_3`)

Goal: implement a full Rust training/evaluation engine and optimize throughput.

Atomic tasks:
1. Implement GPT model and forward loss path in Rust using `tch`.
2. Implement packed dataloader with BOS alignment and zero-padding avoidance.
3. Implement fixed time-budget training loop and schedule controls.
4. Implement BPB evaluation (token-byte weighted metric).
5. Add performance defaults (mixed precision, grad accumulation, pinned transfer, thread tuning).
6. Add integration tests/smoke scripts for `train` and `eval` flows.

Deliverables:
- End-to-end training executable in Rust.
- Metrics-compatible summary output.

## Phase 4 - Python FFI and Developer Experience (`codex/phase_4`)

Goal: provide first-class Python bindings and clear workflows for Rust/Python developers.

Atomic tasks:
1. Expose Rust APIs via `pyo3` (`prepare`, `train`, `evaluate`, report structs).
2. Add Python packaging config (`maturin`) and wheel build instructions.
3. Add Python and Rust usage examples and scripts.
4. Document performance and deployment settings.
5. Add troubleshooting section for GPU, cache, tokenizer, and build issues.

Deliverables:
- Installable Python extension module backed by Rust core.
- End-user docs for both ecosystems.

## Phase 5 - Optional Accelerator Integrations (`codex/phase_5`)

Goal: wire optional external acceleration hooks without blocking core functionality.

Atomic tasks:
1. Add provider abstraction for external training orchestration.
2. Implement optional `groqtrain` adapter if repository/API access is available.
3. Add fallback behavior when integration is unavailable.
4. Document integration setup and benchmark comparisons.

Deliverables:
- Extensible accelerator interface.
- Non-breaking optional integration path.

## Notes

- Branch naming follows `codex/phase_N` to satisfy workspace branch prefix constraints.
- Each atomic task is committed immediately after completion.
- No commits are pushed to `main`.
