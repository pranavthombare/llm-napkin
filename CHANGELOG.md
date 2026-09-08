# Changelog

## Unreleased

- Add per-sequence input/output token budgets with automatic KV-cache estimation.
- Show configuration, weights, cache and totals in bordered terminal tables.
- Add optional workload fields to JSON and component tables with `--details`.

## 0.1.0

- Replace the TypeScript VS Code extension with a native Rust CLI and library.
- Estimate memory from Safetensors/GGUF headers, including sharded and multi-component models.
- Add experimental KV-cache/MoE estimates and hf-mem-compatible JSON output.
- Add bounded HTTP reads, pagination, input validation and offline regression tests.
