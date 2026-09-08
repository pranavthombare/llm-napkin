# Changelog

## 0.1.0

- Replace the TypeScript VS Code extension with a native Rust CLI and library.
- Estimate memory from Safetensors/GGUF headers, including sharded and multi-component models.
- Add experimental KV-cache/MoE estimates and hf-mem-compatible JSON output.
- Add bounded HTTP reads, pagination, input validation and offline regression tests.
