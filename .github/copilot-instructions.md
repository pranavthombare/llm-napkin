# llm-napkin

This project is a Rust CLI and library that estimates Hugging Face inference
memory from bounded HTTP reads of Safetensors and GGUF metadata.

- Keep model downloading, metadata parsing and reporting separate.
- Do not download full weight files or load Python/model runtimes.
- Use asynchronous HTTP and filesystem operations; never log authentication tokens.
- Keep CLI/JSON compatibility and documented accuracy differences explicit.
- Add regression tests for parser, discovery, cache and output behavior.
- Run `cargo fmt --check`, `cargo test --locked`, and
  `cargo clippy --locked --all-targets -- -D warnings`.
