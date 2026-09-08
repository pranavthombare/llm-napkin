Native Rust CLI for estimating Hugging Face model memory from Safetensors and
GGUF headers, without downloading model weights.

- Configure prompt/output token budgets and concurrent batch size.
- Read bordered memory tables or JSON, with optional dtype and MoE breakdowns.
- Handle sharded weights, GGUF variants, embedding models and diffusion pipelines.
- Run one native executable on Linux, macOS or Windows.

Install on Linux or macOS:

```sh
curl -fsSL https://github.com/pranavthombare/llm-napkin/releases/download/v0.1.0/install.sh | sh -s -- --version v0.1.0
```

The installer verifies the archive checksum and installs to `~/.local/bin`.
On Windows, download the ZIP and extract `llm-napkin.exe`.

Try:

```sh
llm-napkin HuggingFaceTB/SmolLM2-135M-Instruct -i 2048 -o 512 -b 4
```

Estimates cover stored weights and approximate KV cache. Runtime workspaces,
activations and allocator overhead require additional memory.
