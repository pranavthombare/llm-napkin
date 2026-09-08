# llm-napkin

**Estimate the memory needed to run a Hugging Face model.**

Give it a model ID, prompt length, output budget and batch size. Get a table of
model weights, KV cache and total memory. It reads tensor headers rather than
downloading model weights. The executable needs no Python, PyTorch or GPU.

```sh
llm-napkin HuggingFaceTB/SmolLM2-135M-Instruct -i 2048 -o 512 -b 4
```

```text
+--------------+---------------+---------+-------+------------+----------+------------+------------+
| Input tokens | Output tokens | Context | Batch | Weights    | KV dtype | KV cache   | Total      |
+--------------+---------------+---------+-------+------------+----------+------------+------------+
| 2048         | 512           | 2560    | 4     | 256.57 MiB | BF16     | 225.00 MiB | 481.57 MiB |
+--------------+---------------+---------+-------+------------+----------+------------+------------+
```

Token counts are **per sequence**. Context is input + output; batch size is the
number of concurrent sequences. Totals cover weights and estimated KV cache.
Allow additional memory for runtime workspaces, activations and allocator overhead.

## Install

### Linux and macOS

```sh
curl -fsSL https://github.com/pranavthombare/llm-napkin/releases/latest/download/install.sh | sh
```

The installer selects your platform, verifies the SHA-256 checksum and installs
`llm-napkin` to `~/.local/bin`. If that directory is not on your PATH, run:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

Rerun the install command to update. To pin a version or choose a destination:

```sh
curl -fsSL https://github.com/pranavthombare/llm-napkin/releases/download/v0.1.0/install.sh | sh -s -- --version v0.1.0 --bin-dir "$HOME/.local/bin"
```

### Windows and manual downloads

Download your archive from [GitHub Releases](https://github.com/pranavthombare/llm-napkin/releases),
extract it, and put the executable on your PATH. On Windows, run
`./llm-napkin.exe --help` from the extracted directory to get started.

| System | Archive suffix |
|---|---|
| Linux, Intel/AMD 64-bit | `x86_64-unknown-linux-musl.tar.gz` |
| Linux, ARM64 | `aarch64-unknown-linux-musl.tar.gz` |
| macOS, Apple Silicon | `aarch64-apple-darwin.tar.gz` |
| macOS, Intel | `x86_64-apple-darwin.tar.gz` |
| Windows, Intel/AMD 64-bit | `x86_64-pc-windows-msvc.zip` |

Each archive starts with `llm-napkin-`. Linux archives use static musl builds.
Remove the installed executable to uninstall.

### From source

With Rust 1.85 or newer:

```sh
cargo install --git https://github.com/pranavthombare/llm-napkin --locked
```

From a checkout, use `cargo install --path . --locked`.

## Use

```sh
# Stored weight memory only
llm-napkin sentence-transformers/all-MiniLM-L6-v2

# Prompt + generated tokens, four concurrent sequences
llm-napkin HuggingFaceTB/SmolLM2-135M-Instruct -i 2048 -o 512 -b 4

# Component/dtype breakdown
llm-napkin hf-internal-testing/tiny-sdxl-pipe --details

# JSON for scripts, with byte counts
llm-napkin HuggingFaceTB/SmolLM2-135M-Instruct -i 2048 -o 512 --json

# Compare the GGUF variants in a repository
llm-napkin bartowski/SmolLM2-135M-Instruct-GGUF -i 2048 -o 512

# Select one GGUF file; selecting a shard includes all sibling shards
llm-napkin bartowski/SmolLM2-135M-Instruct-GGUF --gguf-file SmolLM2-135M-Instruct-Q4_K_M.gguf -i 2048 -o 512
```

| Option | Meaning |
|---|---|
| `-i`, `--input-tokens` | Prompt tokens per sequence |
| `-o`, `--output-tokens` | Maximum generated tokens per sequence |
| `-b`, `--batch-size` | Concurrent sequences; default `1` |
| `-d`, `--details` | Component and dtype breakdowns |
| `--json` | Machine-readable output; alias of `--json-output` |
| `-r`, `--revision` | Branch, tag or commit; default `main` |
| `--kv-cache-dtype` | Override cache precision, e.g. `bfloat16` or `fp8` |

Token options enable KV-cache estimation automatically. An omitted input/output
count is zero; their sum must be positive. You can also use
`--experimental --max-model-len 8192` for a combined context budget.
The existing `--model-id MODEL` syntax remains supported. See `llm-napkin --help`
for all options.

For private or gated models, set `HF_TOKEN` or use a saved Hugging Face login.
Your account must have access to the model.

## Learn more

- [Reference](docs/reference.md): supported models, authentication, JSON fields,
  library usage and accuracy limits.
- [Distribution guide](docs/distributing.md): local packaging, checksum verification
  and publishing native releases.

MIT licensed. See [LICENSE](LICENSE).

## Develop

```sh
cargo fmt --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
python3 -m unittest discover -s tests -p 'test_distribution.py'
```

Python 3.11+ is used only for packaging and distribution tests, not by the installed CLI.
