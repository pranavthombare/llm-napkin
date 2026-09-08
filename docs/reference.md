# Reference

[Back to the README](../README.md)

## Authentication and network options

Authentication precedence is `--hf-token`, `HF_TOKEN`, then a saved Hugging Face
token. Token-file lookup honors `HF_TOKEN_PATH`, `HF_HOME` and `XDG_CACHE_HOME`,
falling back to `~/.cache/huggingface/token`. Your account must have access to
gated/private models. Tokens are never included in reports.

`--revision` accepts branches, tags, commits and refs such as `refs/pr/4`.
`--endpoint` / `HF_ENDPOINT` overrides the Hub base URL. `--max-workers` /
`MAX_WORKERS` controls concurrency (default 8, maximum 128). Requests time out
after 30 seconds and retry rate limits and server errors up to twice.

## Token budgets and tables

`--input-tokens` is the prompt-token count per sequence. `--output-tokens` is the
maximum generated-token count per sequence. The KV-cache context is their sum,
multiplied by `--batch-size` concurrent sequences in the cache calculation.
Weights are resident once and do not grow with batch size.

Supplying either token-count option automatically enables experimental KV-cache
estimation; an omitted counterpart is zero. Zero is allowed for either count,
but their sum must be positive. Batch size defaults to 1 and must be positive.
`--max-model-len` remains available with `--experimental` for a combined context
budget and cannot be combined with the separate input/output options.

The default terminal output is a bordered table containing input tokens, output
tokens, total context, batch size, weights, cache dtype, cache memory and total
memory. Unspecified token splits appear as `-`. GGUF repositories show one row
per variant. `--details` adds component/dtype tables; MoE models also show their
base and expert weights in a table.

The cache estimate represents the configured token budget at the end of
generation. Missing/unsupported cache metadata is labeled `not estimated`, with
weights-only totals. Runtime activations, workspaces and allocator overhead are
additional.

## Coverage

- Single and sharded Safetensors, with actual dtype/shape/offset validation.
- Transformers, diffusion pipeline components and Sentence Transformers Dense modules.
- Little-endian GGUF v2/v3, quantized tensor sizes, variant selection and shard merging.
- Safetensors take precedence when both formats exist. `--gguf-file` overrides this.
  `mmproj-*` GGUF projection files are excluded.
- Experimental MHA/GQA KV-cache estimates, explicit head dimensions, batch/context
  controls, nested VLM text configs, pure sliding windows and hybrid reservations.
- Experimental Safetensors MoE base/expert breakdowns for individually named,
  uniform experts. All experts count toward resident weight memory.
- Stable, sorted JSON and terminal reports. Missing cache metadata produces a
  warning and a weights-only result where possible; invalid inputs fail clearly.

## JSON contract

Without `--details`, a single Safetensors model or selected GGUF model returns:

```json
{
  "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "memory": 269030016,
  "kv_cache": null,
  "total_memory": 269030016
}
```

With `--experimental` or explicit token counts, `kv_cache` is the estimated number
of bytes when supported.
Selected GGUF results also include `filename`. Without `--gguf-file`, GGUF results
map logical filenames to byte counts in `memory` and `kv_cache`; `total_memory`
is null because variants are alternatives. This also applies to a repository
containing one GGUF variant. `--details` expands memory into components, parameter
counts and dtypes, and cache into bytes, dtype, context length and batch size.
MoE information appears under `moe`. Warnings go to stderr so stdout remains JSON.
When input/output options are used, a `workload` object records `input_tokens`,
`output_tokens` and `batch_size` in both compact and detailed JSON. Existing
commands without these options keep their previous JSON structure.

## Rust library

```rust,no_run
use llm_napkin::{Options, estimate};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut options = Options::new("HuggingFaceTB/SmolLM2-135M-Instruct");
    options.input_tokens = Some(2048);
    options.output_tokens = Some(512);
    options.batch_size = 4;
    let result = estimate(&options).await?;
    println!("{}", result.to_json(true)?);
    Ok(())
}
```

`estimate` is asynchronous and uses the caller's Tokio runtime. `Options::new`
uses the public Hub and eight workers; library callers set endpoint/concurrency
fields explicitly. Token environment variables are honored by both interfaces.
Parser and KV-cache modules are public for offline use.

## Accuracy and limits

Weight estimates describe stored tensors. Quantized runtimes may dequantize or
repack them; activations, compute workspaces, CUDA graphs, allocator overhead and
runtime cache padding require extra memory. These figures are not a guarantee
that a model fits a particular GPU. MoE parameter counts describe stored tensor
elements, which may be packed for quantized models.

Calculation details and limits:

- GGUF uses exact block byte sizes from
  [ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h),
  including quantization scales; sizes use integer block ratios rather than
  rounded bits-per-weight estimates.
- Pure sliding-window attention is capped by the window even without an explicit
  `layer_types` list. Mixed full/sliding attention uses the full-context reservation
  upper bound. Linear/recurrent layer types are reported as unsupported for cache
  estimation; their state is not silently treated as ordinary attention.
- MLA models use the uncompressed attention cache estimate with a warning;
  engine-specific latent-cache compression is not modeled. `fp8_ds_mla` selects
  one-byte cache precision, not a latent-cache layout.
- GGUF recognizes explicit key/value head lengths and defaults missing KV-head
  counts to MHA. The cache estimate does not include engine-specific block padding.
- MoE top-1 routing is reported as one active expert. Packed/fused expert tensors
  contribute to total weights but do not receive an individual-expert breakdown.
- Unsupported quantization methods require an explicit cache dtype. Unknown
  tensor types, incomplete shards and overlapping Safetensors offsets fail.
- Metadata is capped at 100 MB per file; ignored Range responses are streamed
  only up to the requested prefix. Repository pagination is followed.
- Training/optimizer estimates and the old VS Code UI are not part of this CLI.

