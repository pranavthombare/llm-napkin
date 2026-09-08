use std::{
    io::{self, Write},
    process::ExitCode,
};

use anyhow::Result;
use clap::Parser;
use llm_napkin::{Options, estimate};

mod report;

#[derive(Parser)]
#[command(
    version,
    about = "Estimate Hugging Face inference memory without downloading model weights"
)]
struct Cli {
    /// Model ID on the Hugging Face Hub (owner/name).
    #[arg(long)]
    model_id: String,
    #[arg(long, default_value = "main")]
    revision: String,
    /// Token for private/gated models. Falls back to HF_TOKEN and the HF token file.
    #[arg(long, hide_env_values = true)]
    hf_token: Option<String>,
    /// Include approximate KV-cache and MoE weight breakdowns.
    #[arg(long)]
    experimental: bool,
    /// Context length including prompt and generated tokens; alternative to input/output counts.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    max_model_len: Option<u64>,
    /// Prompt tokens per sequence. Enables KV cache estimation; omitted output count is zero.
    #[arg(long, conflicts_with = "max_model_len")]
    input_tokens: Option<u64>,
    /// Maximum generated tokens per sequence. Enables KV cache estimation; omitted input count is zero.
    #[arg(long, conflicts_with = "max_model_len")]
    output_tokens: Option<u64>,
    /// Number of concurrent sequences.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..))]
    batch_size: u64,
    /// Cache precision: auto, float16, float32, bfloat16, fp8 variants; GGUF uses F16, Q8_0, etc.
    #[arg(long, default_value = "auto")]
    kv_cache_dtype: String,
    /// Select a GGUF file, or one shard of a complete GGUF model.
    #[arg(long)]
    gguf_file: Option<String>,
    #[arg(long)]
    json_output: bool,
    /// Include component, parameter and dtype breakdowns in tables or JSON.
    #[arg(long)]
    details: bool,
    /// Accepted for hf-mem compatibility; has no effect.
    #[arg(long, hide = true)]
    ignore_table_width: bool,
    /// Hugging Face Hub base URL.
    #[arg(long, env = "HF_ENDPOINT", default_value = "https://huggingface.co")]
    endpoint: String,
    /// Maximum concurrent metadata fetches (1..128).
    #[arg(long, env = "MAX_WORKERS", default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..=128))]
    max_workers: u16,
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let options = Options {
        model_id: cli.model_id,
        revision: cli.revision,
        hf_token: cli.hf_token,
        experimental: cli.experimental,
        max_model_len: cli.max_model_len,
        input_tokens: cli.input_tokens,
        output_tokens: cli.output_tokens,
        batch_size: cli.batch_size,
        kv_cache_dtype: cli.kv_cache_dtype,
        gguf_file: cli.gguf_file,
        endpoint: cli.endpoint,
        max_workers: cli.max_workers.into(),
    };
    let result = estimate(&options).await?;
    for warning in &result.warnings {
        eprintln!("warning: {warning}");
    }
    let mut out = io::stdout().lock();
    if cli.json_output {
        serde_json::to_writer_pretty(&mut out, &result.to_json(cli.details)?)?;
        writeln!(out)?;
    } else {
        report::print(&mut out, &result, &options, cli.details)?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error)
            if error
                .downcast_ref::<io::Error>()
                .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe) =>
        {
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
