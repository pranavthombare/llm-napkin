use std::{
    io::{self, Write},
    process::ExitCode,
};

use anyhow::Result;
use clap::Parser;
use llm_napkin::{
    Options, estimate,
    types::{Estimate, Metadata},
};

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
    /// Context length including prompt and generated tokens.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    max_model_len: Option<u64>,
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
    /// Include component, parameter and dtype breakdowns in JSON.
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

fn human(bytes: u64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < units.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", units[unit])
}

fn print_metadata(out: &mut impl Write, metadata: &Metadata) -> Result<()> {
    writeln!(
        out,
        "  {:<28} {:>16} {:>20}",
        "Component / dtype", "Parameters", "Weights"
    )?;
    for (name, component) in &metadata.components {
        writeln!(
            out,
            "  {name:<28} {:>16} {:>20}",
            component.param_count,
            human(component.bytes)
        )?;
        for (dtype, stats) in &component.dtypes {
            writeln!(
                out,
                "    {dtype:<26} {:>16} {:>20}",
                stats.param_count,
                human(stats.bytes)
            )?;
        }
    }
    writeln!(out, "  Weights: {}", human(metadata.bytes))?;
    if let Some(kv) = &metadata.kv_cache {
        writeln!(
            out,
            "  KV cache: {} ({}, context {}, batch {})",
            human(kv.bytes),
            kv.dtype,
            kv.max_model_len,
            kv.batch_size
        )?;
        let total = metadata
            .bytes
            .checked_add(kv.bytes)
            .ok_or_else(|| anyhow::anyhow!("total memory overflow"))?;
        writeln!(out, "  Total: {}", human(total))?;
    }
    Ok(())
}

fn print_report(out: &mut impl Write, result: &Estimate) -> Result<()> {
    writeln!(out, "{} @ {}", result.model_id, result.revision)?;
    if let Some(metadata) = &result.safetensors {
        print_metadata(out, metadata)?;
    }
    for (filename, metadata) in &result.gguf_files {
        writeln!(out, "\n{filename}")?;
        print_metadata(out, metadata)?;
    }
    if let Some(moe) = &result.moe {
        writeln!(out, "\n  MoE base: {}", human(moe.base_model.bytes))?;
        writeln!(
            out,
            "  {} experts × {} = {} (all expert weights resident)",
            moe.expert_count,
            human(moe.experts.bytes),
            human(moe.experts_total.bytes)
        )?;
        if let Some(active) = moe.active_expert_count {
            writeln!(out, "  Active experts per token: {active}")?;
        }
    }
    writeln!(
        out,
        "\nEstimates cover stored weights{}; runtime workspaces, activations and allocator overhead are additional.",
        if result
            .safetensors
            .as_ref()
            .is_some_and(|m| m.kv_cache.is_some())
            || result.gguf_files.values().any(|m| m.kv_cache.is_some())
        {
            " and approximate KV cache"
        } else {
            ""
        }
    )?;
    Ok(())
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let options = Options {
        model_id: cli.model_id,
        revision: cli.revision,
        hf_token: cli.hf_token,
        experimental: cli.experimental,
        max_model_len: cli.max_model_len,
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
        print_report(&mut out, &result)?;
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
