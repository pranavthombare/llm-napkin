use std::io::Write;

use anyhow::{Context, Result};
use llm_napkin::{
    Options,
    types::{Estimate, Metadata},
};

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

fn table(out: &mut impl Write, headers: &[&str], rows: &[Vec<String>]) -> Result<()> {
    let rows: Vec<Vec<String>> = std::iter::once(headers.iter().map(|s| s.to_string()).collect())
        .chain(rows.iter().cloned())
        .map(|row: Vec<String>| {
            row.into_iter()
                .map(|cell| {
                    cell.chars()
                        .map(|c| {
                            if c.is_control() {
                                c.escape_default().to_string()
                            } else {
                                c.to_string()
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let widths: Vec<usize> = (0..headers.len())
        .map(|i| {
            rows.iter()
                .map(|row| row[i].chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect();
    let border = format!(
        "+{}+",
        widths
            .iter()
            .map(|width| "-".repeat(width + 2))
            .collect::<Vec<_>>()
            .join("+")
    );
    writeln!(out, "{border}")?;
    for (index, row) in rows.iter().enumerate() {
        write!(out, "|")?;
        for (value, width) in row.iter().zip(&widths) {
            write!(out, " {value:width$} |")?;
        }
        writeln!(out)?;
        if index == 0 {
            writeln!(out, "{border}")?;
        }
    }
    writeln!(out, "{border}")?;
    Ok(())
}

fn breakdown(out: &mut impl Write, metadata: &Metadata) -> Result<()> {
    let mut rows = Vec::new();
    for (name, component) in &metadata.components {
        for (dtype, stats) in &component.dtypes {
            rows.push(vec![
                name.clone(),
                dtype.clone(),
                stats.param_count.to_string(),
                human(stats.bytes),
            ]);
        }
    }
    table(out, &["Component", "Dtype", "Parameters", "Weights"], &rows)
}

pub fn print(
    out: &mut impl Write,
    result: &Estimate,
    options: &Options,
    details: bool,
) -> Result<()> {
    writeln!(out, "{} @ {}", result.model_id, result.revision)?;
    let mut variants: Vec<(&str, &Metadata)> = Vec::new();
    if let Some(metadata) = &result.safetensors {
        variants.push(("Safetensors", metadata));
    }
    variants.extend(
        result
            .gguf_files
            .iter()
            .map(|(name, metadata)| (name.as_str(), metadata)),
    );
    let mut headers = vec![
        "Input tokens",
        "Output tokens",
        "Context",
        "Batch",
        "Weights",
        "KV dtype",
        "KV cache",
        "Total",
    ];
    let show_variant = !result.gguf_files.is_empty();
    if show_variant {
        headers.insert(0, "GGUF variant");
    }
    let mut rows = Vec::new();
    for (name, metadata) in &variants {
        let cache = metadata.kv_cache.as_ref();
        let context = result
            .workload
            .as_ref()
            .map(|w| w.context_length())
            .transpose()?
            .or_else(|| cache.map(|kv| kv.max_model_len))
            .or(options.max_model_len);
        let total = metadata
            .bytes
            .checked_add(cache.map_or(0, |kv| kv.bytes))
            .context("total memory overflow")?;
        let mut row = vec![
            result
                .workload
                .as_ref()
                .map_or("-".into(), |w| w.input_tokens.to_string()),
            result
                .workload
                .as_ref()
                .map_or("-".into(), |w| w.output_tokens.to_string()),
            context.map_or("-".into(), |n| n.to_string()),
            options.batch_size.to_string(),
            human(metadata.bytes),
            cache.map_or("-".into(), |kv| kv.dtype.clone()),
            cache.map_or("not estimated".into(), |kv| human(kv.bytes)),
            human(total),
        ];
        if show_variant {
            row.insert(0, (*name).into());
        }
        rows.push(row);
    }
    table(out, &headers, &rows)?;
    if details {
        for (name, metadata) in &variants {
            writeln!(out, "\n{name} weight breakdown")?;
            breakdown(out, metadata)?;
        }
    }
    if let Some(moe) = &result.moe {
        writeln!(out, "\nMoE weights (all experts resident)")?;
        table(
            out,
            &[
                "Base model",
                "Expert count",
                "Per expert",
                "All experts",
                "Active per token",
            ],
            &[vec![
                human(moe.base_model.bytes),
                moe.expert_count.to_string(),
                human(moe.experts.bytes),
                human(moe.experts_total.bytes),
                moe.active_expert_count
                    .map_or("-".into(), |n| n.to_string()),
            ]],
        )?;
    }
    if variants.iter().any(|(_, m)| m.kv_cache.is_none()) {
        writeln!(
            out,
            "\nRows without a KV-cache estimate show weights-only totals."
        )?;
    }
    writeln!(
        out,
        "\nTotals cover stored weights and any estimated KV cache; runtime workspaces, activations and allocator overhead are additional."
    )?;
    Ok(())
}
