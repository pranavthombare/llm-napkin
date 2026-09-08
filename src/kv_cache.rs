use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

use crate::{
    safetensors::dtype_bits,
    types::{KvCache, Metadata, add, product},
};

pub const SAFETENSORS_DTYPES: &[&str] = &[
    "auto",
    "float32",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_ds_mla",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
];

fn torch_dtype(dtype: &str) -> Result<&'static str> {
    Ok(match dtype.strip_prefix("torch.").unwrap_or(dtype) {
        "float32" => "F32",
        "float16" => "F16",
        "bfloat16" => "BF16",
        "float8_e4m3" | "float8_e4m3fn" => "F8_E4M3",
        "float8_e5m2" => "F8_E5M2",
        "int8" => "I8",
        _ => bail!(
            "cannot infer cache precision from dtype {dtype}; set --kv-cache-dtype explicitly"
        ),
    })
}

pub fn resolve_dtype(config: &Value, requested: &str, metadata: &Metadata) -> Result<String> {
    ensure!(
        SAFETENSORS_DTYPES.contains(&requested),
        "invalid Safetensors cache dtype: {requested}"
    );
    match requested {
        "fp8" | "fp8_ds_mla" | "fp8_e4m3" | "fp8_inc" => return Ok("F8_E4M3".into()),
        "fp8_e5m2" => return Ok("F8_E5M2".into()),
        "float32" | "float16" | "bfloat16" => return Ok(torch_dtype(requested)?.into()),
        _ => (),
    }
    let quant = &config["quantization_config"];
    if let Some(method) = quant["quant_method"].as_str() {
        match method {
            "fp8" | "modelopt" => {
                if let Some(fmt) = quant["fmt"].as_str().or_else(|| quant["format"].as_str()) {
                    let fmt = if fmt.starts_with("float8_") {
                        fmt.to_owned()
                    } else {
                        format!("float8_{fmt}")
                    };
                    return Ok(torch_dtype(&fmt)?.into());
                }
                if quant["kv_cache_scheme"]["num_bits"] == 8
                    && quant["kv_cache_scheme"]["type"] == "float"
                {
                    return Ok("F8_E4M3".into());
                }
                return ["F8_E4M3", "F8_E5M2"]
                    .into_iter()
                    .map(|dtype| {
                        (
                            dtype,
                            metadata
                                .components
                                .values()
                                .filter(|c| c.dtypes.contains_key(dtype))
                                .count(),
                        )
                    })
                    .filter(|(_, count)| *count > 0)
                    .max_by_key(|(_, count)| *count)
                    .map(|(dtype, _)| dtype.to_owned())
                    .context(
                        "quantized cache precision is ambiguous; set --kv-cache-dtype explicitly",
                    );
            }
            "compressed-tensors" if quant["kv_cache_scheme"].is_null() => (),
            _ => bail!(
                "cache precision for quantization method {method} requires an explicit --kv-cache-dtype"
            ),
        }
    }
    let dtype = config["torch_dtype"]
        .as_str()
        .or_else(|| config["dtype"].as_str())
        .context("config has no cache dtype; set --kv-cache-dtype explicitly")?;
    Ok(torch_dtype(dtype)?.into())
}

fn positive(config: &Value, key: &str) -> Result<u64> {
    config[key]
        .as_u64()
        .filter(|n| *n > 0)
        .with_context(|| format!("{key} must be a positive integer"))
}

/// Reserved standard MHA/GQA cache footprint. Hybrid sliding layers reserve full pages.
pub fn safetensors_cache(config: &Value, dtype: &str, length: u64, batch: u64) -> Result<KvCache> {
    ensure!(
        length > 0 && batch > 0,
        "context length and batch size must be positive"
    );
    let layers = positive(config, "num_hidden_layers")?;
    let heads = positive(config, "num_attention_heads")?;
    let hidden = positive(config, "hidden_size")?;
    let kv_heads = if config.get("num_key_value_heads").is_some() {
        positive(config, "num_key_value_heads")?
    } else {
        heads
    };
    let head_dim = if config.get("head_dim").is_some() {
        positive(config, "head_dim")?
    } else {
        ensure!(
            hidden % heads == 0,
            "hidden_size must be divisible by attention heads unless head_dim is explicit"
        );
        hidden / heads
    };
    ensure!(
        kv_heads <= heads && heads % kv_heads == 0,
        "invalid number of key/value heads"
    );
    let full = if config.get("sliding_window_pattern").is_some() {
        layers / positive(config, "sliding_window_pattern")?
    } else if let Some(types) = config["layer_types"].as_array() {
        ensure!(
            types.len() as u64 == layers,
            "layer_types length does not match num_hidden_layers"
        );
        ensure!(
            types.iter().all(|t| matches!(
                t.as_str(),
                Some(
                    "attention"
                        | "full_attention"
                        | "global_attention"
                        | "sliding_attention"
                        | "sliding_window"
                )
            )),
            "nonstandard layer types require an architecture-specific cache model"
        );
        types
            .iter()
            .filter(|t| {
                matches!(
                    t.as_str(),
                    Some("attention" | "full_attention" | "global_attention")
                )
            })
            .count() as u64
    } else if config["sliding_window"].as_u64().is_some_and(|n| n > 0) {
        0
    } else {
        layers
    };
    let sliding = layers - full;
    let window = config["sliding_window"]
        .as_u64()
        .filter(|n| *n > 0)
        .unwrap_or(length);
    let sliding_tokens = if full > 0 && sliding > 0 {
        length
    } else {
        window.min(length)
    };
    let total_tokens = add(
        product(&[full, length])?,
        product(&[sliding, sliding_tokens])?,
    )?;
    let bytes = product(&[
        2,
        total_tokens,
        kv_heads,
        head_dim,
        dtype_bits(dtype)?,
        batch,
    ])? / 8;
    Ok(KvCache {
        bytes,
        dtype: dtype.into(),
        max_model_len: length,
        batch_size: batch,
    })
}
