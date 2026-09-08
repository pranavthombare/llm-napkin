use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;

use crate::types::{Component, DtypeMetadata, MoeMetadata, add, product};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub param_count: u64,
    pub bytes: u64,
}

pub fn dtype_bits(dtype: &str) -> Result<u64> {
    Ok(match dtype {
        "F64" | "I64" | "U64" => 64,
        "F32" | "I32" | "U32" => 32,
        "F16" | "BF16" | "I16" | "U16" => 16,
        "BOOL" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "F8_E4M3FN" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" | "I8" | "U8" => 8,
        "F4" | "F4_E2M1" => 4,
        _ => bail!("unsupported Safetensors dtype: {dtype}"),
    })
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Parse only the JSON header. `file_size`, when known, includes the 8-byte prefix.
pub fn parse_header(header: &[u8], file_size: Option<u64>) -> Result<Vec<Tensor>> {
    let raw: BTreeMap<String, Value> =
        serde_json::from_slice(header).context("invalid Safetensors header JSON")?;
    let mut tensors = Vec::new();
    let mut intervals = Vec::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let raw: RawTensor =
            serde_json::from_value(value).with_context(|| format!("invalid tensor {name}"))?;
        let params = product(&raw.shape)?;
        let bytes = product(&[params, dtype_bits(&raw.dtype)?])?.div_ceil(8);
        let [start, end] = raw.data_offsets;
        ensure!(
            end >= start && end - start == bytes,
            "tensor {name} has invalid data offsets or dtype/shape size"
        );
        if let Some(size) = file_size {
            ensure!(
                add(add(end, header.len() as u64)?, 8)? <= size,
                "tensor {name} extends past the end of the file"
            );
        }
        intervals.push((start, end));
        tensors.push(Tensor {
            name,
            dtype: raw.dtype,
            param_count: params,
            bytes,
        });
    }
    intervals.sort_unstable();
    let mut previous_end = 0;
    for (start, end) in intervals {
        ensure!(
            start == previous_end,
            "Safetensors data offsets overlap or contain a gap"
        );
        previous_end = end;
    }
    Ok(tensors)
}

pub fn component(tensors: &[Tensor]) -> Result<Component> {
    let mut component = Component::default();
    for tensor in tensors {
        component.accumulate(&tensor.dtype, tensor.param_count, tensor.bytes)?;
    }
    Ok(component)
}

fn config_count(config: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| config[*key].as_u64().filter(|v| *v > 0))
}

pub fn moe_metadata(
    components: &BTreeMap<String, Vec<Tensor>>,
    config: &Value,
) -> Result<Option<MoeMetadata>> {
    let mut base_model = Component::default();
    let mut experts: BTreeMap<u64, Component> = BTreeMap::new();
    for tensor in components.values().flatten() {
        let parts: Vec<_> = tensor.name.split('.').collect();
        let expert_id = parts.windows(2).find_map(|pair| {
            if matches!(
                pair[0],
                "expert"
                    | "experts"
                    | "local_expert"
                    | "local_experts"
                    | "routed_expert"
                    | "routed_experts"
            ) {
                pair[1].parse::<u64>().ok()
            } else {
                None
            }
        });
        let target = match expert_id {
            Some(id) => experts.entry(id).or_default(),
            None => &mut base_model,
        };
        target.accumulate(&tensor.dtype, tensor.param_count, tensor.bytes)?;
    }
    let Some(template) = experts.values().next().cloned() else {
        return Ok(None);
    };
    let count = experts.len() as u64;
    if let Some(configured) = config_count(
        config,
        &[
            "num_local_experts",
            "n_routed_experts",
            "num_experts",
            "moe_num_experts",
        ],
    ) {
        ensure!(
            count == configured && experts.keys().copied().eq(0..configured),
            "MoE expert IDs do not match the configured expert count"
        );
    }
    ensure!(
        experts.values().all(|expert| *expert == template),
        "MoE experts are not uniform; cannot summarize a representative expert"
    );
    Ok(Some(MoeMetadata {
        base_model,
        expert_count: count,
        active_expert_count: config_count(
            config,
            &["num_experts_per_tok", "num_experts_per_token", "top_k"],
        ),
        experts_total: DtypeMetadata {
            bytes: product(&[template.bytes, count])?,
            param_count: product(&[template.param_count, count])?,
        },
        experts: template,
    }))
}
