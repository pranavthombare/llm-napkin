use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub(crate) fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .context("memory or parameter count overflow")
}

pub(crate) fn product(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1_u64, |n, v| {
        n.checked_mul(*v)
            .context("memory or parameter count overflow")
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DtypeMetadata {
    pub bytes: u64,
    pub param_count: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Component {
    pub bytes: u64,
    pub param_count: u64,
    pub dtypes: BTreeMap<String, DtypeMetadata>,
}

impl Component {
    pub(crate) fn accumulate(&mut self, dtype: &str, params: u64, bytes: u64) -> Result<()> {
        self.bytes = add(self.bytes, bytes)?;
        self.param_count = add(self.param_count, params)?;
        let entry = self.dtypes.entry(dtype.into()).or_default();
        entry.bytes = add(entry.bytes, bytes)?;
        entry.param_count = add(entry.param_count, params)?;
        Ok(())
    }

    pub(crate) fn merge(&mut self, other: &Self) -> Result<()> {
        for (dtype, stats) in &other.dtypes {
            self.accumulate(dtype, stats.param_count, stats.bytes)?;
        }
        Ok(())
    }

    fn to_json(&self, details: bool) -> Value {
        let mut value = json!({"bytes": self.bytes, "param_count": self.param_count});
        if details {
            value["dtypes"] = json!(self.dtypes);
        }
        value
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvCache {
    pub bytes: u64,
    pub dtype: String,
    pub max_model_len: u64,
    pub batch_size: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metadata {
    pub bytes: u64,
    pub components: BTreeMap<String, Component>,
    #[serde(skip)]
    pub kv_cache: Option<KvCache>,
}

impl Metadata {
    pub(crate) fn merge_component(&mut self, name: &str, component: &Component) -> Result<()> {
        self.bytes = add(self.bytes, component.bytes)?;
        self.components
            .entry(name.into())
            .or_default()
            .merge(component)
    }

    pub(crate) fn merge(&mut self, other: &Self) -> Result<()> {
        for (name, component) in &other.components {
            self.merge_component(name, component)?;
        }
        if self.kv_cache.is_none() {
            self.kv_cache = other.kv_cache.clone();
        }
        Ok(())
    }

    fn to_json(&self) -> Value {
        json!({"bytes": self.bytes, "components": self.components})
    }
}

#[derive(Clone, Debug)]
pub struct MoeMetadata {
    pub base_model: Component,
    pub expert_count: u64,
    pub active_expert_count: Option<u64>,
    pub experts_total: DtypeMetadata,
    pub experts: Component,
}

impl MoeMetadata {
    fn to_json(&self, details: bool) -> Value {
        json!({
            "base_model": self.base_model.to_json(details),
            "expert_count": self.expert_count,
            "active_expert_count": self.active_expert_count,
            "experts_total": self.experts_total,
            "experts": self.experts.to_json(details),
        })
    }
}

/// Typed result. `to_json` preserves hf-mem's scalar versus multi-GGUF contract.
#[derive(Clone, Debug)]
pub struct Estimate {
    pub model_id: String,
    pub revision: String,
    pub filename: Option<String>,
    pub safetensors: Option<Metadata>,
    pub gguf_files: BTreeMap<String, Metadata>,
    pub moe: Option<MoeMetadata>,
    pub warnings: Vec<String>,
}

impl Estimate {
    pub fn total_memory(&self) -> Result<Option<u64>> {
        let metadata = self.safetensors.as_ref().or_else(|| {
            self.filename
                .as_ref()
                .and_then(|name| self.gguf_files.get(name))
        });
        metadata
            .map(|m| add(m.bytes, m.kv_cache.as_ref().map_or(0, |kv| kv.bytes)))
            .transpose()
    }

    pub fn to_json(&self, details: bool) -> Result<Value> {
        let mut value = json!({"model_id": self.model_id, "total_memory": self.total_memory()?});
        let single = self.safetensors.as_ref().or_else(|| {
            self.filename
                .as_ref()
                .and_then(|name| self.gguf_files.get(name))
        });
        if let Some(filename) = &self.filename {
            value["filename"] = json!(filename);
        }
        if let Some(metadata) = single {
            value["memory"] = if details {
                metadata.to_json()
            } else {
                json!(metadata.bytes)
            };
            value["kv_cache"] = match &metadata.kv_cache {
                Some(kv) if details => json!(kv),
                Some(kv) => json!(kv.bytes),
                None => Value::Null,
            };
        } else {
            value["memory"] = self
                .gguf_files
                .iter()
                .map(|(name, metadata)| {
                    (
                        name.clone(),
                        if details {
                            metadata.to_json()
                        } else {
                            json!(metadata.bytes)
                        },
                    )
                })
                .collect();
            let caches: serde_json::Map<String, Value> = self
                .gguf_files
                .iter()
                .filter_map(|(name, metadata)| {
                    metadata.kv_cache.as_ref().map(|kv| {
                        (
                            name.clone(),
                            if details { json!(kv) } else { json!(kv.bytes) },
                        )
                    })
                })
                .collect();
            value["kv_cache"] = if caches.is_empty() {
                Value::Null
            } else {
                Value::Object(caches)
            };
        }
        if let Some(moe) = &self.moe {
            value["moe"] = moe.to_json(details);
        }
        Ok(value)
    }
}
