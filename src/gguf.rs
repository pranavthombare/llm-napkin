use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

use crate::types::{Component, KvCache, Metadata, product};

/// (GGML type ID, name, weights per block, bytes per block), including scale overhead.
/// Source: llama.cpp/ggml/src/ggml-common.h; integer ratios avoid rounded bits/weight.
pub const DTYPES: &[(u32, &str, u64, u64)] = &[
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (9, "Q8_1", 32, 36),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (15, "Q8_K", 256, 292),
    (16, "IQ2_XXS", 256, 66),
    (17, "IQ2_XS", 256, 74),
    (18, "IQ3_XXS", 256, 98),
    (19, "IQ1_S", 256, 50),
    (20, "IQ4_NL", 32, 18),
    (21, "IQ3_S", 256, 110),
    (22, "IQ2_S", 256, 82),
    (23, "IQ4_XS", 256, 136),
    (24, "I8", 1, 1),
    (25, "I16", 1, 2),
    (26, "I32", 1, 4),
    (27, "I64", 1, 8),
    (28, "F64", 1, 8),
    (29, "IQ1_M", 256, 56),
    (30, "BF16", 1, 2),
    (34, "TQ1_0", 256, 54),
    (35, "TQ2_0", 256, 66),
    (39, "MXFP4", 32, 17),
];

#[derive(Debug)]
pub struct Truncated;
impl std::fmt::Display for Truncated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "truncated GGUF metadata")
    }
}
impl std::error::Error for Truncated {}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .context("GGUF offset overflow")?;
        let value = self.bytes.get(self.offset..end).ok_or(Truncated)?;
        self.offset = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<&'a str> {
        let len = usize::try_from(self.u64()?).context("GGUF string too large")?;
        std::str::from_utf8(self.take(len)?).context("invalid UTF-8 in GGUF metadata")
    }
    fn value(&mut self, kind: u32, keep: bool, depth: usize) -> Result<Value> {
        ensure!(depth <= 8, "GGUF metadata nesting exceeds limit");
        Ok(match kind {
            0 => Value::from(self.take(1)?[0]),
            1 => Value::from(self.take(1)?[0] as i8),
            2 => Value::from(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => Value::from(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => Value::from(self.u32()?),
            5 => Value::from(self.u32()? as i32),
            6 => Value::from(f32::from_le_bytes(self.take(4)?.try_into()?)),
            7 => Value::from(self.take(1)?[0] != 0),
            8 => {
                let value = self.string()?;
                if keep {
                    Value::from(value)
                } else {
                    Value::Null
                }
            }
            9 => {
                let subtype = self.u32()?;
                let count = self.u64()?;
                ensure!(count <= 100_000_000, "GGUF array exceeds metadata limit");
                // Tokenizer arrays are irrelevant to memory estimates; skip without allocating.
                for _ in 0..count {
                    self.value(subtype, false, depth + 1)?;
                }
                Value::Null
            }
            10 => Value::from(self.u64()?),
            11 => Value::from(self.u64()? as i64),
            12 => Value::from(f64::from_le_bytes(self.take(8)?.try_into()?)),
            _ => bail!("unsupported GGUF metadata type {kind}"),
        })
    }
}

/// Parse GGUF v2/v3 little-endian metadata, stopping before tensor data.
pub fn parse_header(bytes: &[u8]) -> Result<(Metadata, BTreeMap<String, Value>)> {
    let mut reader = Reader { bytes, offset: 0 };
    ensure!(
        reader.take(4)? == b"GGUF",
        "invalid GGUF magic (only little-endian GGUF is supported)"
    );
    let version = reader.u32()?;
    ensure!(
        matches!(version, 2 | 3),
        "unsupported GGUF version {version}"
    );
    let tensor_count = reader.u64()?;
    let field_count = reader.u64()?;
    ensure!(
        tensor_count <= 10_000_000 && field_count <= 10_000_000,
        "GGUF metadata count exceeds limit"
    );
    let mut fields = BTreeMap::new();
    for _ in 0..field_count {
        let key = reader.string()?.to_owned();
        let kind = reader.u32()?;
        let keep = !key.starts_with("tokenizer.");
        let value = reader.value(kind, keep, 0)?;
        if keep {
            ensure!(
                fields.insert(key, value).is_none(),
                "duplicate GGUF metadata key"
            );
        }
    }
    let mut component = Component::default();
    let mut names = std::collections::BTreeSet::new();
    for _ in 0..tensor_count {
        let name = reader.string()?;
        ensure!(names.insert(name), "duplicate GGUF tensor {name}");
        let ndims = reader.u32()?;
        ensure!((1..=4).contains(&ndims), "invalid GGUF tensor dimensions");
        let dims: Vec<u64> = (0..ndims).map(|_| reader.u64()).collect::<Result<_>>()?;
        ensure!(
            dims.iter().all(|dim| *dim > 0),
            "GGUF tensor dimensions must be positive"
        );
        let kind = reader.u32()?;
        let _offset = reader.u64()?;
        let (_, dtype, block, block_bytes) = DTYPES
            .iter()
            .find(|(id, ..)| *id == kind)
            .with_context(|| format!("unsupported GGUF tensor type {kind}"))?;
        ensure!(
            dims[0] % block == 0,
            "GGUF tensor {name} row is not aligned to {dtype} block size"
        );
        let params = product(&dims)?;
        let bytes = product(&[params / block, *block_bytes])?;
        component.accumulate(dtype, params, bytes)?;
    }
    let mut metadata = Metadata::default();
    metadata.merge_component("Transformer", &component)?;
    Ok((metadata, fields))
}

pub fn cache(
    fields: &BTreeMap<String, Value>,
    dtype: &str,
    length: Option<u64>,
    batch: u64,
) -> Result<KvCache> {
    let architecture = fields.get("general.architecture").and_then(Value::as_str);
    let field = |suffix: &str| -> Result<Option<u64>> {
        let value = if let Some(arch) = architecture {
            fields.get(&format!("{arch}.{suffix}"))
        } else {
            let matches: Vec<_> = fields
                .iter()
                .filter(|(k, _)| k.ends_with(&format!(".{suffix}")))
                .collect();
            ensure!(matches.len() <= 1, "ambiguous GGUF cache field {suffix}");
            matches.first().map(|(_, v)| *v)
        };
        value
            .map(|v| {
                v.as_u64()
                    .filter(|n| *n > 0)
                    .with_context(|| format!("GGUF {suffix} must be a positive integer"))
            })
            .transpose()
    };
    let required =
        |suffix: &str| field(suffix)?.with_context(|| format!("missing GGUF cache field {suffix}"));
    let layers = required("block_count")?;
    let heads = required("attention.head_count")?;
    let kv_heads = field("attention.head_count_kv")?.unwrap_or(heads);
    let embedding = required("embedding_length")?;
    let length = match length {
        Some(length) => length,
        None => required("context_length")?,
    };
    ensure!(
        length > 0 && batch > 0,
        "context length and batch size must be positive"
    );
    ensure!(
        heads >= kv_heads && heads % kv_heads == 0,
        "invalid GGUF key/value head count"
    );
    let key_dim = field("attention.key_length")?.unwrap_or(embedding / heads);
    let value_dim = field("attention.value_length")?.unwrap_or(embedding / heads);
    ensure!(key_dim > 0 && value_dim > 0, "invalid GGUF head dimension");
    let dtype = if dtype == "auto" { "F16" } else { dtype };
    let (_, _, block, block_bytes) = DTYPES
        .iter()
        .find(|(_, name, ..)| *name == dtype)
        .context("invalid GGUF cache dtype")?;
    let elements = product(&[
        layers,
        kv_heads,
        crate::types::add(key_dim, value_dim)?,
        length,
        batch,
    ])?;
    let bytes = product(&[elements, *block_bytes])?.div_ceil(*block);
    Ok(KvCache {
        bytes,
        dtype: dtype.into(),
        max_model_len: length,
        batch_size: batch,
    })
}

/// Return the logical filename and shard index/count, retaining the repository directory.
pub fn shard(path: &str) -> Option<(String, u64, u64)> {
    let (prefix, total) = path.strip_suffix(".gguf")?.rsplit_once("-of-")?;
    let (prefix, index) = prefix.rsplit_once('-')?;
    if !index.bytes().all(|c| c.is_ascii_digit()) || !total.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((
        format!("{prefix}.gguf"),
        index.parse().ok()?,
        total.parse().ok()?,
    ))
}
