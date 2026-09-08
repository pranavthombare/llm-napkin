use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail, ensure};
use futures_util::{StreamExt, TryStreamExt, stream};
use serde_json::Value;

use crate::{
    gguf,
    hub::{Hub, HubFile, MAX_METADATA, valid_path},
    kv_cache,
    safetensors::{self, Tensor},
    types::{Estimate, Metadata},
};

/// Options for a single estimate. Tokens are deliberately excluded from Debug output.
#[derive(Clone)]
pub struct Options {
    pub model_id: String,
    pub revision: String,
    pub hf_token: Option<String>,
    pub experimental: bool,
    pub max_model_len: Option<u64>,
    pub batch_size: u64,
    pub kv_cache_dtype: String,
    pub gguf_file: Option<String>,
    pub endpoint: String,
    pub max_workers: usize,
}

impl Options {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            revision: "main".into(),
            hf_token: None,
            experimental: false,
            max_model_len: None,
            batch_size: 1,
            kv_cache_dtype: "auto".into(),
            gguf_file: None,
            endpoint: "https://huggingface.co".into(),
            max_workers: 8,
        }
    }
}

async fn component_files(
    hub: &Hub,
    options: &Options,
    paths: &BTreeSet<&str>,
    dir: &str,
    stems: &[&str],
) -> Result<Vec<String>> {
    let prefix = if dir.is_empty() {
        String::new()
    } else {
        format!("{dir}/")
    };
    for stem in stems {
        let file = format!("{prefix}{stem}.safetensors");
        if paths.contains(file.as_str()) {
            return Ok(vec![file]);
        }
        let index = format!("{file}.index.json");
        if paths.contains(index.as_str()) {
            let json = hub
                .json(&options.model_id, &options.revision, &index)
                .await?;
            let weights = json["weight_map"]
                .as_object()
                .context("Safetensors index has no weight_map")?;
            ensure!(!weights.is_empty(), "Safetensors index is empty");
            return weights
                .values()
                .map(|value| {
                    let shard = value.as_str().context("invalid shard in weight_map")?;
                    ensure!(
                        valid_path(shard) && shard.ends_with(".safetensors"),
                        "invalid Safetensors shard path"
                    );
                    let shard = format!("{prefix}{shard}");
                    ensure!(
                        paths.contains(shard.as_str()),
                        "index references missing shard {shard}"
                    );
                    Ok(shard)
                })
                .collect::<Result<BTreeSet<_>>>()
                .map(|set| set.into_iter().collect());
        }
    }
    Ok(Vec::new())
}

async fn discover_safetensors(
    hub: &Hub,
    options: &Options,
    paths: &BTreeSet<&str>,
) -> Result<BTreeMap<String, Vec<String>>> {
    let mut components = BTreeMap::new();
    let root = component_files(hub, options, paths, "", &["model"]).await?;
    if !root.is_empty() {
        let sentence =
            paths.contains("config_sentence_transformers.json") || paths.contains("modules.json");
        components.insert(
            if sentence {
                "0_Transformer"
            } else {
                "Transformer"
            }
            .into(),
            root,
        );
        if sentence && paths.contains("modules.json") {
            let modules = hub
                .json(&options.model_id, &options.revision, "modules.json")
                .await?;
            for module in modules
                .as_array()
                .context("modules.json must contain an array")?
            {
                if module["type"] == "sentence_transformers.models.Dense" {
                    let path = module["path"]
                        .as_str()
                        .context("Dense module has no path")?;
                    ensure!(valid_path(path), "invalid Dense module path");
                    let files = component_files(hub, options, paths, path, &["model"]).await?;
                    ensure!(
                        !files.is_empty(),
                        "Dense module {path} has no Safetensors weights"
                    );
                    components.insert(path.into(), files);
                }
            }
        }
    } else if paths.contains("model_index.json") {
        let index = hub
            .json(&options.model_id, &options.revision, "model_index.json")
            .await?;
        for name in index
            .as_object()
            .context("model_index.json must contain an object")?
            .keys()
            .filter(|name| !name.starts_with('_'))
        {
            ensure!(valid_path(name), "invalid diffusion component path");
            let files = component_files(
                hub,
                options,
                paths,
                name,
                &["diffusion_pytorch_model", "model"],
            )
            .await?;
            if !files.is_empty() {
                components.insert(name.into(), files);
            }
        }
    } else {
        // Other libraries may use arbitrary tensor filenames; group them by directory.
        for path in paths.iter().filter(|path| path.ends_with(".safetensors")) {
            let name = path.rsplit_once('/').map_or("Transformer", |(dir, _)| dir);
            components
                .entry(name.into())
                .or_insert_with(Vec::new)
                .push((*path).into());
        }
    }
    ensure!(
        !components.is_empty(),
        "no supported Safetensors or GGUF weights found in this repository"
    );
    Ok(components)
}

async fn fetch_safetensors(
    hub: &Hub,
    options: &Options,
    path: &str,
    file_size: Option<u64>,
) -> Result<Vec<Tensor>> {
    let prefix = hub
        .prefix(&options.model_id, &options.revision, path, 8)
        .await?;
    let size = u64::from_le_bytes(
        prefix
            .as_slice()
            .try_into()
            .context("truncated Safetensors length prefix")?,
    );
    ensure!(
        size > 0 && size <= MAX_METADATA as u64,
        "invalid Safetensors header length or header exceeds 100 MB"
    );
    ensure!(
        file_size.is_none_or(|file_size| size + 8 <= file_size),
        "Safetensors header extends past the end of the file"
    );
    let raw = hub
        .prefix(
            &options.model_id,
            &options.revision,
            path,
            size as usize + 8,
        )
        .await?;
    ensure!(
        raw.len() == size as usize + 8,
        "truncated Safetensors header"
    );
    ensure!(
        raw[..8] == prefix,
        "Safetensors length changed between requests"
    );
    safetensors::parse_header(&raw[8..], file_size).with_context(|| format!("parsing {path}"))
}

async fn estimate_safetensors(
    hub: &Hub,
    options: &Options,
    files: &[HubFile],
    result: &mut Estimate,
) -> Result<()> {
    ensure!(
        kv_cache::SAFETENSORS_DTYPES.contains(&options.kv_cache_dtype.as_str()),
        "invalid --kv-cache-dtype for Safetensors"
    );
    let paths: BTreeSet<&str> = files.iter().map(|file| file.path.as_str()).collect();
    let components = discover_safetensors(hub, options, &paths).await?;
    let sizes: BTreeMap<_, _> = files
        .iter()
        .map(|file| (file.path.as_str(), file.size))
        .collect();
    let requests: Vec<_> = components
        .iter()
        .flat_map(|(name, paths)| paths.iter().map(move |path| (name, path)))
        .collect();
    let fetched: Vec<_> = stream::iter(requests)
        .map(|(name, path)| {
            let size = sizes.get(path.as_str()).copied().flatten();
            async move {
                Ok::<_, anyhow::Error>((
                    name.clone(),
                    fetch_safetensors(hub, options, path, size).await?,
                ))
            }
        })
        .buffer_unordered(options.max_workers)
        .try_collect()
        .await?;
    let mut raw: BTreeMap<String, Vec<Tensor>> = BTreeMap::new();
    for (name, tensors) in fetched {
        raw.entry(name).or_default().extend(tensors);
    }
    let mut metadata = Metadata::default();
    for (name, tensors) in &raw {
        let mut names = BTreeSet::new();
        ensure!(
            tensors.iter().all(|t| names.insert(t.name.as_str())),
            "duplicate tensor names across shards in {name}; repository may contain alternative weight variants"
        );
        metadata.merge_component(name, &safetensors::component(tensors)?)?;
    }
    if options.experimental {
        if paths.contains("config.json") {
            let config = hub
                .json(&options.model_id, &options.revision, "config.json")
                .await?;
            let supported = config["architectures"]
                .as_array()
                .is_some_and(|architectures| {
                    architectures.iter().any(|arch| {
                        arch.as_str().is_some_and(|arch| {
                            arch.contains("ForCausalLM")
                                || arch.contains("ForConditionalGeneration")
                        })
                    })
                });
            if supported {
                let mut text_config = config
                    .get("text_config")
                    .cloned()
                    .unwrap_or_else(|| config.clone());
                if let Some(reference) = text_config["_name_or_path"]
                    .as_str()
                    .filter(|r| valid_path(r) && r.split('/').count() == 2)
                {
                    if ["hidden_size", "num_attention_heads", "num_hidden_layers"]
                        .iter()
                        .any(|key| text_config.get(*key).is_none())
                    {
                        let mut referenced = hub
                            .json(reference, &options.revision, "config.json")
                            .await?;
                        referenced
                            .as_object_mut()
                            .context("referenced config must be an object")?
                            .extend(
                                text_config
                                    .as_object()
                                    .context("text_config must be an object")?
                                    .clone(),
                            );
                        text_config = referenced;
                    }
                }
                for key in ["dtype", "torch_dtype", "quantization_config"] {
                    if text_config.get(key).is_none() && config.get(key).is_some() {
                        text_config[key] = config[key].clone();
                    }
                }
                result.moe = safetensors::moe_metadata(&raw, &text_config)?;
                let length = options.max_model_len.or_else(|| {
                    ["max_position_embeddings", "n_positions", "max_seq_len"]
                        .iter()
                        .find_map(|key| text_config[*key].as_u64())
                });
                if let Some(length) = length {
                    if ["hidden_size", "num_hidden_layers", "num_attention_heads"]
                        .iter()
                        .all(|key| text_config.get(*key).is_some())
                    {
                        if text_config.get("kv_lora_rank").is_some() {
                            result.warnings.push("MLA model: the cache estimate uses an uncompressed MHA/GQA upper bound; engines with latent-cache compression can use less memory.".into());
                        }
                        let dtype = kv_cache::resolve_dtype(
                            &text_config,
                            &options.kv_cache_dtype,
                            &metadata,
                        )?;
                        match kv_cache::safetensors_cache(
                            &text_config,
                            &dtype,
                            length,
                            options.batch_size,
                        ) {
                            Ok(cache) => metadata.kv_cache = Some(cache),
                            Err(error) if error.to_string().contains("nonstandard layer types") => {
                                result
                                    .warnings
                                    .push(format!("KV cache unavailable: {error}"))
                            }
                            Err(error) => return Err(error),
                        }
                    } else {
                        result.warnings.push(
                            "KV cache unavailable: config is missing attention dimensions.".into(),
                        );
                    }
                } else {
                    result.warnings.push("KV cache unavailable: specify --max-model-len; config has no context length.".into());
                }
            } else {
                result.warnings.push("KV cache unavailable: architecture is not ForCausalLM or ForConditionalGeneration.".into());
            }
        } else {
            result
                .warnings
                .push("KV cache unavailable: repository has no config.json.".into());
        }
    }
    result.safetensors = Some(metadata);
    Ok(())
}

async fn fetch_gguf(
    hub: &Hub,
    options: &Options,
    path: &str,
) -> Result<(Metadata, BTreeMap<String, Value>)> {
    let mut size = 1_000_000;
    loop {
        let raw = hub
            .prefix(&options.model_id, &options.revision, path, size)
            .await?;
        match gguf::parse_header(&raw) {
            Ok(metadata) => return Ok(metadata),
            Err(error)
                if error.is::<gguf::Truncated>() && raw.len() == size && size < MAX_METADATA =>
            {
                size = (size * 2).min(MAX_METADATA);
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("parsing {path} (GGUF metadata is limited to 100 MB)")
                });
            }
        }
    }
}

async fn estimate_gguf(
    hub: &Hub,
    options: &Options,
    paths: Vec<&str>,
    result: &mut Estimate,
) -> Result<()> {
    ensure!(
        options.kv_cache_dtype == "auto"
            || gguf::DTYPES
                .iter()
                .any(|(_, dtype, ..)| *dtype == options.kv_cache_dtype),
        "invalid --kv-cache-dtype for GGUF"
    );
    let mut groups: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for path in paths {
        let name = gguf::shard(path).map_or_else(|| path.to_owned(), |(name, ..)| name);
        groups.entry(name).or_default().push(path);
    }
    if let Some(requested) = &options.gguf_file {
        let matches: Vec<_> = groups
            .iter()
            .filter(|(name, shards)| {
                name.as_str() == requested
                    || shards.contains(&requested.as_str())
                    || (!requested.contains('/')
                        && (name.rsplit('/').next() == Some(requested.as_str())
                            || shards
                                .iter()
                                .any(|p| p.rsplit('/').next() == Some(requested.as_str()))))
            })
            .map(|(name, _)| name.clone())
            .collect();
        ensure!(
            matches.len() == 1,
            "expected one GGUF match for {requested}, found {}; use the full repository path",
            matches.len()
        );
        groups.retain(|name, _| name == &matches[0]);
        result.filename = Some(matches[0].clone());
    }
    ensure!(!groups.is_empty(), "no matching GGUF files found");
    for (name, paths) in &groups {
        let shards: Vec<_> = paths.iter().filter_map(|path| gguf::shard(path)).collect();
        if let Some((_, _, total)) = shards.first() {
            let ids: BTreeSet<_> = shards.iter().map(|(_, index, _)| *index).collect();
            ensure!(
                *total > 0
                    && *total == paths.len() as u64
                    && ids.len() == paths.len()
                    && ids.iter().copied().eq(1..=*total)
                    && shards.iter().all(|(_, _, count)| count == total),
                "missing or inconsistent GGUF shards for {name}"
            );
        }
    }
    let requests: Vec<_> = groups
        .iter()
        .flat_map(|(name, paths)| paths.iter().map(move |path| (name, *path)))
        .collect();
    let fetched: Vec<_> = stream::iter(requests)
        .map(|(name, path)| async move {
            let (mut metadata, fields) = fetch_gguf(hub, options, path).await?;
            if options.experimental && gguf::shard(path).is_none_or(|(_, index, _)| index == 1) {
                metadata.kv_cache = Some(gguf::cache(
                    &fields,
                    &options.kv_cache_dtype,
                    options.max_model_len,
                    options.batch_size,
                )?);
            }
            Ok::<_, anyhow::Error>((name.clone(), metadata))
        })
        .buffer_unordered(options.max_workers)
        .try_collect()
        .await?;
    for (name, metadata) in fetched {
        result
            .gguf_files
            .entry(name)
            .or_default()
            .merge(&metadata)?;
    }
    Ok(())
}

/// Discover model weights, read their headers concurrently, and estimate inference memory.
pub async fn estimate(options: &Options) -> Result<Estimate> {
    ensure!(
        options.batch_size > 0 && options.max_model_len != Some(0),
        "batch size and context length must be positive"
    );
    ensure!(
        (1..=128).contains(&options.max_workers),
        "max_workers must be between 1 and 128"
    );
    if let Some(file) = &options.gguf_file {
        ensure!(
            valid_path(file) && file.ends_with(".gguf"),
            "--gguf-file must be a repository GGUF path"
        );
    }
    let hub = Hub::new(&options.endpoint, options.hf_token.as_deref()).await?;
    let files = hub.files(&options.model_id, &options.revision).await?;
    let gguf_paths: Vec<_> = files
        .iter()
        .filter(|f| f.path.ends_with(".gguf") && !f.path.contains("mmproj-"))
        .map(|f| f.path.as_str())
        .collect();
    let has_safe = files
        .iter()
        .any(|f| f.path.ends_with(".safetensors") || f.path.ends_with(".safetensors.index.json"));
    let mut result = Estimate {
        model_id: options.model_id.clone(),
        revision: options.revision.clone(),
        filename: None,
        safetensors: None,
        gguf_files: BTreeMap::new(),
        moe: None,
        warnings: Vec::new(),
    };
    if options.gguf_file.is_some() || (!has_safe && !gguf_paths.is_empty()) {
        estimate_gguf(&hub, options, gguf_paths, &mut result).await?;
    } else {
        if !gguf_paths.is_empty() {
            result
                .warnings
                .push("Repository also contains GGUF files; select one with --gguf-file.".into());
        }
        estimate_safetensors(&hub, options, &files, &mut result).await?;
    }
    if result.safetensors.is_none() && result.gguf_files.is_empty() {
        bail!("no model weights found");
    }
    result.total_memory()?;
    Ok(result)
}
