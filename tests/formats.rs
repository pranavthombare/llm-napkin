mod common;
use llm_napkin::{gguf, kv_cache, safetensors, types::Metadata};
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn safetensors_mixed_scalar_empty_and_bool() {
    let raw = common::safetensors(&[
        ("matrix", "BF16", &[3, 4], 24),
        ("scalar", "F32", &[], 4),
        ("empty", "F16", &[0, 4], 0),
        ("flag", "BOOL", &[1], 1),
    ]);
    let length = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let tensors = safetensors::parse_header(&raw[8..8 + length], Some(raw.len() as u64)).unwrap();
    let component = safetensors::component(&tensors).unwrap();
    assert_eq!(component.bytes, 29);
    assert_eq!(component.param_count, 14);
    assert_eq!(component.dtypes["BF16"].bytes, 24);
}

#[test]
fn safetensors_rejects_corrupt_offsets_and_overflow() {
    for header in [
        json!({"x":{"dtype":"F16", "shape":[8], "data_offsets":[0,12]}}),
        json!({"x":{"dtype":"F16", "shape":[u64::MAX,2], "data_offsets":[0,4]}}),
        json!({"x":{"dtype":"UNKNOWN", "shape":[2], "data_offsets":[0,4]}}),
        json!({"x":{"dtype":"F16", "shape":[2], "data_offsets":[4,8]}}),
    ] {
        assert!(safetensors::parse_header(&serde_json::to_vec(&header).unwrap(), None).is_err());
    }
}

#[test]
fn safetensors_detects_truncated_file() {
    let raw = common::safetensors(&[("w", "F16", &[4], 8)]);
    let size = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    assert!(safetensors::parse_header(&raw[8..8 + size], Some(raw.len() as u64 - 1)).is_err());
}

#[test]
fn gguf_exact_quantization_blocks() {
    let raw = common::gguf(
        &[("a", 12, &[256, 2]), ("b", 15, &[256]), ("c", 22, &[256])],
        true,
        0,
    );
    let (metadata, fields) = gguf::parse_header(&raw).unwrap();
    assert_eq!(metadata.bytes, 288 + 292 + 82);
    assert_eq!(metadata.components["Transformer"].param_count, 1024);
    let cache = gguf::cache(&fields, "auto", None, 1).unwrap();
    assert_eq!(cache.bytes, 32768);
    assert_eq!(
        gguf::cache(&fields, "F16", Some(128), 2).unwrap().bytes,
        131072
    );
}

#[test]
fn gguf_truncation_is_distinct_from_invalid_data() {
    let raw = common::gguf(&[("w", 1, &[32])], false, 0);
    for n in 0..raw.len() {
        assert!(
            gguf::parse_header(&raw[..n])
                .unwrap_err()
                .is::<gguf::Truncated>()
        );
    }
    let mut invalid = raw.clone();
    invalid[0] = b'X';
    assert!(
        !gguf::parse_header(&invalid)
            .unwrap_err()
            .is::<gguf::Truncated>()
    );
    invalid = raw;
    invalid[4] = 1;
    assert!(gguf::parse_header(&invalid).is_err());
    assert!(gguf::parse_header(&common::gguf(&[("w", 999, &[32])], false, 0)).is_err());
    assert!(gguf::parse_header(&common::gguf(&[("w", 12, &[255])], false, 0)).is_err());
}

fn config() -> serde_json::Value {
    json!({"hidden_size":128,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":2,"torch_dtype":"bfloat16"})
}

#[test]
fn gqa_explicit_head_dimension_and_batch() {
    let mut config = config();
    assert_eq!(
        kv_cache::safetensors_cache(&config, "BF16", 64, 1)
            .unwrap()
            .bytes,
        32768
    );
    config["head_dim"] = json!(64);
    assert_eq!(
        kv_cache::safetensors_cache(&config, "F8_E4M3", 64, 2)
            .unwrap()
            .bytes,
        65536
    );
}

#[test]
fn sliding_and_hybrid_reservations() {
    let mut config = config();
    config["sliding_window"] = json!(16);
    assert_eq!(
        kv_cache::safetensors_cache(&config, "BF16", 64, 1)
            .unwrap()
            .bytes,
        8192
    );
    config["layer_types"] = json!(["full_attention", "sliding_attention"]);
    assert_eq!(
        kv_cache::safetensors_cache(&config, "BF16", 64, 1)
            .unwrap()
            .bytes,
        32768
    );
    config["sliding_window_pattern"] = json!(2);
    assert_eq!(
        kv_cache::safetensors_cache(&config, "BF16", 64, 1)
            .unwrap()
            .bytes,
        32768
    );
}

#[test]
fn invalid_cache_dimensions_and_unsupported_layers() {
    let mut config = config();
    assert!(kv_cache::safetensors_cache(&config, "F16", u64::MAX, 2).is_err());
    for key in [
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_size",
    ] {
        let mut invalid = config.clone();
        invalid[key] = json!(0);
        assert!(kv_cache::safetensors_cache(&invalid, "F16", 64, 1).is_err());
    }
    config["layer_types"] = json!(["linear_attention", "full_attention"]);
    assert!(kv_cache::safetensors_cache(&config, "F16", 64, 1).is_err());
}

#[test]
fn cache_dtype_resolution() {
    let mut config = config();
    let metadata = Metadata::default();
    assert_eq!(
        kv_cache::resolve_dtype(&config, "auto", &metadata).unwrap(),
        "BF16"
    );
    assert_eq!(
        kv_cache::resolve_dtype(&config, "fp8_e5m2", &metadata).unwrap(),
        "F8_E5M2"
    );
    config["quantization_config"] = json!({"quant_method":"fp8","fmt":"e4m3"});
    assert_eq!(
        kv_cache::resolve_dtype(&config, "auto", &metadata).unwrap(),
        "F8_E4M3"
    );
    config["quantization_config"] =
        json!({"quant_method":"modelopt","kv_cache_scheme":{"num_bits":8,"type":"float"}});
    assert_eq!(
        kv_cache::resolve_dtype(&config, "auto", &metadata).unwrap(),
        "F8_E4M3"
    );
    config["quantization_config"] = json!({"quant_method":"gptq"});
    assert!(kv_cache::resolve_dtype(&config, "auto", &metadata).is_err());
    assert_eq!(
        kv_cache::resolve_dtype(&config, "bfloat16", &metadata).unwrap(),
        "BF16"
    );
}

#[test]
fn moe_accounting_and_active_one() {
    let raw = common::safetensors(&[
        ("embed", "F32", &[4], 16),
        ("layer.experts.0.w", "F16", &[8], 16),
        ("layer.experts.1.w", "F16", &[8], 16),
    ]);
    let size = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let tensors = safetensors::parse_header(&raw[8..8 + size], None).unwrap();
    let components = BTreeMap::from([("Transformer".into(), tensors)]);
    let config = json!({"num_local_experts":2,"num_experts_per_tok":1});
    let moe = safetensors::moe_metadata(&components, &config)
        .unwrap()
        .unwrap();
    assert_eq!(moe.base_model.bytes, 16);
    assert_eq!(moe.experts_total.bytes, 32);
    assert_eq!(moe.active_expert_count, Some(1));
    assert!(safetensors::moe_metadata(&components, &json!({"num_local_experts":3})).is_err());
}

#[test]
fn shard_names_retain_directories() {
    assert_eq!(
        gguf::shard("Q4/model-00001-of-00003.gguf"),
        Some(("Q4/model.gguf".into(), 1, 3))
    );
    assert!(gguf::shard("model-Q4_K_M.gguf").is_none());
}
