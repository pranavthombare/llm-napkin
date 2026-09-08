mod common;
use std::collections::BTreeMap;

use common::{TestHub, gguf, safetensors};
use llm_napkin::estimate;
use serde_json::json;

fn config() -> Vec<u8> {
    serde_json::to_vec(&json!({"architectures":["TestForCausalLM"],"hidden_size":128,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":2,"torch_dtype":"bfloat16","max_position_embeddings":64})).unwrap()
}

#[tokio::test]
async fn safetensors_pagination_revision_auth_and_json() {
    let server = TestHub::new(
        BTreeMap::from([
            ("config.json".into(), config()),
            (
                "model.safetensors".into(),
                safetensors(&[("w", "F16", &[8], 16)]),
            ),
        ]),
        true,
        false,
    )
    .await;
    let mut options = server.options();
    options.experimental = true;
    options.revision = "refs/pr/4".into();
    let result = estimate(&options).await.unwrap();
    assert_eq!(
        result.to_json(false).unwrap(),
        json!({"model_id":"test/model","memory":16,"kv_cache":32768,"total_memory":32784})
    );
    let details = result.to_json(true).unwrap();
    assert_eq!(
        details["memory"]["components"]["Transformer"]["dtypes"]["F16"]["param_count"],
        8
    );
    assert_eq!(details["kv_cache"]["dtype"], "BF16");
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 5);
    assert!(
        requests
            .iter()
            .all(|request| request.contains("refs%2Fpr%2F4"))
    );
    assert!(
        requests
            .iter()
            .all(|request| request.contains("authorization: Bearer test-token"))
    );
    assert!(
        requests
            .iter()
            .any(|request| request.contains("range: bytes=0-7"))
    );
}

#[tokio::test]
async fn server_ignoring_range_and_large_header() {
    let name = "w".repeat(120_000);
    let server = TestHub::new(
        BTreeMap::from([(
            "model.safetensors".into(),
            safetensors(&[(&name, "F16", &[8192], 16384)]),
        )]),
        false,
        true,
    )
    .await;
    let result = estimate(&server.options()).await.unwrap();
    assert_eq!(result.total_memory().unwrap(), Some(16384));
    assert_eq!(server.requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn sharded_safetensors_deduplicates_index_files() {
    let server = TestHub::new(BTreeMap::from([
        ("model.safetensors.index.json".into(),serde_json::to_vec(&json!({"weight_map":{"a":"part1.safetensors","b":"part1.safetensors","c":"part2.safetensors"}})).unwrap()),
        ("part1.safetensors".into(),safetensors(&[("a","F16",&[8],16),("b","F32",&[2],8)])),
        ("part2.safetensors".into(),safetensors(&[("c","F16",&[4],8)])),
    ]),false,false).await;
    let result = estimate(&server.options()).await.unwrap();
    assert_eq!(result.total_memory().unwrap(), Some(32));
    assert_eq!(server.requests.lock().unwrap().len(), 6);
}

#[tokio::test]
async fn sentence_transformer_dense_module() {
    let server = TestHub::new(BTreeMap::from([
        ("model.safetensors".into(),safetensors(&[("w","F16",&[8],16)])),
        ("modules.json".into(),serde_json::to_vec(&json!([{"type":"sentence_transformers.models.Transformer","path":""},{"type":"sentence_transformers.models.Dense","path":"2_Dense"}])).unwrap()),
        ("2_Dense/model.safetensors".into(),safetensors(&[("w","F32",&[4],16)])),
    ]),false,false).await;
    let result = estimate(&server.options()).await.unwrap();
    let metadata = result.safetensors.unwrap();
    assert_eq!(metadata.bytes, 32);
    assert_eq!(metadata.components.len(), 2);
    assert_eq!(metadata.components["0_Transformer"].bytes, 16);
}

#[tokio::test]
async fn diffusion_components_prefer_default_variant() {
    let server = TestHub::new(BTreeMap::from([
        ("model_index.json".into(),serde_json::to_vec(&json!({"_class_name":"TestPipeline","unet":["diffusers","UNet"],"text_encoder":["transformers","CLIP"],"scheduler":["diffusers","DDIM"]})).unwrap()),
        ("unet/diffusion_pytorch_model.safetensors".into(),safetensors(&[("w","F16",&[8],16)])),
        ("unet/diffusion_pytorch_model.fp16.safetensors".into(),safetensors(&[("w","F16",&[8],16)])),
        ("text_encoder/model.safetensors".into(),safetensors(&[("w","F32",&[4],16)])),
    ]),false,false).await;
    let result = estimate(&server.options()).await.unwrap();
    assert_eq!(result.total_memory().unwrap(), Some(32));
    assert_eq!(result.safetensors.unwrap().components.len(), 2);
}

#[tokio::test]
async fn gguf_variants_shards_cache_and_multimodal_projection() {
    let server = TestHub::new(
        BTreeMap::from([
            (
                "Q4/model-00001-of-00002.gguf".into(),
                gguf(&[("a", 12, &[256])], true, 0),
            ),
            (
                "Q4/model-00002-of-00002.gguf".into(),
                gguf(&[("b", 12, &[256])], false, 0),
            ),
            ("model-f16.gguf".into(), gguf(&[("a", 1, &[256])], true, 0)),
            ("mmproj-f16.gguf".into(), vec![0; 16]),
        ]),
        false,
        false,
    )
    .await;
    let mut options = server.options();
    options.experimental = true;
    let result = estimate(&options).await.unwrap();
    let output = result.to_json(false).unwrap();
    assert_eq!(
        output["memory"],
        json!({"Q4/model.gguf":288,"model-f16.gguf":512})
    );
    assert_eq!(
        output["kv_cache"],
        json!({"Q4/model.gguf":32768,"model-f16.gguf":32768})
    );
    assert!(output["total_memory"].is_null());
    options.gguf_file = Some("Q4/model-00002-of-00002.gguf".into());
    let result = estimate(&options).await.unwrap();
    assert_eq!(result.filename.as_deref(), Some("Q4/model.gguf"));
    assert_eq!(result.total_memory().unwrap(), Some(33056));
    assert_eq!(result.to_json(true).unwrap()["memory"]["bytes"], 288);
}

#[tokio::test]
async fn gguf_large_metadata_expands_and_invalid_data_fails_fast() {
    let server = TestHub::new(
        BTreeMap::from([(
            "model.gguf".into(),
            gguf(&[("w", 1, &[32])], false, 1_100_000),
        )]),
        false,
        false,
    )
    .await;
    let result = estimate(&server.options()).await.unwrap();
    assert_eq!(result.gguf_files["model.gguf"].bytes, 64);
    assert_eq!(server.requests.lock().unwrap().len(), 3);
    let invalid = TestHub::new(
        BTreeMap::from([("model.gguf".into(), vec![0; 1_100_000])]),
        false,
        false,
    )
    .await;
    assert!(
        estimate(&invalid.options())
            .await
            .unwrap_err()
            .to_string()
            .contains("parsing model.gguf")
    );
    assert_eq!(invalid.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn missing_shards_and_ambiguous_names_fail_before_fetching() {
    for files in [
        BTreeMap::from([(
            "model-00001-of-00002.gguf".into(),
            gguf(&[("w", 1, &[32])], false, 0),
        )]),
        BTreeMap::from([
            ("a/model.gguf".into(), gguf(&[("w", 1, &[32])], false, 0)),
            ("b/model.gguf".into(), gguf(&[("w", 1, &[32])], false, 0)),
        ]),
    ] {
        let server = TestHub::new(files, false, false).await;
        let mut options = server.options();
        options.gguf_file = Some("model.gguf".into());
        assert!(estimate(&options).await.is_err());
        assert_eq!(server.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn safetensors_preferred_but_gguf_can_be_selected() {
    let server = TestHub::new(
        BTreeMap::from([
            (
                "model.safetensors".into(),
                safetensors(&[("w", "F16", &[8], 16)]),
            ),
            ("model.gguf".into(), gguf(&[("w", 1, &[32])], false, 0)),
        ]),
        false,
        false,
    )
    .await;
    let mut options = server.options();
    assert!(estimate(&options).await.unwrap().safetensors.is_some());
    options.gguf_file = Some("model.gguf".into());
    assert_eq!(
        estimate(&options).await.unwrap().total_memory().unwrap(),
        Some(64)
    );
}

#[tokio::test]
async fn nested_vlm_configuration_inherits_dtype() {
    let mut config: serde_json::Value = serde_json::from_slice(&config()).unwrap();
    config.as_object_mut().unwrap().remove("torch_dtype");
    let server = TestHub::new(BTreeMap::from([
        ("config.json".into(),serde_json::to_vec(&json!({"architectures":["TestForConditionalGeneration"],"torch_dtype":"bfloat16","text_config":config})).unwrap()),
        ("model.safetensors".into(),safetensors(&[("w","F16",&[8],16)])),
    ]),false,false).await;
    let mut options = server.options();
    options.experimental = true;
    assert_eq!(
        estimate(&options).await.unwrap().total_memory().unwrap(),
        Some(32784)
    );
}

#[tokio::test]
async fn missing_context_preserves_weights_and_warns() {
    let mut config: serde_json::Value = serde_json::from_slice(&config()).unwrap();
    config
        .as_object_mut()
        .unwrap()
        .remove("max_position_embeddings");
    let server = TestHub::new(
        BTreeMap::from([
            ("config.json".into(), serde_json::to_vec(&config).unwrap()),
            (
                "model.safetensors".into(),
                safetensors(&[("w", "F16", &[8], 16)]),
            ),
        ]),
        false,
        false,
    )
    .await;
    let mut options = server.options();
    options.experimental = true;
    let result = estimate(&options).await.unwrap();
    assert_eq!(result.total_memory().unwrap(), Some(16));
    assert_eq!(result.warnings.len(), 1);
    options.max_model_len = Some(64);
    assert_eq!(
        estimate(&options).await.unwrap().total_memory().unwrap(),
        Some(32784)
    );
}

#[tokio::test]
async fn cli_produces_machine_readable_json_and_rejects_zero_batch() {
    let server = TestHub::new(
        BTreeMap::from([(
            "model.safetensors".into(),
            safetensors(&[("w", "F16", &[8], 16)]),
        )]),
        false,
        false,
    )
    .await;
    let endpoint = server.endpoint.clone();
    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new(env!("CARGO_BIN_EXE_llm-napkin"))
            .args([
                "--model-id",
                "test/model",
                "--endpoint",
                &endpoint,
                "--hf-token",
                "test-token",
                "--json-output",
            ])
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()["memory"],
        16
    );
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_llm-napkin"))
        .args(["--model-id", "test/model", "--batch-size", "0"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}
