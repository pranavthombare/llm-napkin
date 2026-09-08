mod common;

use std::collections::BTreeMap;

use common::{TestHub, safetensors};
use serde_json::{Value, json};

#[tokio::test]
async fn positional_model_short_options_and_json_alias_work_together() {
    let server = TestHub::new(
        BTreeMap::from([
            (
                "config.json".into(),
                serde_json::to_vec(&json!({
                    "architectures":["TestForCausalLM"], "hidden_size":128,
                    "num_attention_heads":4, "num_key_value_heads":2,
                    "num_hidden_layers":2, "torch_dtype":"bfloat16"
                }))
                .unwrap(),
            ),
            (
                "model.safetensors".into(),
                safetensors(&[("w", "F16", &[8], 16)]),
            ),
        ]),
        false,
        false,
    )
    .await;
    let endpoint = server.endpoint.clone();
    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new(env!("CARGO_BIN_EXE_llm-napkin"))
            .args([
                "test/model",
                "-i",
                "48",
                "-o",
                "16",
                "-b",
                "4",
                "--json",
                "-d",
                "-r",
                "refs/pr/4",
                "--endpoint",
                &endpoint,
                "--hf-token",
                "test-token",
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
    let output: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(output["model_id"], "test/model");
    assert_eq!(
        output["workload"],
        json!({"input_tokens":48,"output_tokens":16,"batch_size":4})
    );
    assert_eq!(output["kv_cache"]["bytes"], 131072);
    assert!(
        server
            .requests
            .lock()
            .unwrap()
            .iter()
            .all(|r| r.contains("refs%2Fpr%2F4"))
    );
}

#[test]
fn missing_or_ambiguous_model_arguments_are_rejected() {
    for args in [vec![], vec!["test/model", "--model-id", "test/other"]] {
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_llm-napkin"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(2));
    }
}
