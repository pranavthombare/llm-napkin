mod common;

use common::{TestHub, gguf, safetensors};
use llm_napkin::estimate;
use serde_json::{Value, json};
use std::collections::BTreeMap;

async fn model() -> TestHub {
    TestHub::new(
        BTreeMap::from([
            (
                "config.json".into(),
                serde_json::to_vec(&json!({
                    "architectures":["TestForCausalLM"], "hidden_size":128,
                    "num_attention_heads":4, "num_key_value_heads":2,
                    "num_hidden_layers":2, "torch_dtype":"bfloat16", "max_position_embeddings":1024
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
    .await
}

async fn cli(server: &TestHub, arguments: &[&str]) -> std::process::Output {
    let endpoint = server.endpoint.clone();
    let arguments: Vec<String> = arguments.iter().map(|s| (*s).into()).collect();
    tokio::task::spawn_blocking(move || {
        std::process::Command::new(env!("CARGO_BIN_EXE_llm-napkin"))
            .args([
                "--model-id",
                "test/model",
                "--endpoint",
                &endpoint,
                "--hf-token",
                "test-token",
            ])
            .args(arguments)
            .output()
            .unwrap()
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn token_budgets_enable_cache_and_preserve_workload_in_json() {
    let server = model().await;
    let mut options = server.options();
    options.input_tokens = Some(48);
    options.output_tokens = Some(16);
    options.batch_size = 4;
    let result = estimate(&options).await.unwrap();
    assert_eq!(result.total_memory().unwrap(), Some(131088));
    for details in [false, true] {
        let output = result.to_json(details).unwrap();
        assert_eq!(
            output["workload"],
            json!({"input_tokens":48,"output_tokens":16,"batch_size":4})
        );
        if details {
            assert_eq!(output["kv_cache"]["max_model_len"], 64);
            assert_eq!(output["kv_cache"]["bytes"], 131072);
        } else {
            assert_eq!(output["kv_cache"], 131072);
        }
    }
    assert!(!options.experimental);
    assert_eq!(options.max_model_len, None);
}

#[tokio::test]
async fn omitted_token_budget_is_zero_and_old_context_option_still_works() {
    let server = model().await;
    for (input, output) in [(Some(64), None), (None, Some(64)), (Some(64), Some(0))] {
        let mut options = server.options();
        options.input_tokens = input;
        options.output_tokens = output;
        let result = estimate(&options).await.unwrap();
        assert_eq!(result.total_memory().unwrap(), Some(32784));
        assert_eq!(result.workload.unwrap().context_length().unwrap(), 64);
    }
    let mut options = server.options();
    options.experimental = true;
    options.max_model_len = Some(64);
    let result = estimate(&options).await.unwrap();
    assert_eq!(result.total_memory().unwrap(), Some(32784));
    assert!(result.to_json(false).unwrap().get("workload").is_none());
}

#[tokio::test]
async fn invalid_workloads_fail_before_any_network_request() {
    let server = model().await;
    let mut cases = Vec::new();
    for (input, output, context, batch) in [
        (Some(0), Some(0), None, 1),
        (Some(0), None, None, 1),
        (Some(u64::MAX), Some(1), None, 1),
        (Some(32), Some(32), Some(64), 1),
        (Some(32), Some(32), None, 0),
    ] {
        let mut options = server.options();
        options.input_tokens = input;
        options.output_tokens = output;
        options.max_model_len = context;
        options.batch_size = batch;
        cases.push(options);
    }
    for options in cases {
        assert!(estimate(&options).await.is_err());
    }
    assert!(server.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn cli_summary_table_contains_settings_and_memory_totals() {
    let server = model().await;
    let output = cli(
        &server,
        &[
            "--input-tokens",
            "48",
            "--output-tokens",
            "16",
            "--batch-size",
            "4",
        ],
    )
    .await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let rows: Vec<Vec<&str>> = stdout
        .lines()
        .filter(|line| line.starts_with('|'))
        .map(|line| line.trim_matches('|').split('|').map(str::trim).collect())
        .collect();
    assert_eq!(
        rows[0],
        [
            "Input tokens",
            "Output tokens",
            "Context",
            "Batch",
            "Weights",
            "KV dtype",
            "KV cache",
            "Total"
        ]
    );
    assert_eq!(
        rows[1],
        [
            "48",
            "16",
            "64",
            "4",
            "16.00 B",
            "BF16",
            "128.00 KiB",
            "128.02 KiB"
        ]
    );
    assert_eq!(
        stdout.lines().filter(|line| line.starts_with('+')).count(),
        3
    );
    let detailed = cli(
        &server,
        &["--input-tokens", "48", "--output-tokens", "16", "--details"],
    )
    .await;
    assert!(detailed.status.success());
    let stdout = String::from_utf8(detailed.stdout).unwrap();
    assert!(stdout.contains("Component"));
    assert!(stdout.contains("Parameters"));
    assert!(stdout.contains("Transformer"));
}

#[tokio::test]
async fn gguf_variant_table_and_json_use_the_same_token_budgets() {
    let server = TestHub::new(
        BTreeMap::from([
            ("a.gguf".into(), gguf(&[("w", 1, &[32])], true, 0)),
            ("b.gguf".into(), gguf(&[("w", 2, &[32])], true, 0)),
        ]),
        false,
        false,
    )
    .await;
    let arguments = [
        "--input-tokens",
        "8",
        "--output-tokens",
        "24",
        "--batch-size",
        "2",
    ];
    let output = cli(&server, &arguments).await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("GGUF variant"));
    assert_eq!(
        stdout.lines().filter(|line| line.starts_with('|')).count(),
        3
    );
    for filename in ["a.gguf", "b.gguf"] {
        let row = stdout.lines().find(|line| line.contains(filename)).unwrap();
        assert!(row.contains("32.00 KiB"));
    }
    let mut json_args = arguments.to_vec();
    json_args.push("--json-output");
    let output = cli(&server, &json_args).await;
    assert!(output.status.success());
    let json: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["kv_cache"], json!({"a.gguf":32768,"b.gguf":32768}));
    assert_eq!(
        json["workload"],
        json!({"input_tokens":8,"output_tokens":24,"batch_size":2})
    );
    assert!(json["total_memory"].is_null());
}

#[tokio::test]
async fn weights_only_table_labels_missing_cache_and_cli_rejects_conflicts() {
    let server = model().await;
    let output = cli(&server, &[]).await;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("not estimated"));
    assert!(stdout.contains("weights-only totals"));
    let output = cli(&server, &["--input-tokens", "48", "--max-model-len", "64"]).await;
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot be used with"));
    let output = cli(&server, &["--output-tokens", "-1"]).await;
    assert!(!output.status.success());
}
