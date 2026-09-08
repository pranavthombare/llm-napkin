use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use llm_napkin::{Options, estimate};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

#[tokio::test]
async fn ignored_range_does_not_wait_for_or_download_the_weight_body() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut options = Options::new("test/model");
    options.endpoint = format!("http://{}", listener.local_addr().unwrap());
    options.hf_token = Some(String::new());
    let header = br#"{"w":{"dtype":"F16","shape":[1000000000],"data_offsets":[0,2000000000]}}"#;
    let mut prefix = (header.len() as u64).to_le_bytes().to_vec();
    prefix.extend(header);
    let prefix = Arc::new(prefix);
    let disconnected = Arc::new(AtomicUsize::new(0));
    let count = disconnected.clone();
    let handle = tokio::spawn(async move {
        let mut connections = tokio::task::JoinSet::new();
        for _ in 0..3 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let prefix = prefix.clone();
            let count = count.clone();
            connections.spawn(async move {
                let mut request = vec![];
                loop {
                    let mut buffer = [0;1024];
                    let n = socket.read(&mut buffer).await.unwrap();
                    request.extend_from_slice(&buffer[..n]);
                    if request.windows(4).any(|w| w == b"\r\n\r\n") { break; }
                }
                if request.starts_with(b"GET /api/") {
                    let listing = format!("[{{\"type\":\"file\",\"path\":\"model.safetensors\",\"size\":{}}}]",2_000_000_000+prefix.len());
                    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{listing}",listing.len()).as_bytes()).await.unwrap();
                } else {
                    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",2_000_000_000+prefix.len()).as_bytes()).await.unwrap();
                    socket.write_all(&prefix).await.unwrap();
                    // No tensor bytes are ever sent. A bounded client closes after the header.
                    let mut probe = [0;1];
                    let n = socket.read(&mut probe).await;
                    assert!(matches!(n,Ok(0) | Err(_)));
                    count.fetch_add(1,Ordering::SeqCst);
                }
            });
        }
        while let Some(result) = connections.join_next().await {
            result.unwrap();
        }
    });
    let result = tokio::time::timeout(Duration::from_secs(3), estimate(&options))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.total_memory().unwrap(), Some(2_000_000_000));
    tokio::time::timeout(Duration::from_secs(3), handle)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(disconnected.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn authorization_errors_explain_token_access_without_printing_response_secrets() {
    for status in ["401 Unauthorized", "403 Forbidden"] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut options = Options::new("test/private");
        options.endpoint = format!("http://{}", listener.local_addr().unwrap());
        options.hf_token = Some("fake-secret".into());
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            let _ = socket.read(&mut request).await;
            socket.write_all(format!("HTTP/1.1 {status}\r\nContent-Length: 11\r\nConnection: close\r\n\r\nfake-secret").as_bytes()).await.unwrap();
        });
        let error = format!("{:#}", estimate(&options).await.unwrap_err());
        assert!(error.contains("HF_TOKEN"));
        assert!(!error.contains("fake-secret"));
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn pagination_cannot_forward_credentials_to_another_origin() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut options = Options::new("test/model");
    options.endpoint = format!("http://{}", listener.local_addr().unwrap());
    options.hf_token = Some("fake-secret".into());
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0; 4096];
        let _ = socket.read(&mut request).await;
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nLink: <https://example.com/steal>; rel=\"next\"\r\nConnection: close\r\n\r\n[]").await.unwrap();
    });
    let error = estimate(&options).await.unwrap_err();
    assert!(error.to_string().contains("pagination link leaves"));
    handle.await.unwrap();
}
