#![allow(dead_code)]
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    task::JoinHandle,
};

pub fn safetensors(tensors: &[(&str, &str, &[u64], u64)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0;
    for (name, dtype, shape, bytes) in tensors {
        header.insert(
            (*name).into(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        offset += bytes;
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut result = (header.len() as u64).to_le_bytes().to_vec();
    result.extend(header);
    result.resize(result.len() + offset as usize, 0);
    result
}

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend((value.len() as u64).to_le_bytes());
    out.extend(value.as_bytes());
}

pub fn gguf(tensors: &[(&str, u32, &[u64])], with_cache: bool, padding: usize) -> Vec<u8> {
    let mut fields = vec![];
    string(&mut fields, "general.architecture");
    fields.extend(8_u32.to_le_bytes());
    string(&mut fields, "llama");
    let mut field_count = 1_u64;
    if with_cache {
        for (name, value) in [
            ("block_count", 2_u32),
            ("attention.head_count", 4),
            ("attention.head_count_kv", 2),
            ("embedding_length", 128),
            ("context_length", 64),
        ] {
            string(&mut fields, &format!("llama.{name}"));
            fields.extend(4_u32.to_le_bytes());
            fields.extend(value.to_le_bytes());
            field_count += 1;
        }
    }
    if padding > 0 {
        string(&mut fields, "tokenizer.padding");
        fields.extend(8_u32.to_le_bytes());
        string(&mut fields, &"x".repeat(padding));
        field_count += 1;
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3_u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend(field_count.to_le_bytes());
    out.extend(fields);
    for (name, dtype, shape) in tensors {
        string(&mut out, name);
        out.extend((shape.len() as u32).to_le_bytes());
        for dim in *shape {
            out.extend(dim.to_le_bytes());
        }
        out.extend(dtype.to_le_bytes());
        out.extend(0_u64.to_le_bytes());
    }
    out
}

pub struct TestHub {
    pub endpoint: String,
    pub requests: Arc<Mutex<Vec<String>>>,
    handle: JoinHandle<()>,
}

impl Drop for TestHub {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

impl TestHub {
    pub async fn new(files: BTreeMap<String, Vec<u8>>, paginate: bool, ignore_range: bool) -> Self {
        let listing: Vec<Value> = files
            .iter()
            .map(|(path, data)| json!({"type":"file", "path":path, "size":data.len()}))
            .collect();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(vec![]));
        let captured = requests.clone();
        let base = endpoint.clone();
        let handle = tokio::spawn(async move {
            let files = Arc::new(files);
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let captured = captured.clone();
                let files = files.clone();
                let listing = listing.clone();
                let base = base.clone();
                tokio::spawn(async move {
                    let mut raw = Vec::new();
                    loop {
                        let mut chunk = [0_u8; 1024];
                        let n = socket.read(&mut chunk).await.unwrap();
                        if n == 0 {
                            return;
                        }
                        raw.extend_from_slice(&chunk[..n]);
                        if raw.windows(4).any(|p| p == b"\r\n\r\n") {
                            break;
                        }
                    }
                    let request = String::from_utf8(raw).unwrap();
                    captured.lock().unwrap().push(request.clone());
                    let path = request.split_whitespace().nth(1).unwrap();
                    let mut extra = String::new();
                    let mut status = "200 OK";
                    let mut body = if path.starts_with("/api/models/") {
                        if paginate && !path.contains("cursor=next") {
                            let path = path.split('?').next().unwrap();
                            extra = format!("Link: <{base}{path}?cursor=next>; rel=\"next\"\r\n");
                            serde_json::to_vec(&listing[..listing.len() / 2]).unwrap()
                        } else if paginate {
                            serde_json::to_vec(&listing[listing.len() / 2..]).unwrap()
                        } else {
                            serde_json::to_vec(&listing).unwrap()
                        }
                    } else {
                        let file = path
                            .split("/resolve/")
                            .nth(1)
                            .and_then(|p| p.split_once('/'))
                            .map(|(_, f)| f)
                            .unwrap_or("");
                        match files.get(file) {
                            Some(body) => body.clone(),
                            None => {
                                status = "404 Not Found";
                                vec![]
                            }
                        }
                    };
                    if let Some(range) = request.lines().find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("range: bytes=0-")
                            .map(str::to_owned)
                    }) {
                        if !ignore_range && !body.is_empty() {
                            let length =
                                (range.trim().parse::<usize>().unwrap() + 1).min(body.len());
                            extra.push_str(&format!(
                                "Content-Range: bytes 0-{}/{}\r\n",
                                length - 1,
                                body.len()
                            ));
                            body.truncate(length);
                            status = "206 Partial Content";
                        }
                    }
                    let header = format!(
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
                        body.len()
                    );
                    if socket.write_all(header.as_bytes()).await.is_ok() {
                        let _ = socket.write_all(&body).await;
                    }
                });
            }
        });
        Self {
            endpoint,
            requests,
            handle,
        }
    }

    pub fn options(&self) -> llm_napkin::Options {
        let mut options = llm_napkin::Options::new("test/model");
        options.endpoint = self.endpoint.clone();
        options.hf_token = Some("test-token".into());
        options
    }
}
