use std::{collections::BTreeSet, path::PathBuf, time::Duration};

use anyhow::{Context, Result, bail, ensure};
use reqwest::{Client, Response, StatusCode, Url, header};
use serde::Deserialize;
use serde_json::Value;

pub(crate) const MAX_METADATA: usize = 100_000_000;

#[derive(Debug, Deserialize)]
pub(crate) struct HubFile {
    #[serde(rename = "type")]
    pub kind: String,
    pub path: String,
    pub size: Option<u64>,
}

pub(crate) struct Hub {
    client: Client,
    base: Url,
    token: Option<String>,
}

pub(crate) fn valid_path(path: &str) -> bool {
    !path.is_empty()
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
        && !path.contains(['\\', '\n', '\r', '\0'])
}

impl Hub {
    pub async fn new(endpoint: &str, explicit_token: Option<&str>) -> Result<Self> {
        let base = Url::parse(endpoint).context("invalid Hugging Face endpoint")?;
        ensure!(
            matches!(base.scheme(), "https" | "http") && base.host_str().is_some(),
            "endpoint must be an HTTP(S) URL"
        );
        ensure!(
            base.username().is_empty()
                && base.password().is_none()
                && base.query().is_none()
                && base.fragment().is_none(),
            "endpoint must not contain credentials, a query, or a fragment"
        );
        let token = match explicit_token {
            Some(token) => Some(token.to_owned()),
            None => match std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty()) {
                Some(token) => Some(token),
                None => {
                    let home = std::env::var_os("HOME").map(PathBuf::from);
                    let path = if let Some(path) = std::env::var_os("HF_TOKEN_PATH") {
                        Some(PathBuf::from(path))
                    } else if let Some(path) = std::env::var_os("HF_HOME") {
                        let path = PathBuf::from(path);
                        Some(
                            if path.is_absolute() {
                                path
                            } else {
                                home.clone().unwrap_or_default().join(path)
                            }
                            .join("token"),
                        )
                    } else {
                        std::env::var_os("XDG_CACHE_HOME")
                            .map(PathBuf::from)
                            .or_else(|| home.map(|p| p.join(".cache")))
                            .map(|p| p.join("huggingface/token"))
                    };
                    match path {
                        Some(path) => match tokio::fs::read_to_string(path).await {
                            Ok(token) => Some(token),
                            Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
                            Err(err) => {
                                return Err(err).context("could not read Hugging Face token file");
                            }
                        },
                        None => None,
                    }
                }
            },
        }
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
        Ok(Self {
            client: Client::builder()
                .user_agent(concat!("llm-napkin/", env!("CARGO_PKG_VERSION")))
                .timeout(Duration::from_secs(30))
                .connect_timeout(Duration::from_secs(10))
                .build()?,
            base,
            token,
        })
    }

    fn url(&self, model: &str, revision: &str, file: Option<&str>) -> Result<Url> {
        ensure!(
            valid_path(model) && model.split('/').count() <= 2,
            "invalid model ID: expected name or owner/name"
        );
        ensure!(!revision.is_empty(), "revision must not be empty");
        let mut url = self.base.clone();
        {
            let mut parts = url
                .path_segments_mut()
                .map_err(|_| anyhow::anyhow!("invalid endpoint"))?;
            parts.pop_if_empty();
            if file.is_none() {
                parts.extend(["api", "models"]);
            }
            parts.extend(model.split('/'));
            parts.push(if file.is_some() { "resolve" } else { "tree" });
            parts.push(revision);
            if let Some(file) = file {
                ensure!(valid_path(file), "invalid repository file path");
                parts.extend(file.split('/'));
            }
        }
        if file.is_none() {
            url.set_query(Some("recursive=true&limit=1000"));
        }
        Ok(url)
    }

    async fn request(&self, url: Url, prefix: Option<usize>) -> Result<Response> {
        for attempt in 0..3_u64 {
            let mut request = self
                .client
                .get(url.clone())
                .header(header::ACCEPT_ENCODING, "identity");
            if let Some(token) = &self.token {
                request = request.bearer_auth(token);
            }
            if let Some(length) = prefix {
                request = request.header(header::RANGE, format!("bytes=0-{}", length - 1));
            }
            let response = request
                .send()
                .await
                .context("Hugging Face request failed")?;
            let status = response.status();
            if (status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()) && attempt < 2
            {
                let delay = response
                    .headers()
                    .get(header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(1 << attempt)
                    .min(30);
                tokio::time::sleep(Duration::from_secs(delay)).await;
                continue;
            }
            if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
                bail!(
                    "Hugging Face returned {status}; for private or gated models, set HF_TOKEN and ensure access is granted"
                );
            }
            ensure!(
                status.is_success(),
                "Hugging Face returned {status} for {url}"
            );
            if prefix.is_some() && status == StatusCode::PARTIAL_CONTENT {
                let range = response
                    .headers()
                    .get(header::CONTENT_RANGE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");
                ensure!(
                    range.starts_with("bytes 0-"),
                    "server returned an unexpected Content-Range"
                );
            }
            return Ok(response);
        }
        unreachable!()
    }

    async fn body(mut response: Response, limit: usize, allow_prefix: bool) -> Result<Vec<u8>> {
        if !allow_prefix {
            ensure!(
                response
                    .content_length()
                    .is_none_or(|size| size <= limit as u64),
                "JSON metadata exceeds size limit"
            );
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .context("reading metadata response")?
        {
            let remaining = limit - bytes.len();
            if !allow_prefix {
                ensure!(chunk.len() <= remaining, "JSON metadata exceeds size limit");
            }
            bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
            // Drop the response here even if a server ignores Range and streams full weights.
            if allow_prefix && bytes.len() == limit {
                break;
            }
        }
        Ok(bytes)
    }

    pub async fn prefix(
        &self,
        model: &str,
        revision: &str,
        file: &str,
        length: usize,
    ) -> Result<Vec<u8>> {
        ensure!(
            (1..=MAX_METADATA + 8).contains(&length),
            "metadata exceeds the 100 MB limit"
        );
        let response = self
            .request(self.url(model, revision, Some(file))?, Some(length))
            .await?;
        Self::body(response, length, true)
            .await
            .with_context(|| format!("reading {file}"))
    }

    pub async fn json(&self, model: &str, revision: &str, file: &str) -> Result<Value> {
        let response = self
            .request(self.url(model, revision, Some(file))?, None)
            .await?;
        serde_json::from_slice(&Self::body(response, MAX_METADATA, false).await?)
            .with_context(|| format!("invalid JSON in {file}"))
    }

    pub async fn files(&self, model: &str, revision: &str) -> Result<Vec<HubFile>> {
        let mut next = Some(self.url(model, revision, None)?);
        let mut visited = BTreeSet::new();
        let mut files = Vec::new();
        while let Some(url) = next.take() {
            ensure!(
                visited.insert(url.to_string()) && visited.len() <= 10000,
                "invalid or excessive repository pagination"
            );
            let response = self.request(url.clone(), None).await?;
            if let Some(links) = response
                .headers()
                .get(header::LINK)
                .and_then(|v| v.to_str().ok())
            {
                for link in links.split(',') {
                    let mut parts = link.trim().split(';');
                    let target = parts.next().unwrap_or_default().trim();
                    if parts.any(|p| p.trim() == "rel=\"next\"" || p.trim() == "rel=next") {
                        let target = target
                            .strip_prefix('<')
                            .and_then(|p| p.strip_suffix('>'))
                            .context("invalid pagination link")?;
                        let page = url.join(target)?;
                        ensure!(
                            page.origin() == self.base.origin() && page.path() == url.path(),
                            "pagination link leaves the repository endpoint"
                        );
                        next = Some(page);
                    }
                }
            }
            let page: Vec<HubFile> =
                serde_json::from_slice(&Self::body(response, MAX_METADATA, false).await?)
                    .context("invalid repository file listing")?;
            files.extend(page.into_iter().filter(|file| file.kind == "file"));
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        files.dedup_by(|a, b| a.path == b.path);
        Ok(files)
    }
}
