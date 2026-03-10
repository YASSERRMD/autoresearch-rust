use std::fs::{self, File};
use std::io::{copy, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use rayon::prelude::*;
use reqwest::blocking::Client;
use tracing::{info, warn};

use crate::config::{CachePaths, CoreConstants, PrepareConfig, DEFAULT_DATASET_BASE_URL};

#[derive(Debug, Clone)]
pub struct DownloadReport {
    pub total_shards: usize,
    pub existing_shards: usize,
    pub downloaded_shards: usize,
}

impl DownloadReport {
    pub fn ready_shards(&self) -> usize {
        self.existing_shards + self.downloaded_shards
    }
}

pub fn download_data(
    paths: &CachePaths,
    constants: CoreConstants,
    prepare: &PrepareConfig,
) -> Result<DownloadReport> {
    fs::create_dir_all(&paths.data_dir)
        .with_context(|| format!("failed to create data directory {}", paths.data_dir.display()))?;

    let shard_ids = prepare.required_shard_ids(constants);
    let existing_shards = shard_ids
        .iter()
        .filter(|&&id| paths.shard_path(id).exists())
        .count();

    if existing_shards == shard_ids.len() {
        info!(
            "all {} requested shards already exist at {}",
            shard_ids.len(),
            paths.data_dir.display()
        );
        return Ok(DownloadReport {
            total_shards: shard_ids.len(),
            existing_shards,
            downloaded_shards: 0,
        });
    }

    let to_download: Vec<usize> = shard_ids
        .iter()
        .copied()
        .filter(|id| !paths.shard_path(*id).exists())
        .collect();

    let worker_count = prepare.download_workers.max(1).min(to_download.len().max(1));
    info!(
        "downloading {} shard(s) with {} worker(s)",
        to_download.len(),
        worker_count
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .context("failed to build download worker pool")?;

    let client = Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(180))
        .build()
        .context("failed to build HTTP client")?;

    let results = pool.install(|| {
        to_download
            .par_iter()
            .map(|shard_id| {
                download_single_shard(
                    &client,
                    paths,
                    *shard_id,
                    DEFAULT_DATASET_BASE_URL,
                    5,
                )
                .map(|_| *shard_id)
            })
            .collect::<Vec<_>>()
    });

    let mut downloaded_shards = 0;
    for result in results {
        match result {
            Ok(_) => downloaded_shards += 1,
            Err(err) => warn!("download error: {err:#}"),
        }
    }

    Ok(DownloadReport {
        total_shards: shard_ids.len(),
        existing_shards,
        downloaded_shards,
    })
}

fn download_single_shard(
    client: &Client,
    paths: &CachePaths,
    shard_id: usize,
    base_url: &str,
    max_attempts: usize,
) -> Result<()> {
    let filename = CachePaths::shard_filename(shard_id);
    let shard_path = paths.shard_path(shard_id);
    if shard_path.exists() {
        return Ok(());
    }

    let url = format!("{base_url}/{filename}");
    for attempt in 1..=max_attempts {
        match download_to_temp(client, &url, &shard_path) {
            Ok(_) => {
                info!("downloaded {filename}");
                return Ok(());
            }
            Err(err) => {
                warn!("attempt {attempt}/{max_attempts} failed for {filename}: {err:#}");
                if attempt == max_attempts {
                    return Err(err).context(format!("failed to download {filename}"));
                }
                std::thread::sleep(Duration::from_secs(1 << attempt.min(4)));
            }
        }
    }

    Err(anyhow::anyhow!("unreachable download state"))
}

fn download_to_temp(client: &Client, url: &str, target_path: &Path) -> Result<()> {
    let tmp_path = target_path.with_extension("parquet.tmp");
    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("request failed for {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;

    let mut file = File::create(&tmp_path)
        .with_context(|| format!("failed to create temp file {}", tmp_path.display()))?;
    copy(&mut response, &mut file)
        .with_context(|| format!("failed to write temp file {}", tmp_path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush temp file {}", tmp_path.display()))?;

    fs::rename(&tmp_path, target_path).with_context(|| {
        format!(
            "failed to move temp file {} to {}",
            tmp_path.display(),
            target_path.display()
        )
    })?;

    Ok(())
}
