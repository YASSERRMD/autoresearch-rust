use std::path::PathBuf;

use anyhow::{bail, Result};

use crate::config::{CachePaths, CoreConstants, PrepareConfig};
use crate::download::{download_data, DownloadReport};
use crate::parquet_text::list_parquet_files;
use crate::tokenizer::{train_tokenizer, TokenizerReport};

#[derive(Debug, Clone)]
pub struct PrepareSummary {
    pub cache_dir: PathBuf,
    pub data_dir: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub download: DownloadReport,
    pub tokenizer: TokenizerReport,
}

pub fn run_prepare(
    paths: &CachePaths,
    constants: CoreConstants,
    config: &PrepareConfig,
) -> Result<PrepareSummary> {
    let download = download_data(paths, constants, config)?;

    let parquet_files = list_parquet_files(paths)?;
    if parquet_files.len() < 2 {
        bail!(
            "need at least 2 shards for prepare (1 train + 1 val), found {}",
            parquet_files.len()
        );
    }

    let tokenizer = train_tokenizer(paths, constants, config)?;

    Ok(PrepareSummary {
        cache_dir: paths.cache_dir.clone(),
        data_dir: paths.data_dir.clone(),
        tokenizer_dir: paths.tokenizer_dir.clone(),
        download,
        tokenizer,
    })
}
