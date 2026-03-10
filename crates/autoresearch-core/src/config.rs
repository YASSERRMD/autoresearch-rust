use std::path::PathBuf;

use anyhow::{Context, Result};
use dirs::home_dir;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CoreConstants {
    pub max_seq_len: usize,
    pub time_budget_seconds: u64,
    pub eval_tokens: usize,
    pub max_shard: usize,
    pub val_shard: usize,
    pub vocab_size: usize,
}

impl Default for CoreConstants {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            time_budget_seconds: 300,
            eval_tokens: 40 * 524_288,
            max_shard: 6542,
            val_shard: 6542,
            vocab_size: 8192,
        }
    }
}

pub const DEFAULT_DATASET_BASE_URL: &str =
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePaths {
    pub cache_dir: PathBuf,
    pub data_dir: PathBuf,
    pub tokenizer_dir: PathBuf,
}

impl CachePaths {
    pub fn from_cache_dir(cache_dir: PathBuf) -> Self {
        let data_dir = cache_dir.join("data");
        let tokenizer_dir = cache_dir.join("tokenizer");
        Self {
            cache_dir,
            data_dir,
            tokenizer_dir,
        }
    }

    pub fn default_cache_dir() -> Result<PathBuf> {
        let home = home_dir().context("could not locate home directory")?;
        Ok(home.join(".cache").join("autoresearch"))
    }

    pub fn new_default() -> Result<Self> {
        Ok(Self::from_cache_dir(Self::default_cache_dir()?))
    }

    pub fn shard_filename(shard_id: usize) -> String {
        format!("shard_{shard_id:05}.parquet")
    }

    pub fn shard_path(&self, shard_id: usize) -> PathBuf {
        self.data_dir.join(Self::shard_filename(shard_id))
    }

    pub fn tokenizer_json_path(&self) -> PathBuf {
        self.tokenizer_dir.join("tokenizer.json")
    }

    pub fn token_bytes_path(&self) -> PathBuf {
        self.tokenizer_dir.join("token_bytes.json")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareConfig {
    pub num_shards: usize,
    pub download_workers: usize,
    pub max_chars_for_tokenizer: usize,
    pub doc_char_cap: usize,
}

impl Default for PrepareConfig {
    fn default() -> Self {
        Self {
            num_shards: 10,
            download_workers: 8,
            max_chars_for_tokenizer: 1_000_000_000,
            doc_char_cap: 10_000,
        }
    }
}

impl PrepareConfig {
    pub fn required_shard_ids(&self, constants: CoreConstants) -> Vec<usize> {
        let num_train = self.num_shards.min(constants.max_shard);
        let mut ids: Vec<usize> = (0..num_train).collect();
        if !ids.contains(&constants.val_shard) {
            ids.push(constants.val_shard);
        }
        ids
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub depth: usize,
    pub aspect_ratio: usize,
    pub head_dim: usize,
    pub total_batch_size: usize,
    pub device_batch_size: usize,
    pub eval_batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub warmup_ratio: f64,
    pub warmdown_ratio: f64,
    pub final_lr_fraction: f64,
    pub accelerator_cmd: Option<String>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            depth: 8,
            aspect_ratio: 64,
            head_dim: 128,
            total_batch_size: 1 << 19,
            device_batch_size: 128,
            eval_batch_size: 128,
            learning_rate: 4e-4,
            weight_decay: 0.1,
            warmup_ratio: 0.0,
            warmdown_ratio: 0.5,
            final_lr_fraction: 0.0,
            accelerator_cmd: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_names_are_zero_padded() {
        assert_eq!(CachePaths::shard_filename(3), "shard_00003.parquet");
        assert_eq!(CachePaths::shard_filename(6542), "shard_06542.parquet");
    }

    #[test]
    fn cache_layout_is_stable() {
        let base = PathBuf::from("/tmp/autoresearch");
        let paths = CachePaths::from_cache_dir(base.clone());
        assert_eq!(paths.cache_dir, base);
        assert_eq!(paths.data_dir, PathBuf::from("/tmp/autoresearch/data"));
        assert_eq!(
            paths.tokenizer_dir,
            PathBuf::from("/tmp/autoresearch/tokenizer")
        );
    }
}
