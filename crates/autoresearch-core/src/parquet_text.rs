use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use arrow_array::{Array, LargeStringArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::config::{CachePaths, CoreConstants, PrepareConfig};

pub fn list_parquet_files(paths: &CachePaths) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&paths.data_dir)
        .with_context(|| format!("failed to read {}", paths.data_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let is_parquet = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext == "parquet")
            .unwrap_or(false);
        if is_parquet {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

pub fn split_train_val_paths(
    paths: &CachePaths,
    constants: CoreConstants,
) -> Result<(Vec<PathBuf>, PathBuf)> {
    let files = list_parquet_files(paths)?;
    if files.is_empty() {
        bail!("no parquet files found in {}", paths.data_dir.display());
    }

    let val_path = paths.shard_path(constants.val_shard);
    if !val_path.exists() {
        bail!(
            "validation shard is missing: {}",
            val_path.to_string_lossy()
        );
    }

    let train_paths = files.into_iter().filter(|path| path != &val_path).collect();
    Ok((train_paths, val_path))
}

pub fn collect_training_documents(
    paths: &CachePaths,
    constants: CoreConstants,
    prepare: &PrepareConfig,
) -> Result<Vec<String>> {
    let (train_paths, _) = split_train_val_paths(paths, constants)?;
    if train_paths.is_empty() {
        bail!("need at least one training shard (validation shard is excluded)");
    }

    let mut docs = Vec::new();
    let mut chars = 0usize;
    for path in train_paths {
        let rows = read_text_column(&path)?;
        for mut text in rows {
            if text.len() > prepare.doc_char_cap {
                text.truncate(prepare.doc_char_cap);
            }
            chars += text.len();
            docs.push(text);
            if chars >= prepare.max_chars_for_tokenizer {
                return Ok(docs);
            }
        }
    }

    Ok(docs)
}

pub fn read_text_column(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open parquet file {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to build parquet reader for {}", path.display()))?;

    let schema = builder.schema();
    let text_idx = schema
        .index_of("text")
        .with_context(|| format!("column 'text' not found in {}", path.display()))?;
    let mut reader = builder
        .with_batch_size(1024)
        .build()
        .with_context(|| format!("failed to read parquet batches from {}", path.display()))?;

    let mut out = Vec::new();
    for batch in reader.by_ref() {
        let batch = batch.with_context(|| format!("invalid record batch in {}", path.display()))?;
        let col = batch.column(text_idx);

        if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    out.push(arr.value(i).to_owned());
                }
            }
            continue;
        }

        if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    out.push(arr.value(i).to_owned());
                }
            }
            continue;
        }

        bail!("text column in {} is not a UTF-8 string type", path.display());
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_uses_pinned_validation_shard() {
        let paths = CachePaths::from_cache_dir(PathBuf::from("/tmp/does-not-exist"));
        let constants = CoreConstants::default();
        let result = split_train_val_paths(&paths, constants);
        assert!(result.is_err());
    }
}
