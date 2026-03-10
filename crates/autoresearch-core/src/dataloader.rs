#![cfg(feature = "train")]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};

use crate::config::{CachePaths, CoreConstants};
use crate::parquet_text::{read_text_column, split_train_val_paths};
use crate::tokenizer::RuntimeTokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Val,
}

impl Split {
    fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
        }
    }
}

struct DocumentStream {
    split: Split,
    tokenizer: RuntimeTokenizer,
    shard_paths: Vec<PathBuf>,
    shard_idx: usize,
    docs: Vec<String>,
    doc_idx: usize,
    epoch: usize,
}

impl DocumentStream {
    fn new(
        paths: &CachePaths,
        constants: CoreConstants,
        split: Split,
        tokenizer: RuntimeTokenizer,
    ) -> Result<Self> {
        let (train_paths, val_path) = split_train_val_paths(paths, constants)?;
        let shard_paths = match split {
            Split::Train => train_paths,
            Split::Val => vec![val_path],
        };

        if shard_paths.is_empty() {
            bail!("split '{}' has no shards", split.as_str());
        }

        Ok(Self {
            split,
            tokenizer,
            shard_paths,
            shard_idx: 0,
            docs: Vec::new(),
            doc_idx: 0,
            epoch: 1,
        })
    }

    fn next_doc_tokens(&mut self) -> Result<(Vec<u32>, usize)> {
        if self.doc_idx >= self.docs.len() {
            self.load_next_shard()?;
        }

        let text = &self.docs[self.doc_idx];
        self.doc_idx += 1;
        let ids = self.tokenizer.encode(text, true)?;
        Ok((ids, self.epoch))
    }

    fn load_next_shard(&mut self) -> Result<()> {
        if self.shard_idx >= self.shard_paths.len() {
            self.shard_idx = 0;
            if self.split == Split::Train {
                self.epoch += 1;
            }
        }

        let shard_path = &self.shard_paths[self.shard_idx];
        self.shard_idx += 1;
        self.docs = read_text_column(shard_path)?;
        self.doc_idx = 0;

        if self.docs.is_empty() {
            bail!("shard {} has no text rows", shard_path.display());
        }

        Ok(())
    }
}

pub struct PackedBatchLoader {
    stream: DocumentStream,
    batch_size: usize,
    seq_len: usize,
    buffer_size: usize,
    doc_buffer: Vec<Vec<u32>>,
    device: Arc<Device>,
    epoch: usize,
}

impl PackedBatchLoader {
    pub fn new(
        paths: &CachePaths,
        constants: CoreConstants,
        tokenizer: RuntimeTokenizer,
        split: Split,
        batch_size: usize,
        seq_len: usize,
        buffer_size: usize,
        device: Arc<Device>,
    ) -> Result<Self> {
        let stream = DocumentStream::new(paths, constants, split, tokenizer)?;
        Ok(Self {
            stream,
            batch_size,
            seq_len,
            buffer_size,
            doc_buffer: Vec::new(),
            device,
            epoch: 1,
        })
    }

    pub fn next_batch(&mut self) -> Result<(Tensor, Tensor, usize)> {
        let row_capacity = self.seq_len + 1;
        let mut inputs = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(self.batch_size * self.seq_len);

        for _ in 0..self.batch_size {
            let mut row = vec![0u32; row_capacity];
            let mut pos = 0usize;
            while pos < row_capacity {
                while self.doc_buffer.len() < self.buffer_size {
                    let (tokens, epoch) = self.stream.next_doc_tokens()?;
                    self.epoch = epoch;
                    self.doc_buffer.push(tokens);
                }

                let remaining = row_capacity - pos;
                if let Some(best_idx) = largest_fit_doc_index(&self.doc_buffer, remaining) {
                    let doc = self.doc_buffer.swap_remove(best_idx);
                    let doc_len = doc.len();
                    row[pos..pos + doc_len].copy_from_slice(&doc);
                    pos += doc_len;
                } else {
                    let shortest_idx = shortest_doc_index(&self.doc_buffer);
                    let doc = self.doc_buffer.swap_remove(shortest_idx);
                    row[pos..].copy_from_slice(&doc[..remaining]);
                    pos += remaining;
                }
            }

            inputs.extend_from_slice(&row[..self.seq_len]);
            targets.extend_from_slice(&row[1..]);
        }

        let shape = (self.batch_size, self.seq_len);
        let input_t = Tensor::from_vec(inputs, shape, self.device.as_ref())?;
        let target_t = Tensor::from_vec(targets, shape, self.device.as_ref())?;
        Ok((input_t, target_t, self.epoch))
    }
}

fn largest_fit_doc_index(doc_buffer: &[Vec<u32>], remaining: usize) -> Option<usize> {
    let mut best_idx = None;
    let mut best_len = 0usize;
    for (idx, doc) in doc_buffer.iter().enumerate() {
        let len = doc.len();
        if len <= remaining && len > best_len {
            best_idx = Some(idx);
            best_len = len;
        }
    }
    best_idx
}

fn shortest_doc_index(doc_buffer: &[Vec<u32>]) -> usize {
    let mut best_idx = 0usize;
    let mut best_len = usize::MAX;
    for (idx, doc) in doc_buffer.iter().enumerate() {
        let len = doc.len();
        if len < best_len {
            best_idx = idx;
            best_len = len;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_picker_prefers_largest_doc_that_fits() {
        let docs = vec![vec![1_u32, 2], vec![1_u32, 2, 3], vec![1_u32]];
        assert_eq!(largest_fit_doc_index(&docs, 2), Some(0));
        assert_eq!(largest_fit_doc_index(&docs, 3), Some(1));
        assert_eq!(largest_fit_doc_index(&docs, 0), None);
    }

    #[test]
    fn shortest_picker_finds_min_doc() {
        let docs = vec![vec![1_u32, 2, 3], vec![1_u32], vec![1_u32, 2]];
        assert_eq!(shortest_doc_index(&docs), 1);
    }
}
