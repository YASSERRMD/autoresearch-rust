use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};
use tracing::info;

use crate::config::{CachePaths, CoreConstants, PrepareConfig};
use crate::parquet_text::collect_training_documents;

pub const BOS_TOKEN: &str = "<|reserved_0|>";
pub const SPECIAL_TOKENS: [&str; 4] = [
    "<|reserved_0|>",
    "<|reserved_1|>",
    "<|reserved_2|>",
    "<|reserved_3|>",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerReport {
    pub vocab_size: usize,
    pub bos_token_id: u32,
    pub trained_docs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenByteTable {
    pub token_bytes: Vec<u32>,
}

#[derive(Clone)]
pub struct RuntimeTokenizer {
    tokenizer: Tokenizer,
    bos_token_id: u32,
}

impl RuntimeTokenizer {
    pub fn from_cache(paths: &CachePaths) -> Result<Self> {
        let tokenizer = load_tokenizer(paths)?;
        let bos_token_id = tokenizer
            .token_to_id(BOS_TOKEN)
            .ok_or_else(|| anyhow!("tokenizer is missing BOS token {BOS_TOKEN}"))?;
        Ok(Self {
            tokenizer,
            bos_token_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn encode(&self, text: &str, prepend_bos: bool) -> Result<Vec<u32>> {
        let mut ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| anyhow!(err.to_string()))?
            .get_ids()
            .to_vec();
        if prepend_bos {
            ids.insert(0, self.bos_token_id);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|err| anyhow!(err.to_string()))
    }
}

pub fn train_tokenizer(
    paths: &CachePaths,
    constants: CoreConstants,
    prepare: &PrepareConfig,
) -> Result<TokenizerReport> {
    fs::create_dir_all(&paths.tokenizer_dir).with_context(|| {
        format!(
            "failed to create tokenizer directory {}",
            paths.tokenizer_dir.display()
        )
    })?;

    if paths.tokenizer_json_path().exists() && paths.token_bytes_path().exists() {
        let runtime = RuntimeTokenizer::from_cache(paths)?;
        return Ok(TokenizerReport {
            vocab_size: runtime.vocab_size(),
            bos_token_id: runtime.bos_token_id(),
            trained_docs: 0,
        });
    }

    let docs = collect_training_documents(paths, constants, prepare)?;
    if docs.is_empty() {
        bail!("tokenizer training corpus is empty");
    }

    let bpe = BPE::builder()
        .unk_token("<|unk|>".to_owned())
        .build()
        .map_err(|err| anyhow!(err.to_string()))?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));

    let special_tokens = SPECIAL_TOKENS
        .iter()
        .map(|token| AddedToken::from((*token).to_owned(), true))
        .collect::<Vec<_>>();
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
        .show_progress(false)
        .vocab_size(constants.vocab_size)
        .special_tokens(special_tokens)
        .build()
        .into();

    let training_file = paths.tokenizer_dir.join("train_corpus.txt");
    write_training_corpus(&docs, &training_file)?;

    tokenizer
        .train_from_files(
            &mut trainer,
            vec![training_file.to_string_lossy().to_string()],
        )
        .map_err(|err| anyhow!(err.to_string()))?;

    save_tokenizer(&tokenizer, &paths.tokenizer_json_path())?;
    let table = build_token_bytes(&tokenizer)?;
    save_token_bytes(&table, &paths.token_bytes_path())?;

    let bos_token_id = tokenizer
        .token_to_id(BOS_TOKEN)
        .ok_or_else(|| anyhow!("trained tokenizer is missing BOS token {BOS_TOKEN}"))?;

    info!(
        "tokenizer trained with vocab={} docs={}",
        tokenizer.get_vocab_size(false),
        docs.len()
    );

    Ok(TokenizerReport {
        vocab_size: tokenizer.get_vocab_size(false),
        bos_token_id,
        trained_docs: docs.len(),
    })
}

pub fn load_tokenizer(paths: &CachePaths) -> Result<Tokenizer> {
    let path = paths.tokenizer_json_path();
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("tokenizer path is not valid UTF-8: {}", path.display()))?;
    Tokenizer::from_file(path_str)
        .map_err(|err| anyhow!(err.to_string()))
        .with_context(|| format!("failed to load tokenizer from {}", path.display()))
}

pub fn load_token_bytes(paths: &CachePaths) -> Result<TokenByteTable> {
    let raw = fs::read_to_string(paths.token_bytes_path()).with_context(|| {
        format!(
            "failed to read token-byte table from {}",
            paths.token_bytes_path().display()
        )
    })?;
    serde_json::from_str(&raw).context("invalid token-byte table JSON")
}

fn build_token_bytes(tokenizer: &Tokenizer) -> Result<TokenByteTable> {
    let vocab_size = tokenizer.get_vocab_size(false);
    let mut token_bytes = vec![0u32; vocab_size];
    let vocab = tokenizer.get_vocab(true);

    for (token, token_id) in vocab {
        let idx = token_id as usize;
        if idx >= token_bytes.len() {
            continue;
        }

        if SPECIAL_TOKENS.contains(&token.as_str()) {
            token_bytes[idx] = 0;
            continue;
        }

        let decoded = tokenizer
            .decode(&[token_id], true)
            .map_err(|err| anyhow!(err.to_string()))
            .with_context(|| format!("failed to decode token id {token_id}"))?;
        token_bytes[idx] = decoded.as_bytes().len() as u32;
    }

    Ok(TokenByteTable { token_bytes })
}

fn save_tokenizer(tokenizer: &Tokenizer, path: &Path) -> Result<()> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", path.display()))?;
    tokenizer
        .save(path_str, false)
        .map_err(|err| anyhow!(err.to_string()))
        .with_context(|| format!("failed to save tokenizer to {}", path.display()))?;
    Ok(())
}

fn write_training_corpus(docs: &[String], path: &Path) -> Result<()> {
    let mut file = fs::File::create(path)
        .with_context(|| format!("failed to create training corpus {}", path.display()))?;
    for doc in docs {
        file.write_all(doc.as_bytes())
            .with_context(|| format!("failed to write {}", path.display()))?;
        file.write_all(b"\n")
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    file.flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

fn save_token_bytes(table: &TokenByteTable, path: &Path) -> Result<()> {
    let encoded = serde_json::to_string(table).context("failed to encode token-byte table")?;
    fs::write(path, encoded)
        .with_context(|| format!("failed to save token-byte table to {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_tokens_are_unique() {
        let mut sorted = SPECIAL_TOKENS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), SPECIAL_TOKENS.len());
    }
}
