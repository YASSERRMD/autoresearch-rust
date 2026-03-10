use std::path::PathBuf;

use anyhow::{Context, Result};
use autoresearch_core::{run_prepare, CachePaths, CoreConstants, PrepareConfig};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "autoresearch", version, about = "Autoresearch Rust CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Download shards and train tokenizer artifacts.
    Prepare {
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        num_shards: Option<isize>,
        #[arg(long)]
        download_workers: Option<usize>,
    },
}

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Prepare {
            cache_dir,
            num_shards,
            download_workers,
        } => run_prepare_command(cache_dir, num_shards, download_workers),
    }
}

fn run_prepare_command(
    cache_dir: Option<PathBuf>,
    num_shards: Option<isize>,
    download_workers: Option<usize>,
) -> Result<()> {
    let paths = match cache_dir {
        Some(path) => CachePaths::from_cache_dir(path),
        None => CachePaths::new_default().context("failed to resolve default cache directory")?,
    };

    let constants = CoreConstants::default();
    let mut prepare = PrepareConfig::default();
    if let Some(raw) = num_shards {
        prepare.num_shards = if raw < 0 {
            constants.max_shard
        } else {
            raw as usize
        };
    }
    if let Some(workers) = download_workers {
        prepare.download_workers = workers.max(1);
    }

    let summary = run_prepare(&paths, constants, &prepare)?;

    println!("Cache directory: {}", summary.cache_dir.display());
    println!("Data directory: {}", summary.data_dir.display());
    println!("Tokenizer directory: {}", summary.tokenizer_dir.display());
    println!(
        "Shards ready: {}/{}",
        summary.download.ready_shards(),
        summary.download.total_shards
    );
    println!(
        "Tokenizer: vocab={} bos={} docs={}",
        summary.tokenizer.vocab_size,
        summary.tokenizer.bos_token_id,
        summary.tokenizer.trained_docs
    );

    Ok(())
}
