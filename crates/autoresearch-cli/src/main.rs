use std::path::PathBuf;

use anyhow::{Context, Result};
use autoresearch_core::{run_prepare, CachePaths, CoreConstants, PrepareConfig};
#[cfg(feature = "train")]
use autoresearch_core::{run_train, TrainConfig};
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
    /// Run a fixed-budget Rust training run.
    #[cfg(feature = "train")]
    Train {
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        depth: Option<usize>,
        #[arg(long)]
        total_batch_size: Option<usize>,
        #[arg(long)]
        device_batch_size: Option<usize>,
        #[arg(long)]
        eval_batch_size: Option<usize>,
        #[arg(long)]
        learning_rate: Option<f64>,
        #[arg(long)]
        weight_decay: Option<f64>,
        #[arg(long)]
        accelerator_cmd: Option<String>,
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
        #[cfg(feature = "train")]
        Commands::Train {
            cache_dir,
            depth,
            total_batch_size,
            device_batch_size,
            eval_batch_size,
            learning_rate,
            weight_decay,
            accelerator_cmd,
        } => run_train_command(
            cache_dir,
            depth,
            total_batch_size,
            device_batch_size,
            eval_batch_size,
            learning_rate,
            weight_decay,
            accelerator_cmd,
        ),
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

#[cfg(feature = "train")]
#[allow(clippy::too_many_arguments)]
fn run_train_command(
    cache_dir: Option<PathBuf>,
    depth: Option<usize>,
    total_batch_size: Option<usize>,
    device_batch_size: Option<usize>,
    eval_batch_size: Option<usize>,
    learning_rate: Option<f64>,
    weight_decay: Option<f64>,
    accelerator_cmd: Option<String>,
) -> Result<()> {
    let paths = match cache_dir {
        Some(path) => CachePaths::from_cache_dir(path),
        None => CachePaths::new_default().context("failed to resolve default cache directory")?,
    };
    let constants = CoreConstants::default();

    let mut train = TrainConfig::default();
    if let Some(v) = depth {
        train.depth = v;
    }
    if let Some(v) = total_batch_size {
        train.total_batch_size = v;
    }
    if let Some(v) = device_batch_size {
        train.device_batch_size = v;
    }
    if let Some(v) = eval_batch_size {
        train.eval_batch_size = v;
    }
    if let Some(v) = learning_rate {
        train.learning_rate = v;
    }
    if let Some(v) = weight_decay {
        train.weight_decay = v;
    }
    train.accelerator_cmd = accelerator_cmd;

    let summary = run_train(&paths, constants, &train)?;
    println!("{}", summary.as_pretty_block());
    Ok(())
}
