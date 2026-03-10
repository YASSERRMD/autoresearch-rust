#![cfg(feature = "train")]

use std::process::Command;

use anyhow::{bail, Context, Result};
use tracing::info;

/// Runs an optional external accelerator command before training.
///
/// This allows plugging tools like `groqtrain` without hard-coding
/// vendor-specific dependencies in the core engine.
pub fn maybe_run_accelerator(accelerator_cmd: Option<&str>) -> Result<()> {
    let Some(cmd) = accelerator_cmd else {
        return Ok(());
    };

    let status = Command::new("sh")
        .arg("-lc")
        .arg(cmd)
        .status()
        .with_context(|| format!("failed to execute accelerator command: {cmd}"))?;

    if !status.success() {
        bail!("accelerator command failed with status {status}: {cmd}");
    }

    info!("accelerator command completed successfully");
    Ok(())
}
