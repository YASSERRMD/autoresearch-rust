#![cfg(feature = "train")]

use std::process::Command;

use anyhow::{bail, Context, Result};
use tracing::info;

#[derive(Debug, Clone, Copy)]
pub enum AcceleratorPhase {
    Training,
    Inference,
}

impl AcceleratorPhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Training => "training",
            Self::Inference => "inference",
        }
    }
}

/// Runs an optional external accelerator command for a specific phase.
///
/// This allows plugging tools like `barqtrain` without hard-coding
/// vendor-specific dependencies in the core engine.
pub fn maybe_run_accelerator(
    accelerator_cmd: Option<&str>,
    phase: AcceleratorPhase,
) -> Result<()> {
    let Some(cmd) = accelerator_cmd else {
        return Ok(());
    };

    let status = Command::new("sh")
        .arg("-lc")
        .arg(cmd)
        .status()
        .with_context(|| {
            format!(
                "failed to execute accelerator command for {}: {cmd}",
                phase.as_str()
            )
        })?;

    if !status.success() {
        bail!(
            "accelerator command failed during {} with status {status}: {cmd}",
            phase.as_str()
        );
    }

    info!(
        "accelerator command completed successfully for {}",
        phase.as_str()
    );
    Ok(())
}
