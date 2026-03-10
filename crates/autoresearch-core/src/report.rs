use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    Keep,
    Discard,
    Crash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub val_bpb: f64,
    pub training_seconds: f64,
    pub total_seconds: f64,
    pub peak_vram_mb: f64,
    pub mfu_percent: f64,
    pub total_tokens_m: f64,
    pub num_steps: usize,
    pub num_params_m: f64,
    pub depth: usize,
}

impl RunSummary {
    pub fn as_pretty_block(&self) -> String {
        format!(
            "---\nval_bpb:          {:.6}\ntraining_seconds: {:.1}\ntotal_seconds:    {:.1}\npeak_vram_mb:     {:.1}\nmfu_percent:      {:.2}\ntotal_tokens_M:   {:.1}\nnum_steps:        {}\nnum_params_M:     {:.1}\ndepth:            {}",
            self.val_bpb,
            self.training_seconds,
            self.total_seconds,
            self.peak_vram_mb,
            self.mfu_percent,
            self.total_tokens_m,
            self.num_steps,
            self.num_params_m,
            self.depth,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretty_output_contains_required_fields() {
        let summary = RunSummary {
            val_bpb: 1.01,
            training_seconds: 300.0,
            total_seconds: 325.0,
            peak_vram_mb: 2048.0,
            mfu_percent: 11.0,
            total_tokens_m: 12.5,
            num_steps: 10,
            num_params_m: 50.2,
            depth: 8,
        };
        let out = summary.as_pretty_block();
        assert!(out.contains("val_bpb:"));
        assert!(out.contains("peak_vram_mb:"));
        assert!(out.contains("depth:"));
    }
}
