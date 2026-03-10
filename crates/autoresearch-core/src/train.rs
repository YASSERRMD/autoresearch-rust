#![cfg(feature = "train")]

use std::f64::consts::LN_2;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Result};
use candle_core::{DType, Device};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

use crate::config::{CachePaths, CoreConstants, TrainConfig};
use crate::dataloader::{PackedBatchLoader, Split};
use crate::model::{GptModel, GptModelConfig};
use crate::report::RunSummary;
use crate::tokenizer::{load_token_bytes, RuntimeTokenizer};

const LOADER_BUFFER_SIZE: usize = 1000;

pub fn run_train(
    paths: &CachePaths,
    constants: CoreConstants,
    train: &TrainConfig,
) -> Result<RunSummary> {
    let overall_start = Instant::now();
    let device = Arc::new(Device::Cpu);

    let tokenizer = RuntimeTokenizer::from_cache(paths)?;
    let token_bytes = load_token_bytes(paths)?.token_bytes;
    let vocab_size = tokenizer.vocab_size();

    if vocab_size == 0 {
        bail!("tokenizer vocabulary is empty");
    }

    let model_cfg = GptModelConfig::from_train_config(train, constants, vocab_size)?;
    let model = GptModel::new(model_cfg.clone(), device.as_ref())?;
    let num_params = model.num_parameters();

    let tokens_per_micro = train.device_batch_size * constants.max_seq_len;
    if tokens_per_micro == 0 || train.total_batch_size % tokens_per_micro != 0 {
        bail!(
            "invalid batch setup: total_batch_size={} must be divisible by device_batch_size * seq_len={}",
            train.total_batch_size,
            tokens_per_micro
        );
    }
    let grad_accum_steps = train.total_batch_size / tokens_per_micro;

    let vars = model.variables();
    let mut optimizer = AdamW::new(
        vars,
        ParamsAdamW {
            lr: train.learning_rate,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: train.weight_decay,
        },
    )?;

    let mut train_loader = PackedBatchLoader::new(
        paths,
        constants,
        tokenizer.clone(),
        Split::Train,
        train.device_batch_size,
        constants.max_seq_len,
        LOADER_BUFFER_SIZE,
        device.clone(),
    )?;

    let mut total_training_time = 0.0_f64;
    let mut smooth_train_loss = 0.0_f64;
    let mut step = 0usize;

    while total_training_time < constants.time_budget_seconds as f64 || step <= 10 {
        let step_start = Instant::now();
        let mut loss_sum = 0.0_f64;
        let progress = (total_training_time / constants.time_budget_seconds as f64).clamp(0.0, 1.0);
        let lr_mult = lr_multiplier(progress, train);
        let step_lr = train.learning_rate * lr_mult / grad_accum_steps as f64;
        optimizer.set_learning_rate(step_lr);

        for _ in 0..grad_accum_steps {
            let (x, y, _) = train_loader.next_batch()?;
            let loss = model.forward_loss(&x, &y)?;
            loss_sum += loss.to_scalar::<f32>()? as f64;
            optimizer.backward_step(&loss)?;
        }

        let step_secs = step_start.elapsed().as_secs_f64();
        if step > 10 {
            total_training_time += step_secs;
        }

        let train_loss = loss_sum / grad_accum_steps as f64;
        smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * train_loss;
        let debiased = smooth_train_loss / (1.0 - 0.9_f64.powi((step + 1) as i32));

        let pct_done = 100.0 * progress;
        let remaining = (constants.time_budget_seconds as f64 - total_training_time).max(0.0);
        let tok_per_sec = if step_secs > 0.0 {
            (train.total_batch_size as f64 / step_secs) as usize
        } else {
            0
        };

        println!(
            "\rstep {:05} ({:.1}%) | loss: {:.6} | lr_mult: {:.3} | dt: {:.0}ms | tok/sec: {} | remaining: {:.0}s",
            step,
            pct_done,
            debiased,
            lr_mult,
            step_secs * 1000.0,
            tok_per_sec,
            remaining,
        );

        step += 1;

        if step > 10 && total_training_time >= constants.time_budget_seconds as f64 {
            break;
        }
    }

    let val_bpb = evaluate_bpb(
        &model,
        paths,
        constants,
        train,
        tokenizer,
        token_bytes,
        device,
    )?;

    let training_seconds = total_training_time;
    let total_seconds = overall_start.elapsed().as_secs_f64();
    let total_tokens_m = (step * train.total_batch_size) as f64 / 1e6;

    Ok(RunSummary {
        val_bpb,
        training_seconds,
        total_seconds,
        peak_vram_mb: 0.0,
        mfu_percent: 0.0,
        total_tokens_m,
        num_steps: step,
        num_params_m: num_params as f64 / 1e6,
        depth: model_cfg.n_layer,
    })
}

pub fn evaluate_bpb(
    model: &GptModel,
    paths: &CachePaths,
    constants: CoreConstants,
    train: &TrainConfig,
    tokenizer: RuntimeTokenizer,
    token_bytes: Vec<u32>,
    device: Arc<Device>,
) -> Result<f64> {
    let mut val_loader = PackedBatchLoader::new(
        paths,
        constants,
        tokenizer,
        Split::Val,
        train.eval_batch_size,
        constants.max_seq_len,
        LOADER_BUFFER_SIZE,
        device,
    )?;

    let eval_steps = constants.eval_tokens / (train.eval_batch_size * constants.max_seq_len);
    if eval_steps == 0 {
        bail!("evaluation setup produced zero steps");
    }

    let mut total_nats = 0.0_f64;
    let mut total_bytes = 0.0_f64;

    for _ in 0..eval_steps {
        let (x, y, _) = val_loader.next_batch()?;
        let logits = model.forward_logits(&x)?;
        let (b, t, v) = logits.dims3()?;

        let flat_logits = logits.reshape((b * t, v))?;
        let flat_targets = y.reshape((b * t,))?.to_dtype(DType::U32)?;

        let log_probs = candle_nn::ops::log_softmax(&flat_logits, 1)?;
        let gathered = log_probs
            .gather(&flat_targets.unsqueeze(1)?, 1)?
            .squeeze(1)?;
        let nll = gathered.neg()?;

        let nll_vec = nll.to_vec1::<f32>()?;
        let target_vec = flat_targets.to_vec1::<u32>()?;

        for (token_id, nll_val) in target_vec.into_iter().zip(nll_vec.into_iter()) {
            let nbytes = token_bytes
                .get(token_id as usize)
                .copied()
                .unwrap_or(0) as f64;
            if nbytes > 0.0 {
                total_nats += nll_val as f64;
                total_bytes += nbytes;
            }
        }
    }

    Ok(total_nats / (LN_2 * total_bytes.max(1.0)))
}

fn lr_multiplier(progress: f64, train: &TrainConfig) -> f64 {
    if progress < train.warmup_ratio {
        if train.warmup_ratio > 0.0 {
            progress / train.warmup_ratio
        } else {
            1.0
        }
    } else if progress < 1.0 - train.warmdown_ratio {
        1.0
    } else {
        let cooldown = (1.0 - progress) / train.warmdown_ratio.max(1e-9);
        cooldown + (1.0 - cooldown) * train.final_lr_fraction
    }
}
