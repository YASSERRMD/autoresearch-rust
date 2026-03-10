use anyhow::Result;
use autoresearch_core::{
    run_prepare, run_train, CachePaths, CoreConstants, PrepareConfig, RunSummary, TrainConfig,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
struct PyPrepareSummary {
    #[pyo3(get)]
    cache_dir: String,
    #[pyo3(get)]
    data_dir: String,
    #[pyo3(get)]
    tokenizer_dir: String,
    #[pyo3(get)]
    total_shards: usize,
    #[pyo3(get)]
    ready_shards: usize,
    #[pyo3(get)]
    vocab_size: usize,
    #[pyo3(get)]
    bos_token_id: u32,
}

#[pyclass]
#[derive(Debug, Clone)]
struct PyRunSummary {
    #[pyo3(get)]
    val_bpb: f64,
    #[pyo3(get)]
    training_seconds: f64,
    #[pyo3(get)]
    total_seconds: f64,
    #[pyo3(get)]
    peak_vram_mb: f64,
    #[pyo3(get)]
    mfu_percent: f64,
    #[pyo3(get)]
    total_tokens_m: f64,
    #[pyo3(get)]
    num_steps: usize,
    #[pyo3(get)]
    num_params_m: f64,
    #[pyo3(get)]
    depth: usize,
}

impl From<RunSummary> for PyRunSummary {
    fn from(summary: RunSummary) -> Self {
        Self {
            val_bpb: summary.val_bpb,
            training_seconds: summary.training_seconds,
            total_seconds: summary.total_seconds,
            peak_vram_mb: summary.peak_vram_mb,
            mfu_percent: summary.mfu_percent,
            total_tokens_m: summary.total_tokens_m,
            num_steps: summary.num_steps,
            num_params_m: summary.num_params_m,
            depth: summary.depth,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct AutoResearch {
    cache_dir: Option<String>,
}

#[pymethods]
impl AutoResearch {
    #[new]
    #[pyo3(signature = (cache_dir=None))]
    fn new(cache_dir: Option<String>) -> Self {
        Self { cache_dir }
    }

    #[pyo3(signature = (num_shards=None, download_workers=None))]
    fn prepare(
        &self,
        num_shards: Option<isize>,
        download_workers: Option<usize>,
    ) -> PyResult<PyPrepareSummary> {
        prepare(self.cache_dir.clone(), num_shards, download_workers)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (depth=None, total_batch_size=None, device_batch_size=None, eval_batch_size=None, learning_rate=None, weight_decay=None))]
    fn train(
        &self,
        depth: Option<usize>,
        total_batch_size: Option<usize>,
        device_batch_size: Option<usize>,
        eval_batch_size: Option<usize>,
        learning_rate: Option<f64>,
        weight_decay: Option<f64>,
    ) -> PyResult<PyRunSummary> {
        train(
            self.cache_dir.clone(),
            depth,
            total_batch_size,
            device_batch_size,
            eval_batch_size,
            learning_rate,
            weight_decay,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (cache_dir=None, num_shards=None, download_workers=None))]
fn prepare(
    cache_dir: Option<String>,
    num_shards: Option<isize>,
    download_workers: Option<usize>,
) -> PyResult<PyPrepareSummary> {
    let paths = resolve_paths(cache_dir).map_err(to_py_err)?;
    let constants = CoreConstants::default();
    let mut cfg = PrepareConfig::default();
    if let Some(raw) = num_shards {
        cfg.num_shards = if raw < 0 {
            constants.max_shard
        } else {
            raw as usize
        };
    }
    if let Some(workers) = download_workers {
        cfg.download_workers = workers.max(1);
    }

    let summary = run_prepare(&paths, constants, &cfg).map_err(to_py_err)?;
    Ok(PyPrepareSummary {
        cache_dir: summary.cache_dir.display().to_string(),
        data_dir: summary.data_dir.display().to_string(),
        tokenizer_dir: summary.tokenizer_dir.display().to_string(),
        total_shards: summary.download.total_shards,
        ready_shards: summary.download.ready_shards(),
        vocab_size: summary.tokenizer.vocab_size,
        bos_token_id: summary.tokenizer.bos_token_id,
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (cache_dir=None, depth=None, total_batch_size=None, device_batch_size=None, eval_batch_size=None, learning_rate=None, weight_decay=None))]
fn train(
    cache_dir: Option<String>,
    depth: Option<usize>,
    total_batch_size: Option<usize>,
    device_batch_size: Option<usize>,
    eval_batch_size: Option<usize>,
    learning_rate: Option<f64>,
    weight_decay: Option<f64>,
) -> PyResult<PyRunSummary> {
    let paths = resolve_paths(cache_dir).map_err(to_py_err)?;
    let constants = CoreConstants::default();
    let mut cfg = TrainConfig::default();

    if let Some(v) = depth {
        cfg.depth = v;
    }
    if let Some(v) = total_batch_size {
        cfg.total_batch_size = v;
    }
    if let Some(v) = device_batch_size {
        cfg.device_batch_size = v;
    }
    if let Some(v) = eval_batch_size {
        cfg.eval_batch_size = v;
    }
    if let Some(v) = learning_rate {
        cfg.learning_rate = v;
    }
    if let Some(v) = weight_decay {
        cfg.weight_decay = v;
    }

    let summary = run_train(&paths, constants, &cfg).map_err(to_py_err)?;
    Ok(summary.into())
}

#[pyfunction]
fn rust_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn pyautoresearch(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<AutoResearch>()?;
    module.add_class::<PyPrepareSummary>()?;
    module.add_class::<PyRunSummary>()?;
    module.add_function(wrap_pyfunction!(prepare, module)?)?;
    module.add_function(wrap_pyfunction!(train, module)?)?;
    module.add_function(wrap_pyfunction!(rust_version, module)?)?;
    Ok(())
}

fn resolve_paths(cache_dir: Option<String>) -> Result<CachePaths> {
    match cache_dir {
        Some(path) => Ok(CachePaths::from_cache_dir(path.into())),
        None => CachePaths::new_default(),
    }
}

fn to_py_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}
