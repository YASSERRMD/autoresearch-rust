# autoresearch-rust

Rust-first implementation of `karpathy/autoresearch` with Python bindings.

## Appreciation

This project is built on the ideas and baseline workflow from [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch). Appreciation to Andrej Karpathy for open-sourcing the original project.

## What this repo provides

- High-throughput data preparation in Rust:
  - shard download with retries and parallel workers
  - parquet text ingestion
  - BPE tokenizer training and token-byte lookup generation
- Rust training engine (feature-gated):
  - GPT-style model in Candle (`train` feature)
  - packed BOS-aligned dataloader
  - fixed time-budget training loop
  - BPB evaluation metric
  - optional external acceleration hook for training and inference stages
- Python FFI via `pyo3`:
  - `prepare(...)` and `train(...)`
  - object-oriented `AutoResearch` API

## Repository layout

- `crates/autoresearch-core`: core library (prep, tokenizer, train engine)
- `crates/autoresearch-cli`: Rust CLI for `prepare` and `train`
- `crates/pyautoresearch`: Python extension module (`pyo3`)
- `tmp/autoresearch`: upstream reference clone (ignored)

## Rust usage

### 1) Prepare data/tokenizer

```bash
cargo run -p autoresearch-cli -- prepare --num-shards 10 --download-workers 8
```

Optional cache override:

```bash
cargo run -p autoresearch-cli -- prepare --cache-dir /path/to/cache
```

### 2) Run training (Candle backend)

```bash
cargo run -p autoresearch-cli --features train -- train
```

With custom knobs:

```bash
cargo run -p autoresearch-cli --features train -- train \
  --depth 8 \
  --total-batch-size 524288 \
  --device-batch-size 128 \
  --eval-batch-size 128 \
  --learning-rate 0.0004 \
  --weight-decay 0.1 \
  --accelerator-cmd "barqtrain --warmup --mode train" \
  --inference-accelerator-cmd "barqtrain --warmup --mode inference"
```

## Python usage

### 1) Build/install extension in your environment

```bash
python3 -m pip install maturin
maturin develop -m crates/pyautoresearch/Cargo.toml --release
```

### 2) Use Python API

```python
import pyautoresearch

prep = pyautoresearch.prepare(num_shards=10, download_workers=8)
print(prep.vocab_size, prep.ready_shards)

run = pyautoresearch.train(
    depth=8,
    total_batch_size=2**19,
    accelerator_cmd="barqtrain --warmup --mode train",
    inference_accelerator_cmd="barqtrain --warmup --mode inference",
)
print(run.val_bpb, run.training_seconds)
```

Or use the class API:

```python
from pyautoresearch import AutoResearch

ar = AutoResearch(cache_dir=None)
ar.prepare(num_shards=10)
result = ar.train(depth=8)
print(result.val_bpb)
```

## Performance notes

- Use release mode for production runs: `--release`
- Data prep scales with worker count (`--download-workers`)
- Tokenizer and parquet ingestion are CPU-bound; run on a machine with high memory bandwidth
- Training backend is Candle; this repo currently defaults to CPU device selection in code for portability

## About `YASSERRMD/BarqTrain`

This repo supports optional integration with [YASSERRMD/BarqTrain](https://github.com/YASSERRMD/BarqTrain) through:

- `--accelerator-cmd` (CLI) / `accelerator_cmd` (Python) for training setup
- `--inference-accelerator-cmd` (CLI) / `inference_accelerator_cmd` (Python) for inference/evaluation setup

Appreciation: thanks to `YASSERRMD` for publishing BarqTrain.

## License

MIT (see `LICENSE`)
