import pyautoresearch


def main() -> None:
    prep = pyautoresearch.prepare(num_shards=10, download_workers=8)
    print(f"Tokenizer vocab: {prep.vocab_size} | shards ready: {prep.ready_shards}/{prep.total_shards}")

    result = pyautoresearch.train(
        depth=8,
        total_batch_size=2**19,
        device_batch_size=128,
        eval_batch_size=128,
        learning_rate=4e-4,
        weight_decay=0.1,
        accelerator_cmd="groqtrain --warmup",
    )
    print(f"val_bpb={result.val_bpb:.6f} steps={result.num_steps} seconds={result.training_seconds:.1f}")


if __name__ == "__main__":
    main()
