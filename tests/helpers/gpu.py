"""Subprocess helpers for tests that require a visible NVIDIA GPU."""

from __future__ import annotations

import subprocess
from dataclasses import replace


def query_nvidia_gpu_names() -> list[str]:
    """Return GPU names from nvidia-smi without importing JAX.

    :return list[str]: Visible NVIDIA GPU names, or an empty list on failure.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def run_training_smoke(run_dir: str, *, backend: str) -> None:
    """Run one real training step for a GPU smoke-test backend.

    :param str run_dir: Fresh run directory.
    :param str backend: ``dummy`` or ``megalodon``.
    :raises ValueError: If backend is unsupported.
    """
    from chomp.config import Config
    from chomp.train import run
    from tests.helpers.config_factories import make_tiny_megalodon_model

    cfg = Config()
    tokenizer = replace(cfg.data.tokenizer, kind="byte", add_bos=False, add_eos=False)
    if backend == "dummy":
        model = replace(cfg.model, backend="dummy", vocab_size=256, d_model=32, dropout=0.0)
        data = replace(
            cfg.data,
            backend="local_text",
            local_text="hello from gpu",
            repeat=True,
            max_eval_samples=4,
            packing_mode="sequential",
            tokenizer=tokenizer,
        )
        jit = False
        deterministic = True
        marker = "GPU_SMOKE_OK"
    elif backend == "megalodon":
        model = make_tiny_megalodon_model(
            vocab_size=256,
            use_checkpoint=True,
            compute_dtype="bfloat16",
            loss_chunk_size=7,
        )
        data = replace(
            cfg.data,
            backend="local_text",
            local_text="real megalodon gpu smoke path with packed documents",
            repeat=True,
            max_eval_samples=0,
            packing_mode="bin",
            packing_buffer_docs=4,
            packing_strict_segments=True,
            mask_boundary_loss=True,
            tokenizer=tokenizer,
        )
        jit = True
        deterministic = False
        marker = "MEGALODON_GPU_SMOKE_OK"
    else:
        raise ValueError(f"Unsupported GPU smoke backend: {backend!r}")

    cfg = replace(
        cfg,
        model=model,
        data=data,
        train=replace(
            cfg.train,
            steps=1,
            batch_size=1,
            seq_len=32,
            grad_accum=1,
            allow_cpu=False,
            log_every=1,
            eval_every=0,
            jit=jit,
            deterministic=deterministic,
        ),
        optim=replace(cfg.optim, lr=1e-3, warmup_steps=0, min_lr_ratio=0.0),
        checkpoint=replace(cfg.checkpoint, enabled=False),
        logging=replace(
            cfg.logging,
            run_dir=run_dir,
            wandb=replace(cfg.logging.wandb, enabled=False),
        ),
    )
    output_dir = run(cfg)
    metrics_path = output_dir / cfg.logging.metrics_file
    assert metrics_path.exists() and metrics_path.read_text().strip()
    print(marker)
