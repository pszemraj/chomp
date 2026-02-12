"""Shared config builders for integration-style tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from chomp.config import Config, load_config

DEFAULT_SMALL_RUN_TEXT = "hello from chomp"


def make_small_run_cfg(
    tmp_path: Path,
    *,
    run_subdir: str = "run",
    local_text: str = DEFAULT_SMALL_RUN_TEXT,
    decay_steps: int | None = None,
) -> tuple[Config, Path]:
    """Build a tiny local-text config for fast train/checkpoint tests.

    :param Path tmp_path: Temporary directory provided by pytest.
    :param str run_subdir: Name of the run subdirectory under tmp_path.
    :param str local_text: Local text corpus for the dataset backend.
    :param int | None decay_steps: Optional optimizer decay horizon override.
    :return tuple[Config, Path]: (cfg, config_path) for smoke-sized training runs.
    """
    config_src = Path(__file__).resolve().parents[2] / "configs" / "debug_smoke.yaml"
    cfg = load_config(str(config_src))

    optim = replace(cfg.optim, warmup_steps=0)
    if decay_steps is not None:
        optim = replace(optim, decay_steps=int(decay_steps))

    cfg = replace(
        cfg,
        train=replace(
            cfg.train,
            steps=2,
            batch_size=1,
            seq_len=16,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1,
            eval_every=0,
            generate_every=0,
        ),
        data=replace(
            cfg.data,
            backend="local_text",
            repeat=True,
            packing_mode="sequential",
            packing_buffer_docs=4,
            grain_prefetch=0,
            local_text=local_text,
        ),
        checkpoint=replace(
            cfg.checkpoint,
            enabled=True,
            save_every=1,
            max_to_keep=2,
            async_save=False,
        ),
        optim=optim,
        logging=replace(
            cfg.logging,
            run_dir=str(tmp_path / run_subdir),
            console_use_rich=False,
        ),
        debug=replace(
            cfg.debug,
            nan_check=True,
            check_device_every=0,
        ),
    )
    return cfg, config_src
