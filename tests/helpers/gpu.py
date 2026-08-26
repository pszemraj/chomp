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


def run_export_placement_smoke(run_dir: str) -> None:
    """Train one megalodon step on GPU, then export and check where the params went.

    Export moves bytes and computes nothing, so it must not put a second copy of
    the parameters on the accelerator: the end-of-run export runs inside a
    process still holding a memory pool sized for that run's train step, and a
    finished long run is the worst possible moment to OOM. The assertion is only
    meaningful on a GPU host, which is why it lives here.

    :param str run_dir: Fresh run directory.
    :raises AssertionError: If any exported parameter was not on the host.
    """
    from dataclasses import replace as dc_replace

    import jax
    import megalodon_jax

    from chomp.config import Config
    from chomp.export import export_checkpoint
    from chomp.train import run
    from tests.helpers.config_factories import make_tiny_megalodon_model

    cfg = Config()
    cfg = dc_replace(
        cfg,
        model=make_tiny_megalodon_model(vocab_size=256, compute_dtype="bfloat16"),
        data=dc_replace(
            cfg.data,
            backend="local_text",
            local_text="real megalodon gpu export path",
            repeat=True,
            max_eval_samples=0,
            packing_mode="sequential",
            tokenizer=dc_replace(cfg.data.tokenizer, kind="byte", add_bos=False, add_eos=False),
        ),
        train=dc_replace(
            cfg.train,
            steps=1,
            batch_size=1,
            seq_len=32,
            grad_accum=1,
            allow_cpu=False,
            log_every=1,
            eval_every=0,
        ),
        optim=dc_replace(cfg.optim, lr=1e-3, warmup_steps=0, min_lr_ratio=0.0),
        checkpoint=dc_replace(cfg.checkpoint, save_every=1, async_save=False),
        logging=dc_replace(
            cfg.logging, run_dir=run_dir, wandb=dc_replace(cfg.logging.wandb, enabled=False)
        ),
    )

    assert jax.default_backend() == "gpu", jax.default_backend()
    output_dir = run(cfg)

    # The run exported itself on the way out, and that export must load.
    assert (output_dir / "export" / "model.safetensors").is_file()

    platforms: set[str] = set()
    original = megalodon_jax.save_checkpoint

    def _spy(model: object, path: object) -> None:
        """Record parameter device platforms, then save normally."""
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(model) if hasattr(leaf, "devices")]
        platforms.update(device.platform for leaf in leaves for device in leaf.devices())
        original(model, path)

    megalodon_jax.save_checkpoint = _spy
    try:
        export_checkpoint(output_dir, output_dir / "reexport", verify=False)
    finally:
        megalodon_jax.save_checkpoint = original

    assert platforms == {"cpu"}, platforms
    print("MEGALODON_GPU_EXPORT_OK")


def run_policy_dtype_equivalence_smoke(run_dir: str) -> None:
    """Prove the policy-dtype export computes the identical thing, on a real GPU.

    The CPU suite asserts this through greedy generation. That instrument does
    not survive the move to a GPU: this repo runs fast, non-deterministic
    kernels, so the *same* fp32 export decodes to different text in two
    processes, and greedy decoding turns one flipped argmax into a different
    continuation. Comparing text on GPU would therefore fail for the fp32
    export against itself.

    What is well posed on GPU is the model's own arithmetic, and that is exact:
    the uncached forward, the cached prefill, and teacher-forced decode steps
    all match bit for bit. That is the property the variant actually claims --
    the forward pass casts these tensors to bf16 anyway, so the export drops
    only bits that never reached the arithmetic.

    :param str run_dir: Fresh run directory.
    :raises AssertionError: If the two exports disagree anywhere, or the file is not mixed.
    """
    from dataclasses import replace as dc_replace

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from chomp.config import Config
    from chomp.export import DTYPE_POLICY, WEIGHTS_FILENAME, export_checkpoint, load_export
    from chomp.train import run
    from tests.helpers.config_factories import make_tiny_megalodon_model

    cfg = Config()
    cfg = dc_replace(
        cfg,
        model=make_tiny_megalodon_model(vocab_size=256, compute_dtype="bfloat16"),
        data=dc_replace(
            cfg.data,
            backend="local_text",
            local_text="real megalodon gpu policy dtype export",
            repeat=True,
            max_eval_samples=0,
            packing_mode="sequential",
            tokenizer=dc_replace(cfg.data.tokenizer, kind="byte", add_bos=False, add_eos=False),
        ),
        train=dc_replace(
            cfg.train,
            steps=1,
            batch_size=1,
            seq_len=32,
            grad_accum=1,
            allow_cpu=False,
            log_every=1,
            eval_every=0,
        ),
        optim=dc_replace(cfg.optim, lr=1e-3, warmup_steps=0, min_lr_ratio=0.0),
        checkpoint=dc_replace(cfg.checkpoint, save_every=1, async_save=False),
        logging=dc_replace(
            cfg.logging, run_dir=run_dir, wandb=dc_replace(cfg.logging.wandb, enabled=False)
        ),
    )

    assert jax.default_backend() == "gpu", jax.default_backend()
    output_dir = run(cfg)

    canonical = export_checkpoint(output_dir, output_dir / "exp-fp32").export_dir
    policy = export_checkpoint(output_dir, output_dir / "exp-policy", dtype=DTYPE_POLICY).export_dir

    from safetensors import safe_open

    with safe_open(str(policy / WEIGHTS_FILENAME), framework="numpy") as handle:
        names = handle.keys()  # noqa: SIM118
        dtypes = {name: str(handle.get_slice(name).get_dtype()) for name in names}
    assert "F32" in dtypes.values(), "no tensor stayed fp32; the policy became a blanket cast"
    assert "BF16" in dtypes.values(), "no tensor became bf16"

    a = load_export(canonical)
    b = load_export(policy)
    model_a = eqx.combine(a.params, a.static)
    model_b = eqx.combine(b.params, b.static)

    seq = [5, 91, 40, 17, 88, 231, 6, 45, 200, 12, 77, 34]
    for length in (4, 8, 12):
        ids = jnp.array([seq[:length]], dtype=jnp.int32)
        la, _ = model_a(ids, deterministic=True)
        lb, _ = model_b(ids, deterministic=True)
        assert bool(jnp.array_equal(la, lb)), f"uncached forward differs at length {length}"

    prefill = jnp.array([seq[:8]], dtype=jnp.int32)
    la, cache_a = model_a(prefill, deterministic=True, return_cache=True)
    lb, cache_b = model_b(prefill, deterministic=True, return_cache=True)
    assert bool(jnp.array_equal(la, lb)), "cached prefill differs"

    for step, token in enumerate(seq[8:]):
        step_ids = jnp.array([[token]], dtype=jnp.int32)
        la, cache_a = model_a(step_ids, cache=cache_a, deterministic=True, return_cache=True)
        lb, cache_b = model_b(step_ids, cache=cache_b, deterministic=True, return_cache=True)
        assert bool(jnp.array_equal(la, lb)), f"cached decode differs at step {step}"

    print("MEGALODON_GPU_POLICY_DTYPE_OK")


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
