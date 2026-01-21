#!/usr/bin/env python3
"""Diagnose abnormally high loss in chomp training runs.

This script inspects tokenizer/model special tokens, batch masking stats, and
optionally runs a step-0 forward pass to report loss and logit statistics.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from typing import Any

import numpy as np

IGNORE_INDEX = -100


def _count_valid_targets(labels: np.ndarray, attention_mask: np.ndarray | None) -> int:
    """Count valid targets after the causal shift (labels[:, 1:])."""
    tgt = labels[:, 1:]
    valid = tgt != IGNORE_INDEX
    if attention_mask is not None:
        valid &= attention_mask[:, 1:]
    return int(valid.sum())


def _masked_fraction(labels: np.ndarray, attention_mask: np.ndarray | None) -> float:
    b, t = labels.shape
    denom = b * max(t - 1, 1)
    valid = _count_valid_targets(labels, attention_mask)
    return float(1.0 - (valid / denom))


def _segment_stats(segment_ids: np.ndarray) -> dict[str, float]:
    """Compute basic segment statistics for a [B, T] segment id tensor."""
    seg_counts: list[int] = []
    seg_lens: list[int] = []

    for row in segment_ids:
        row = row[row > 0]
        if row.size == 0:
            seg_counts.append(0)
            continue
        changes = np.count_nonzero(row[1:] != row[:-1])
        seg_counts.append(1 + changes)

        start = 0
        for j in range(1, row.size):
            if row[j] != row[j - 1]:
                seg_lens.append(j - start)
                start = j
        seg_lens.append(row.size - start)

    return {
        "avg_segments_per_seq": float(np.mean(seg_counts)) if seg_counts else 0.0,
        "min_segments_per_seq": float(np.min(seg_counts)) if seg_counts else 0.0,
        "max_segments_per_seq": float(np.max(seg_counts)) if seg_counts else 0.0,
        "avg_segment_len": float(np.mean(seg_lens)) if seg_lens else 0.0,
        "min_segment_len": float(np.min(seg_lens)) if seg_lens else 0.0,
        "max_segment_len": float(np.max(seg_lens)) if seg_lens else 0.0,
    }


def _decoder(tokenizer: Any) -> Any | None:
    """Return a decode function if available."""
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode
    if hasattr(tokenizer, "_tok") and hasattr(tokenizer._tok, "decode"):
        return tokenizer._tok.decode
    return None


def _preview_decodes(
    tokenizer: Any, input_ids: np.ndarray, segment_ids: np.ndarray, max_spans: int
) -> list[str]:
    """Decode a few contiguous segment spans from the first sample."""
    decode = _decoder(tokenizer)
    if decode is None:
        return []

    ids = input_ids[0].tolist()
    seg = segment_ids[0].tolist()

    spans: list[tuple[int, int]] = []
    start = None
    for i, s in enumerate(seg):
        if s <= 0:
            if start is not None:
                spans.append((start, i))
                start = None
            continue
        if start is None:
            start = i
        elif s != seg[i - 1]:
            spans.append((start, i))
            start = i
    if start is not None:
        spans.append((start, len(seg)))

    previews: list[str] = []
    for a, b in spans[:max_spans]:
        text = decode(ids[a:b])
        previews.append(f"[seg span {a}:{b}] {text[:400]!r}")
    return previews


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a JAX array or numpy array to numpy on host."""
    try:
        import jax

        return np.asarray(jax.device_get(x))
    except Exception:
        return np.asarray(x)


def _token_info(tokenizer: Any) -> dict[str, int | None]:
    """Extract vocab and special token ids when available."""
    info: dict[str, int | None] = {"vocab_size": None, "bos": None, "eos": None, "pad": None}
    try:
        info["vocab_size"] = int(len(tokenizer))
    except Exception:
        info["vocab_size"] = None

    info["bos"] = getattr(tokenizer, "bos_token_id", None)
    info["eos"] = getattr(tokenizer, "eos_token_id", None)
    info["pad"] = getattr(tokenizer, "pad_token_id", None)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to chomp YAML config.")
    ap.add_argument("--batches", type=int, default=2, help="Number of batches to inspect.")
    ap.add_argument(
        "--max-decode-spans",
        type=int,
        default=3,
        help="Number of decoded spans to print per batch.",
    )
    ap.add_argument("--no-model", action="store_true", help="Skip model forward/loss.")
    ap.add_argument("--force-init-mode", type=str, default=None, help="Override init_mode.")
    ap.add_argument("--force-chunk-size", type=int, default=None, help="Override chunk_size.")
    args = ap.parse_args()

    from chomp.config import load_config, validate_config
    from chomp.data import build_train_iterator, prepare_tokenizer_and_config
    from chomp.model import build_model, training_loss
    from chomp.types import Batch

    cfg = load_config(args.config)
    if args.force_init_mode is not None:
        cfg = replace(cfg, model=replace(cfg.model, init_mode=args.force_init_mode))
    if args.force_chunk_size is not None:
        cfg = replace(cfg, model=replace(cfg.model, chunk_size=int(args.force_chunk_size)))
    validate_config(cfg)

    cfg, tokenizer = prepare_tokenizer_and_config(cfg)
    tok_info = _token_info(tokenizer)

    print("\n=== Tokenizer vs Model Special Tokens ===")
    print(f"tokenizer kind: {cfg.data.tokenizer.kind}")
    print(f"tokenizer vocab_size: {tok_info['vocab_size']}")
    print(f"model vocab_size: {cfg.model.vocab_size}")
    print(
        "tokenizer bos/eos/pad: "
        f"{tok_info['bos']} / {tok_info['eos']} / {tok_info['pad']}"
    )
    print(
        "model     bos/eos/pad: "
        f"{cfg.model.bos_token_id} / {cfg.model.eos_token_id} / {cfg.model.pad_token_id}"
    )
    if tok_info["pad"] is not None and tok_info["eos"] is not None:
        if tok_info["pad"] == tok_info["eos"]:
            print("WARNING: tokenizer pad_token_id == eos_token_id (pad==eos).")
    print(f"cfg.model.init_mode: {cfg.model.init_mode}")
    print(f"cfg.model.chunk_size: {cfg.model.chunk_size}")
    print(f"cfg.train.grad_accum: {cfg.train.grad_accum}")
    print(f"cfg.train.batch_size: {cfg.train.batch_size}")
    print(f"cfg.train.effective_batch: {cfg.train.grad_accum * cfg.train.batch_size}\n")

    it = build_train_iterator(cfg, tokenizer=tokenizer)

    for bi in range(args.batches):
        batch = next(it)
        inp = _to_numpy(batch.input_ids)
        lab = _to_numpy(batch.labels)
        attn = _to_numpy(batch.attention_mask) if batch.attention_mask is not None else None
        seg = _to_numpy(batch.segment_ids) if batch.segment_ids is not None else None

        a_dim, micro_b, t_dim = inp.shape
        print(f"=== Batch {bi} ===")
        print(f"shape: A={a_dim}, microB={micro_b}, T={t_dim}")

        total_valid = 0
        total_denom = 0
        masked_fracs = []
        for a in range(a_dim):
            labels2 = lab[a]
            attn2 = attn[a] if attn is not None else None
            v = _count_valid_targets(labels2, attn2)
            denom = labels2.shape[0] * max(labels2.shape[1] - 1, 1)
            total_valid += v
            total_denom += denom
            masked_fracs.append(_masked_fraction(labels2, attn2))

        print(
            f"valid_targets: {total_valid} / {total_denom} "
            f"({(total_valid / max(total_denom, 1)):.3%} valid)"
        )
        print(f"masked_fraction avg: {float(np.mean(masked_fracs)):.3%}")

        if seg is not None:
            stats = _segment_stats(seg[0])
            print(
                "segment stats (microbatch 0): "
                + ", ".join(f"{k}={v:.2f}" for k, v in stats.items())
            )
            previews = _preview_decodes(tokenizer, inp[0], seg[0], args.max_decode_spans)
            if previews:
                print("decoded previews (microbatch 0, sample 0):")
                for p in previews:
                    print(f"  {p}")

        eos_id = cfg.model.eos_token_id
        if eos_id is not None:
            eos_targets = int(
                ((lab[:, :, 1:] == eos_id) & (lab[:, :, 1:] != IGNORE_INDEX)).sum()
            )
            print(f"eos_targets (non-ignored, after shift): {eos_targets}")
        print()

    if args.no_model:
        return

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    batch = next(it)
    inp = jnp.asarray(batch.input_ids[0])
    lab = jnp.asarray(batch.labels[0])
    attn = jnp.asarray(batch.attention_mask[0]) if batch.attention_mask is not None else None
    seg = jnp.asarray(batch.segment_ids[0]) if batch.segment_ids is not None else None

    key = jax.random.PRNGKey(cfg.train.seed)
    key, k_model = jax.random.split(key)
    params, static = build_model(cfg, key=k_model)
    model = eqx.combine(params, static)

    use_segment_ids = bool(cfg.model.segment_masking)
    if cfg.model.backend == "dummy":
        logits = model(inp, attention_mask=attn, deterministic=True, key=None)
    else:
        logits = model(
            inp,
            attention_mask=attn,
            cache=None,
            return_cache=False,
            deterministic=True,
            key=None,
            segment_ids=seg if use_segment_ids else None,
        )

    if isinstance(logits, tuple):
        logits = logits[0]

    micro = Batch(
        input_ids=inp,
        labels=lab,
        attention_mask=attn,
        segment_ids=seg,
    )
    loss = training_loss(
        params,
        static,
        batch=micro,
        deterministic=True,
        key=None,
        use_segment_ids=use_segment_ids,
    )

    logits = jnp.asarray(logits)
    vocab = logits.shape[-1]
    expected_uniform = math.log(vocab)
    logit_std = float(jnp.std(logits))
    logit_min = float(jnp.min(logits))
    logit_max = float(jnp.max(logits))
    loss_val = float(loss)

    print("=== Model step-0 diagnostics (microbatch 0) ===")
    print(f"loss (mean CE): {loss_val:.4f}")
    print(f"expected ~log(vocab): log({vocab}) = {expected_uniform:.4f}")
    print(f"logits: std={logit_std:.4f}, min={logit_min:.4f}, max={logit_max:.4f}")
    if loss_val > expected_uniform + 5.0:
        print(
            "WARNING: loss is far above log(vocab). Check init_mode and scaling.",
        )
    if logit_std > 5.0:
        print("WARNING: logits std is very large. Check init_mode/precision.")


if __name__ == "__main__":
    main()
