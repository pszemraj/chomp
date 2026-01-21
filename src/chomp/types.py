"""Core pytrees and shared types.

Keep this file small: it defines the **runtime contracts** between subsystems.

- `Batch` is what the dataloader yields and the compiled step consumes.
- `TrainState` is arrays-only (checkpoint friendly) by construction.

**Batch contract**

chomp standardizes on fixed shapes:
  input_ids:      [A, B, T]
  labels:         [A, B, T]
  attention_mask: [A, B, T] boolean
where:
  A = grad_accum (microbatches per optimizer update)
  B = batch_size
  T = seq_len

This is the compile-once contract. The data pipeline is responsible for producing
these fixed shapes (tokenize+pack for streaming docs).

Later, a Grain pipeline may produce [B,T] microbatches and group A of them, or produce
[A,B,T] directly. Either way, the compiled step stays fixed.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax


class Batch(eqx.Module):
    """A fixed-shape training batch.

    All fields are arrays. `attention_mask` is always present.
    """

    input_ids: jax.Array
    labels: jax.Array
    attention_mask: jax.Array


class TrainState(eqx.Module):
    """Arrays-only state for training.

    Do **not** store the full model (eqx.Module) here. Store only `params` (arrays).
    The model's `static` pytree is closed over by the compiled step.
    """

    step: jax.Array
    params: Any
    opt_state: Any
    rng: jax.Array
