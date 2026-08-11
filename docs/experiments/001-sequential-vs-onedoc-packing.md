# 001 — Does sequential packing hurt a recurrent-state model?

**Status:** treatment arm run and stopped early 2026-08-10 with a clear
negative result — one-doc continuation makes clean-input loss *worse* and heals
far too slowly to repay it. Control arm running. See [Results](#results).

## Question

Sequential packing cuts the corpus into `seq_len` windows without regard to
document boundaries, so a single row usually holds pieces of several unrelated
documents. For a transformer that is only an attention-masking question. For
Megalodon it is not: CEMA and TimestepNorm carry **recurrent state** along the
row, and `data.mask_boundary_loss` removes the cross-document next-token pairs
from the loss without cleaning that state. Every document after the first in a
row is therefore conditioned on the tail of an unrelated document.

> Does that contamination measurably cost anything, and is it worth the ~2x
> throughput that avoiding it costs?

The measurement is eval loss on a fixed held-out split, comparing two
continuations of one checkpoint that differ only in packing.

## Part 1 — the shared baseline

100,000 steps of sequential packing, finished 2026-08-07. This is the starting
point for both arms, not an arm itself.

- Config: `configs/custom/finepdfs_200m_2048-packed-fast.yaml` (gitignored)
- Run dir: `runs/chomp-200-2608-part1`, final checkpoint `checkpoints/100000`
- 188,777,472 params, `seq_len` 2048, `batch_size` 8 x `grad_accum` 8 =
  131,072 tokens/step, SwiGLU FFN, Muon
- Launched with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` (mandatory — see
  [training](../training.md))

Final `metrics.jsonl` line at step 100,000:

| metric | value |
|---|---|
| `eval_loss` | 2.8899 |
| `tokens_seen` | 13,094,263,113 (69 tokens/param) |
| `docs_seen` | 6,542,213 |
| `packing_utilization` | 1.000 |
| `segments_per_seq` mean / min / max | 1.75 / 1 / 6 |
| `boundary_transitions` per step | 48 |
| `step_time_s` | 1.795 |
| `tokens_per_sec` | 72,964 |
| `peak_memory_gb` | 29.691 |
| `lr` | 3.0e-5 (min; the cosine is fully spent) |

`segments_per_seq_mean` of 1.75 is the exposure being tested: three quarters of
all rows contain at least one document boundary that the recurrent state
crossed.

Eval trajectory over the last three evals was 2.8957 (95k), 2.8994 (97.5k),
2.8899 (100k) — a ~0.005-nat band. **Any phase-2 effect smaller than about
0.01 nats is inside the noise of this instrument** and should not be called a
result.

## Design

Two arms, both `--resume latest` from `runs/chomp-200-2608-part1`'s step-100000
checkpoint, each into its **own copy** of the run dir (resume writes in place,
and `checkpoint.max_to_keep: 3` would otherwise evict part 1's history):

```bash
cp -r runs/chomp-200-2608-part1 runs/chomp-200-2608-p2-onedoc
cp -r runs/chomp-200-2608-part1 runs/chomp-200-2608-p2-seq
```

**Treatment — one document per row.** `configs/custom/finepdfs_200m_2048-onedoc-part2.yaml`:

```yaml
packing_mode: bin
packing_max_docs_per_bin: 1
packing_strict_segments: false
```

`bin` with a one-document cap makes `_place_first_fit` reject every non-empty
bin, so each row holds exactly one document (or one capacity-sized chunk of a
long one) and is right-padded. CEMA and TimestepNorm state never crosses a
document boundary. `packing_strict_segments` stays false because it is
redundant at one segment per row and costs ~2x.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 chomp train \
  configs/custom/finepdfs_200m_2048-onedoc-part2.yaml \
  --run-dir runs/chomp-200-2608-p2-onedoc --resume latest
```

**Control — keep packing sequential.** No config edit; the baseline config with
a longer stop point:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 chomp train \
  configs/custom/finepdfs_200m_2048-packed-fast.yaml \
  --run-dir runs/chomp-200-2608-p2-seq --resume latest \
  -o train.steps=125000 -o checkpoint.resume_compat=warn
```

`resume_compat: warn` is required on the control arm too, because part 1's
checkpoint predates `data.eval_packing` and its metadata records no value for
it. That is the only strict-level entry the change adds; see
[the eval instrument](#the-eval-instrument) below.

Serialize the two — there is one GPU.

### The eval instrument

Evaluation no longer inherits the training packer. `data.eval_packing` defaults
to `onedoc`, so both arms score the same held-out documents one-per-row,
padded, under the condition generation actually runs in — a single document in
context, no CEMA/TimestepNorm state carried in from an unrelated one. Without
that, each arm would be measured by a device the arm itself changed, and the
two `eval_loss` series would not be comparable.

The offset this introduces was measured directly, by restoring part 1's
step-100000 weights and scoring the same 1000 held-out documents twice on the
same process (2026-08-10, RTX 5090):

| instrument | eval_loss | valid tokens | batches |
|---|---|---|---|
| `eval_packing: train` (sequential, part 1's own) | **2.8899** | 1,863,259 | 114 |
| `eval_packing: onedoc` | **2.8820** | 1,863,614 | 195 |

The `train` figure reproduces part 1's logged step-100000 `eval_loss` to four
decimals, which is what makes the `onedoc` figure from the same process
trustworthy. One-doc eval reads **0.0079 nats lower** on identical weights —
there are no contaminated prefixes left to mispredict after. The 195-vs-114
batch count is the padding cost, consistent with ~0.6 utilization.

**Part 1's clean-instrument baseline is 2.8820; that is the number phase 2 has
to beat.** The 30-step probe below recorded 2.8849, which is 0.0029 too high:
it was taken *after* some one-doc training steps rather than on the untouched
checkpoint. To reproduce, score the checkpoint directly rather than reading a
probe's first eval — restore params with `restore_params_only`, collect the
documents once with `load_or_create_eval_tokens`, then sum
`make_eval_step` over `build_eval_iterator` for each `eval_packing` value and
divide by the returned token count. There is no `chomp eval` subcommand.

`data.eval_packing` is fingerprinted, so switching it mid-run is caught like
any other eval-selection change. Part 1's checkpoint predates the field, which
is why both arms need `resume_compat: warn` — the expected warning entries are
`train.steps` and `data.eval_packing` on the control, plus the four packing
knobs on the treatment. Anything else in that list is an accident.

### Held constant

- **Learning rate.** `optim.decay_steps: 98000` is pinned, so the cosine is
  already spent at step 100,000 and both arms run at a constant 3e-5. Raising
  `train.steps` extends at the minimum LR rather than starting a new cosine, so
  the schedule is not a confound. This is why `decay_steps` is pinned rather
  than derived from `train.steps`.
- **Eval split.** `data.hf_eval_split: null` partitions the train split by
  content hash, and neither arm changes the hash or the seed, so both evaluate
  on the same documents part 1 did.
- **Eval instrument.** `data.eval_packing: onedoc` in both arms, so the rows
  those documents are laid out in are identical regardless of how each arm
  trains. See [the eval instrument](#the-eval-instrument).
- **Corpus position.** Both arms continue the stream from token 13.09B rather
  than replaying it. This required a code change (below).
- **Optimizer state, RNG, step counter.** Carried by `--resume`;
  `--init-from` would reset all of them plus the data position, which is why it
  is the wrong tool here.

### The one that is not held constant

`train.steps` differs from the checkpoint's value in both arms. Under
`checkpoint.resume_compat: strict` that is a warning rather than an error only
because `decay_steps` is pinned. Both arms nonetheless need `resume_compat:
warn`: the treatment because a `data.packing_mode` change is a hard error under
`strict`, and both because part 1's metadata records no `data.eval_packing`.
**Read the resume warning block at startup and confirm the only entries are the
packing knobs, `data.eval_packing`, and `train.steps`.** Anything else in that list
is an accident, not this experiment.

## Confound: equal steps is not equal tokens

One-document rows waste the tail of every row. Measured over 12,525 steps of
the treatment arm, `packing_utilization` averaged **0.61** — so at equal steps
the treatment arm sees roughly 39% fewer valid tokens than the control:

| | steps | utilization | valid tokens | wall clock |
|---|---|---|---|---|
| control (sequential) | 25,000 | 1.000 | ~3.28B | ~12.4 h |
| treatment (one-doc) | 25,000 | 0.61 | ~2.00B | ~12.4 h |
| treatment, token-matched | ~41,000 | 0.61 | ~3.28B | ~20 h |

(`packing_utilization` is measured; token counts and wall clock are arithmetic
from it and from `step_time_s`.)

The 30-step probe read 0.534 and
`configs/custom/finepdfs_200m_2048-onedoc-62k.yaml` claims 0.701 in its header;
both were short-window samples of a heavy-tailed length distribution. The
12,525-step arm settles it between them at ~0.61.

**Plan: run equal steps first**, because the result is decisive in one
direction and cheap:

- Treatment matches or beats control *despite* 47% fewer tokens → contamination
  is real and costs more than the padding does. Done.
- Treatment loses by more than ~0.01 nats → ambiguous, could be the token
  deficit alone. Pay for the ~47k-step token-matched arm to disambiguate.
- Difference under ~0.01 nats → inside the eval band established above; report
  as "no detectable effect at this scale" and keep sequential for the
  throughput.

Do not report a bare "part 2 improved / did not improve" without the token
count beside it.

## Probe (2026-08-07, 30 steps, scratch copy, exit 0)

Run before committing to the full arms, to confirm the mechanism works on the
real checkpoint rather than on a smoke config.

| check | result |
|---|---|
| `segments_per_seq` mean / min / max | **1.0 / 1 / 1** — separation achieved |
| stream continued? | `tokens_seen` 13,094,263,113 → 13,096,527,178 |
| `docs_seen` | 6,540,950 (below part 1's logged 6,542,213 — the `grain_prefetch: 32` lead, not a regression) |
| `packing_utilization` | 0.534 |
| `step_time_s` | 1.778 — indistinguishable from sequential |
| `tokens_per_sec` (valid) | 41,711 |
| `peak_memory_gb` | 29.136 — fits the 0.90 pool |
| eval under one-doc packing | 2.8849 / 2.8885 / 2.8902 |

**The eval instrument moves less than the effect.** At the time of the probe,
eval batches were packed with the same config as training, so switching packing
also switched the measuring device. The probe read 2.8849 one-doc against part
1's sequential 2.8899 and called the shift 0.005 nats. Both halves of that were
slightly wrong: direct scoring of the untouched checkpoint (see [the eval
instrument](#the-eval-instrument)) puts the offset at **0.0079** and the one-doc
baseline at **2.8820**, because the probe's first eval already included some
one-doc training steps. That measurement is what
[`data.eval_packing`](../config-reference.yaml) was subsequently built on: the
instrument is now pinned to one-doc for both arms rather than following each
arm's training packer, so the shift is a one-time offset at step 100,000
instead of a per-arm confound.

## Prerequisite fixes

### Resuming across a packing-mode change

Resuming across a `packing_mode` change was impossible before commit
`2b24331`: it died with `KeyError: 'pending_tokens_i32_b64'`. The two packer
families store disjoint state — sequential carries a token remainder, `bin`
carries document and row queues — and neither could load the other's buffer.
The only alternative, `--init-from`, resets the corpus position and would have
replayed the 13.1B tokens the model had already trained on.

Producer state is `{"text": ..., "packer": ...}` and only the packer half is
family-specific. On a family mismatch the stream position and the document
counters now restore normally and only the buffers are dropped, with a warning.
The discarded tokens are whatever the old packer had accepted but not emitted:
under one window plus at most one document tail. Reachable only under
`resume_compat: warn`; `strict` still refuses. See
[packing](../packing.md) and [checkpointing](../checkpointing.md).

### An eval instrument that survives a packing change

Evaluation used to pack its rows with the training knobs, so each arm would
have been scored by a device it had just changed. `data.eval_packing` splits
the two: eval row layout is now its own decision, defaulting to one document
per row, and the training packer no longer reaches it. Selection is untouched —
the same held-out documents either way. See
[training](../training.md#evaluation).

## Results

### Treatment (one-doc) — stopped early at step 112,525 of 125,000

Run dir `runs/chomp-200-2608-p2-onedoc`, wandb
`200m-finepdfs-onedoc-seq2048-b8x8-swiglu-part2-25k` (`bp6a7nsj`), 2026-08-10.
Stopped deliberately after 12,525 of 25,000 steps; the checkpoint at 112,500 is
intact and `--resume latest` continues it.

All values on the pinned one-doc instrument, against the measured 2.8820
baseline:

| step | eval_loss | Δ baseline | Δ previous |
|---|---|---|---|
| 100,000 | 2.8820 | — | — |
| 102,500 | 2.9046 | +0.0226 | +0.0226 |
| 105,000 | 2.9082 | +0.0262 | +0.0036 |
| 107,500 | 2.9142 | +0.0322 | +0.0060 |
| 110,000 | 2.9123 | +0.0303 | −0.0019 |
| 112,500 | 2.9104 | +0.0284 | −0.0019 |

**One-doc continuation makes clean-input loss worse, and heals too slowly to
repay it.** Loss rose for 7,500 steps to a peak of +0.0322, then began a steady
linear recovery of −0.0019 per 2,500 steps. Extrapolating that rate, returning
to 2.8820 needs roughly **37,400 further steps** — three times the budget that
remained, and that assumes a linear recovery rather than the more usual decay
toward an asymptote above baseline.

This is not the token deficit. A token deficit slows improvement; it cannot
push loss above the starting point. Nor is it the schedule: part 1's last 5,000
steps at this same constant 3e-5 moved eval ~0.000. The cause is the packing
change.

**Mechanism is not established.** An earlier draft of this section blamed lost
long-range context — that under sequential packing a long document spans
consecutive rows and CEMA/TimestepNorm carry context across them. That is
false: consecutive windows land in different *rows*, and no recurrent state
crosses rows or steps (`TrainState` is step/params/opt_state/rng only, and
training never uses a cache). Every row starts cold under both schemes.

The surviving candidate is the effective-batch one. One-doc rows deliver ~39%
fewer valid tokens per optimizer step at an unchanged LR, and a smaller
effective batch at fixed LR raises the SGD noise floor — which produces a rise
to a worse plateau rather than a transient, matching the shape observed. This
predicts that `bin` packing without a per-row document cap, which keeps
utilization at ~1.0, would *not* show the damage. That arm has not been run;
see below.

Operationally: `segments_per_seq` held at exactly 1.0/1/1, `step_time_s` 1.78
(indistinguishable from sequential), `peak_memory_gb` 29.2 under the 0.90 pool,
`tokens_seen` 13.094B → 14.096B. Measured packing utilization over the whole
arm was **~0.61**, not the 0.534 the 30-step probe suggested — so the
equal-steps token deficit is ~1.65x, not ~1.9x. The 0.701 in the
`-onedoc-62k` header remains the high end; ~0.61 is the settled figure at this
shape.

### Control (sequential)

Running. Paired against the treatment at identical steps, same checkpoint,
optimizer state, RNG, corpus position, LR, and eval instrument:

| step | control | treatment | gap |
|---|---|---|---|
| 102,500 | 2.8836 (+0.0016) | 2.9046 (+0.0226) | 0.0210 |
| 105,000 | 2.8815 (−0.0005) | 2.9082 (+0.0262) | 0.0267 |
| 107,500 | 2.8805 (−0.0015) | 2.9142 (+0.0322) | 0.0337 |

The control is not flat — it drifts down ~0.0016 per 2,500 steps, so part 1's
last-5k flatness understated ongoing progress at min LR. Consequence: most of
the treatment's late "recovery" (−0.0019 per 2,500 steps) is ordinary training,
not healing. The packing-attributable component is ~0.0003 per 2,500 steps,
indistinguishable from zero at this band, so the arms do not converge on any
practical horizon. (Rate differences of 0.0003–0.0016 against a ~0.005
single-point band are suggestive only; the 0.0337 gap is the solid part.)

## Not tested: `bin` packing

Only two of three packing regimes have been run. Note the config naming trap:
`-packed-` in `configs/custom/` means **`packing_mode: sequential`**, not `bin`.

| regime | config | utilization | run? |
|---|---|---|---|
| `sequential` | `-packed-fast`, `-packed-62k` | 1.0 | yes (part 1, control) |
| `bin` + `max_docs_per_bin: 1` | `-onedoc-*` | ~0.61 | yes (treatment) |
| `bin`, no doc cap, `strict_segments: true` | none | ~1.0 | **no** |

`packing_strict_segments: true` in the `-packed-` configs is inert: it applies
only to `bin`/`multipack`. The untested third row is the clean test of the
original question — it removes cross-document contamination via full state
isolation (see [packing](../packing.md)) while keeping utilization at ~1.0, so
it carries neither the padding waste nor the token deficit that the one-doc arm
introduced. It costs ~2x attention FLOPs. It is also the arm that discriminates
the effective-batch explanation above from a genuine contamination effect.

<!--
When an arm finishes, record here: final eval_loss and the step it came from,
the eval trajectory over the last ~5 evals (not just the last point — see the
0.005-nat band above), final tokens_seen, measured packing_utilization over the
whole arm, and the wandb run name. Then update Status and the index row in
README.md.
-->
