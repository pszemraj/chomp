# 001 — Does sequential packing hurt a recurrent-state model?

**Status:** planned. Part 1 (the shared baseline) finished 2026-08-07. Neither
phase-2 arm has been launched.

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

The 30-step probe below measured the offset this introduces on part 1's
weights: 2.8849 one-doc against the 2.8899 sequential eval logged at step
100,000, a 0.005-nat shift — the same size as the run-to-run eval band, but it
is a real discontinuity in the series at step 100,000 in *both* arms, not a
result. **Part 1's clean-instrument baseline is 2.8849; that is the number
phase 2 has to beat.**

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

One-document rows waste the tail of every row. Measured over a 30-step probe
from the real checkpoint, `packing_utilization` was **0.534** — so at equal
steps the treatment arm sees roughly 47% fewer valid tokens than the control:

| | steps | utilization | valid tokens | wall clock |
|---|---|---|---|---|
| control (sequential) | 25,000 | 1.000 | ~3.28B | ~12.4 h |
| treatment (one-doc) | 25,000 | 0.534 | ~1.75B | ~12.4 h |
| treatment, token-matched | ~46,800 | 0.534 | ~3.28B | ~23 h |

(`packing_utilization` is measured; token counts and wall clock are arithmetic
from it and from `step_time_s`.)

Note `configs/custom/finepdfs_200m_2048-onedoc-62k.yaml` claims 0.701 in its
header. Both that and the 0.534 here are short-window samples of a heavy-tailed
length distribution and neither is settled; the real number lands somewhere in
between and a full arm will pin it down.

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

**The eval instrument barely moves.** At the time of the probe, eval batches
were packed with the same config as training, so switching packing also
switched the measuring device. On essentially unchanged weights, one-doc eval
read 2.8849 against part 1's sequential 2.8899 — a 0.005-nat shift, the same
size as the run-to-run eval band. That measurement is what
[`data.eval_packing`](../config-reference.yaml) was subsequently built on: the
instrument is now pinned to one-doc for both arms rather than following each
arm's training packer, so the shift is a one-time offset at step 100,000
instead of a per-arm confound. 2.8849 is the baseline to beat.

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

Not yet run.

<!--
When an arm finishes, record here: final eval_loss and the step it came from,
the eval trajectory over the last ~5 evals (not just the last point — see the
0.005-nat band above), final tokens_seen, measured packing_utilization over the
whole arm, and the wandb run name. Then update Status and the index row in
README.md.
-->
