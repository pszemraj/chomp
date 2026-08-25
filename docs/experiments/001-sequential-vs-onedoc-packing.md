# 001 — Does sequential packing hurt a recurrent-state model?

**Status:** DONE, 2026-08-13. All three packing regimes measured against a
common baseline, then decomposed by position. In aggregate, cross-document
contamination is real but small: strict segment isolation at full utilization
buys 0.0043 nats (t = −4.1) for 2.8x the wall clock per token, while
one-document-per-row costs 0.0282 (t = +16.1). But that aggregate is a
token-weighted mean, and it hides its own shape — strict is worth 0.1319 nats
over the first 15 positions of a document, and more sequential tokens never
close that gap. **Keep `sequential` for the bulk and end with a ≲2,500-step
strict-bin tail.** "Train contaminated, heal after" does not work as a
whole-run strategy. See [Results](#results) and
[the position decomposition](#follow-up--where-the-strict-advantage-lives-2026-08-13).

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
Reachable only under `resume_compat: warn`; `strict` still refuses. See
[packing](../packing.md) and [checkpointing](../checkpointing.md).

**Correction.** This section originally bounded the discarded tokens at "under
one window plus at most one document tail." That is true only when leaving
`sequential`, whose buffer is a flat token carry. Leaving `bin` discards the
FFD pending queue, which holds up to `max(bins_per_pack, lookahead_docs)`
chunks plus any rendered rows — 275,945 tokens at this recipe's geometry,
roughly 55x the stated bound. Those documents are counted as consumed in
`docs_seen` and `source_tokens_*` but are never trained on. The warning now
reports the measured count per family rather than the sequential bound. The
arms below are unaffected: every switch they perform leaves `sequential`,
which is the direction the original bound described correctly.

### An eval instrument that survives a packing change

Evaluation used to pack its rows with the training knobs, so each arm would
have been scored by a device it had just changed. `data.eval_packing` splits
the two: eval row layout is now its own decision, defaulting to one document
per row, and the training packer no longer reaches it. Selection is untouched —
the same held-out documents either way. See
[training](../training.md#evaluation).

## Results

### Summary — all three packing regimes

Baseline is part 1's step-100000 weights scored directly on the one-doc
instrument: **2.8820**. All arms resume that checkpoint and differ only in
packing.

| arm | packing | evals | mean eval | vs control | t | util | s/step | valid tok/s |
|---|---|---|---|---|---|---|---|---|
| control | `sequential` | 10 | 2.8818 | — | — | 1.000 | 1.80 | 72,767 |
| treatment | `bin`, 1 doc/row | 5 | 2.9099 | **+0.0282** | +16.1 | 0.608 | 1.79 | 44,704 |
| arm 3 | `bin` + strict | 4 | **2.8775** | **−0.0043** | −4.10 | 0.989 | 5.03 | 25,776 |

**Both effects are real and they point in opposite directions.** Removing
contamination the *right* way (strict isolation, full utilization) buys 0.0043
nats. Removing it the *wrong* way (one document per row, 39% padding) costs
0.0282 — roughly 6.5x larger, and in the wrong direction.

**Operational conclusion: keep `sequential` for the bulk, and end with a short
strict-packed tail.** 0.0043 nats is not worth 2.8x the wall clock per token
(25,776 vs 72,767 valid tok/s) across a whole run — the same time on sequential
buys ~2.8x more tokens, which dominates at any live point on an LR schedule. But
the aggregate is a token-weighted mean that hides a 24x-larger effect in the
first tens of positions, and that part is *not* purchasable with more tokens.
A tail of ≲2,500 strict steps captures it for ~8% wall-clock overhead. See
[the position decomposition](#follow-up--where-the-strict-advantage-lives-2026-08-13).

**And the original question — "train contaminated, heal it after" — is
answered no.** See the treatment arm: the heal exists but repays ~0.0019 per
2,500 steps against a ~0.03 deficit.

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
utilization at 1.000, would *not* show the damage. That arm is running — see
[`bin` + strict segments](#bin--strict-segments--running), which was launched
to test exactly this.

Operationally: `segments_per_seq` held at exactly 1.0/1/1, `step_time_s` 1.78
(indistinguishable from sequential), `peak_memory_gb` 29.2 under the 0.90 pool,
`tokens_seen` 13.094B → 14.096B. Measured packing utilization over the whole
arm was **~0.61**, not the 0.534 the 30-step probe suggested — so the
equal-steps token deficit is ~1.65x, not ~1.9x. The 0.701 in the
`-onedoc-62k` header remains the high end; ~0.61 is the settled figure at this
shape.

### Control (sequential) — complete, 125,000 steps

Run dir `runs/chomp-200-2608-p2-seq`, finished 2026-08-11, 16,367,832,519
tokens, `packing_utilization` 1.000, exported to `<run_dir>/export/`.

| step | 102.5k | 105k | 107.5k | 110k | 112.5k | 115k | 117.5k | 120k | 122.5k | 125k |
|---|---|---|---|---|---|---|---|---|---|---|
| eval | 2.8836 | 2.8815 | 2.8805 | 2.8812 | 2.8801 | 2.8792 | 2.8824 | 2.8819 | 2.8822 | 2.8849 |

Mean **2.8818**, spread **0.0058**, **no trend** — first five average 2.8814,
last five 2.8821. 25,000 further steps and 3.27B further tokens at the spent
minimum LR move this model essentially nowhere, which is what makes it a usable
reference. It also pins the eval band at 0.0058 over ten points, confirming the
0.005 used above.

**A retraction.** An intermediate reading of the first three control points
claimed the control drifts down ~0.0016 per 2,500 steps, and concluded that
most of the treatment's late recovery was ordinary training rather than
healing. Ten points show that drift was inside the band. The control is flat,
so the treatment's −0.0019 per 2,500 steps is genuine healing measured against
a flat reference. Do not read a trend off three points spanning 0.003 when the
band is 0.006.

Paired against the treatment at identical steps — same checkpoint, optimizer
state, RNG, corpus position, LR, and eval instrument, differing only in
packing:

| step | control | treatment | gap |
|---|---|---|---|
| 102,500 | 2.8836 | 2.9046 | 0.0210 |
| 105,000 | 2.8815 | 2.9082 | 0.0267 |
| 107,500 | 2.8805 | 2.9142 | 0.0337 |
| 110,000 | 2.8812 | 2.9123 | 0.0311 |
| 112,500 | 2.8801 | 2.9104 | 0.0303 |

The gap peaks at 0.0337 and closes at ~0.0017 per 2,500 steps, so the treatment
needs on the order of 45,000 further steps merely to draw level.

### `bin` + strict segments — complete, 10,000 steps

Run dir `runs/chomp-200-2608-p2-binstrict`, wandb `qxo3pm35`, finished
2026-08-11. 14,390,891,457 tokens, `packing_utilization` 0.995, exported.

| step | 102,500 | 105,000 | 107,500 | 110,000 |
|---|---|---|---|---|
| eval | 2.8758 | 2.8781 | 2.8797 | 2.8763 |
| paired Δ vs control | −0.0077 | −0.0034 | −0.0008 | −0.0049 |

Mean **2.8775** (sd 0.00178, n=4) against the control's 2.8818 (sd 0.00168,
n=10): **−0.0043, t = −4.10**. All four paired differences are negative.

**Strict segment isolation is a real improvement over sequential packing, and
it is small.** This is the first confound-free measurement of the question this
experiment asks, because utilization is 0.989 — no padding, no token deficit,
same 131,072 valid tokens per step as the control.

Two readings-off-noise happened while this arm ran and are recorded as a
caution: the paired gap went −0.0077 → −0.0034 → −0.0008, which was narrated as
a decay toward zero, and then the fourth point came back to −0.0049. Adjacent
eval points differ by ~0.002 against a per-point sd of ~0.0017; **only the arm
mean against the control mean is a statistic.** The same error was made earlier
on the control's first three points. Do not narrate point-to-point movement.

#### Instrument caveat

Evaluation is one document per row, so an arm whose *training* also isolated
documents is better matched to the measuring condition. That is correct for the
operational question — which model is better under the conditions generation
runs in — but it means −0.0043 cannot be read as "contamination damages the
model in general," only as "under single-document evaluation." Note the one-doc
arm had the same structural advantage and still lost by 0.028, which makes its
damage more striking rather than less.

### `bin` + strict segments — configuration and costs

The third packing regime. Note the config naming trap: `-packed-` in
`configs/custom/` means **`packing_mode: sequential`**, not `bin`, and the
`packing_strict_segments: true` in those files is inert because it applies only
to `bin`/`multipack`.

| regime | config | utilization | status |
|---|---|---|---|
| `sequential` | `-packed-fast`, `-packed-62k` | 1.000 | part 1 + control, done |
| `bin` + `max_docs_per_bin: 1` | `-onedoc-*` | 0.608 | treatment, done |
| `bin`, no doc cap, `strict_segments: true` | `-bin-part2` | 0.989 | done |

Full state isolation (see [packing](../packing.md)) removes cross-document
contamination while utilization stays near 1.0 — no padding, no token deficit,
so neither confound the one-doc arm carried.

Two costs measured before committing GPU time, both absent from
[packing](../packing.md), which says only that backward peak "must still be
measured":

- With `use_checkpoint: false` at bs8 x ga8 x seq2048, strict segments ask for a
  single **32.56 GiB** allocation in `jit_train_step` and die with
  `RESOURCE_EXHAUSTED` on a 32 GB card.
- With `use_checkpoint: true`: **13.4 GB** peak (including a generation sample)
  and **6.055 s/step**, i.e. **3.4x** the sequential arms' 1.790 s, not the ~2x
  the attention-FLOP figure implies. Checkpointing's rematerialization compounds
  with strict mode's chunk re-anchoring.

Gradient checkpointing is the confound-free way to fit it: rematerialization is
bit-identical math, whereas narrowing the micro-batch would reassociate the
gradient accumulation sum and make the accumulation partition a second variable
alongside packing.

## Follow-up — where the strict advantage lives (2026-08-13)

The −0.0043 headline is a token-weighted mean over 2048 positions, so it can
hide structure. TimestepNorm's statistics are cumulative along the sequence: a
document starting at row position 900 has its first token normalized by
statistics settled over 900 tokens of unrelated text, while at inference the
same token is normalized by statistics from *one* token. Cumulative statistics
converge fast, which predicts the advantage is concentrated near a cold start.

It is — but that turned out to be only half the story.

### Method

Scripts and raw arrays in `scratch/exp001-position/` (gitignored).

All checkpoints are scored through **one** explicitly built config, **one** eval
token set, and **one** model graph; only the weights vary. That is deliberately
stronger than using each run's own config — and necessary, because these run
dirs snapshot part 1's config rather than their own (see the provenance note
below). `hf_revision` is pinned to the commit both arms resolved at launch
(`8904a95…`) so eval documents cannot drift.

Per-token loss comes from the backend's `compute_loss(reduction="none")`, shape
`(B, T-1)`, where index `j` scores the target at row position `j+1`. The
hand-built valid mask is asserted to reconstruct the stock `make_eval_step`
aggregate exactly, on both loss sum and token count.

Position semantics: the one-doc instrument puts one document *chunk* per row
starting at row position 0, and documents longer than `seq_len` are pre-chunked
to capacity, so **row position is position-since-cold-start**. For the 82% of
eval documents that fit in one row it is also position within the document; the
rest start mid-document but still start cold, which is the condition under test.
`add_bos` is false, so target position 1 is predicting the *second* token from
the first with no context at all.

Eval set: 1000 documents, 1,864,169 tokens, median length 768, 17.9% over
`seq_len`, 1560 rows, 1,863,614 valid targets.

Harness validation: part1@100000 reproduces the recorded **2.8820 exactly**
(delta −0.0000), and the three strict checkpoints reproduce their logged
training-time evals (2.8781 / 2.8797 / 2.8763).

The arms sit at different steps — retention kept only the last 3 checkpoints per
run, so no matched step exists. The sequential reference is therefore linearly
interpolated to step 110,000 from part1@100000 and seq@120000, with seq@125000
as a linearity check. Uncertainty is a 95% paired bootstrap over the 1560 eval
rows (2000 resamples); rows are identical across checkpoints, asserted.

### Result — a front spike plus a uniform floor

Step-matched to 110,000. `contrib` = delta x token share; the column sums to the
aggregate difference.

| target positions | token share | seq@110k | strict | delta | 95% CI | contrib | % of total |
|---|---|---|---|---|---|---|---|
| **1–15** | 1.3% | 3.9046 | 3.7726 | **−0.1319** | [−0.1454, −0.1186] | −0.00165 | **29.4%** |
| 16–31 | 1.3% | 3.2151 | 3.1799 | −0.0353 | [−0.0434, −0.0263] | −0.00047 | 8.4% |
| 32–63 | 2.7% | 3.0481 | 3.0350 | −0.0130 | [−0.0187, −0.0074] | −0.00035 | 6.2% |
| 64–127 | 5.2% | 2.9302 | 2.9247 | −0.0055 | [−0.0095, −0.0015] | −0.00029 | 5.1% |
| 128–255 | 9.8% | 2.8714 | 2.8654 | −0.0060 | [−0.0088, −0.0031] | −0.00059 | 10.5% |
| 256–511 | 17.1% | 2.8051 | 2.8017 | −0.0034 | [−0.0056, −0.0013] | −0.00058 | 10.4% |
| 512–1023 | 26.4% | 2.8780 | 2.8766 | −0.0014 | [−0.0035, +0.0004] | −0.00038 | 6.7% |
| 1024–2047 | 36.2% | 2.8573 | 2.8537 | −0.0036 | [−0.0055, −0.0017] | −0.00130 | 23.2% |
| **total** | 100% | 2.8820 | 2.8763 | **−0.0056** | | −0.00560 | 100% |

The first 15 target positions hold **1.3% of the tokens but carry 29% of the
total advantage**, at 24x the aggregate effect size. Through position 8: 0.7% of
tokens, 22.6% of the advantage.

**But the effect is not purely front-loaded.** A floor of roughly −0.0035
persists at every depth; positions 1024–2047 contribute another 23% with a CI
that excludes zero. The structure is a steep front spike *plus* a small uniform
advantage — not a spike decaying to nothing.

### More tokens do not fix the front deficit

Sequential training does not close the front gap with additional budget. Over
25,000 steps (~3.3B tokens) the sequential arm's 1–15 bucket went 3.9098 →
3.8993 → 3.9086: flat, inside its own 0.0105 band.

This is the load-bearing finding. The "sequential wins because the same wall
clock buys 2.8x more tokens" argument does **not** apply to this deficit,
because sequential packing structurally almost never presents the
cold-start-at-document-start condition: row starts are cold but land
mid-document, and document starts land mid-row with warm state. More tokens of a
distribution that omits a condition does not teach that condition. It is a
coverage problem, not a budget problem — which is exactly why the aggregate,
dominated by the 36% of tokens past position 1024, hid it.

### The fix is fast — a short strict tail is enough

Bucket 1–15 across the strict arm, which branched from part1@100000:

| step | strict steps elapsed | bucket 1–15 |
|---|---|---|
| 100,000 | 0 (branch point) | 3.9098 |
| 105,000 | 5,000 | **3.7727** |
| 107,500 | 7,500 | 3.7744 |
| 110,000 | 10,000 | 3.7726 |

The entire 0.133-nat gain is present at the first available checkpoint and then
flat (spread 0.0018, noise). It does not accumulate. The aggregate advantage was
likewise at full magnitude by the first eval at 102,500, so the true healing
time is **≤2,500 steps** — under 3% of a 100,000-step run.

### What this changes

- **Keep `sequential` for the bulk.** Unchanged: the aggregate is still ~0.005
  nats against 2.8x wall clock per token.
- **Add a short strict-packed tail.** A few thousand steps captures essentially
  the whole benefit, including the part extra tokens cannot buy, at ~3% of the
  run at 2.8x cost — roughly 8% wall-clock overhead.
- **Do not quote −0.0043 as the cost of contamination for short-context work.**
  For a prompt in the first tens of tokens the gap is ~0.13 nats, 24x larger.
  Generation from a short prompt, chat turn openings, and SFT example starts all
  live in the damaged region; long few-shot contexts largely do not.
- **Multiple-choice loglikelihood evals cancel this only partially.** Candidates
  share a prefix and therefore share the *same* state, but for a short stem that
  shared state is the miscalibrated one. This was not measured — it is the
  obvious next experiment, and it needs an accuracy metric, not a loss.

### Provenance note

Every phase-2 run dir contains part 1's `config_original.yaml` /
`config_resolved.json` (identical md5 across all four runs, missing the
`eval_packing` key entirely). This is **not** a chomp defect: the arms were
branched by copying part 1's run dir, and `utils/io.py` deliberately does not
rewrite the snapshot on resume. The runs themselves were correct — `launch.log`
records the real effective config (`packing_mode` current=`'bin'`,
`strict_segments`=True, `use_checkpoint`=True, `steps`=110000). The residual
gap: a resumed run leaves no on-disk record of what it actually ran except that
log. Read `launch.log`, not the config snapshot, when auditing these dirs.

<!--
When an arm finishes, record here: final eval_loss and the step it came from,
the eval trajectory over the last ~5 evals (not just the last point — see the
0.005-nat band above), final tokens_seen, measured packing_utilization over the
whole arm, and the wandb run name. Then update Status and the index row in
README.md.
-->
