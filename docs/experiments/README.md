# Experiments

One file per experiment, numbered in the order they were started:
`NNN-short-slug.md`. Numbers are never reused and never renumbered, so a
number in a commit message or a wandb tag keeps pointing at the same thing.

An experiment gets a file here when it is a **question about the model or the
data**, answered by running training and comparing outcomes. Engineering work
that merely makes a run possible belongs in the regular docs
([training](../training.md), [packing](../packing.md),
[checkpointing](../checkpointing.md)) — link to it from here instead of
restating it.

## What a file must contain

- **Question** — one sentence, phrased so that a number can answer it.
- **Setup** — configs, checkpoints, and launch commands, verbatim enough to
  re-run. Name the one variable that differs between arms.
- **Confounds** — what else moved, and whether it was controlled.
- **Status** — `planned`, `running`, or `done`, with dates.
- **Results** — measured numbers with the step they came from. Record the
  disappointing ones; an experiment that answered "no difference" is a result,
  and rerunning it later because nobody wrote it down is the waste this
  directory exists to prevent.

Keep measured values distinguishable from expected ones. A number that came
out of `metrics.jsonl` should say so; a number that came out of arithmetic
should say that too.

## Index

| # | Experiment | Status |
|---|---|---|
| [001](001-sequential-vs-onedoc-packing.md) | Does sequential packing hurt? CEMA/TimestepNorm state contamination across document boundaries | done — barely: strict isolation buys 0.0043 nats for 2.8x the cost, one-doc-per-row costs 0.0282. Keep sequential |
