# Weight Export

Related: [Checkpointing](checkpointing.md), [Training](training.md), [Config Reference](config-reference.yaml).

A training checkpoint is not a model. `checkpoints/<step>/train_state` is an
Orbax pytree keyed by chomp's config: it carries optimizer moments and RNG
alongside the parameters, sits next to a serialized data-iterator position, and
only chomp can interpret it. For the 200M recipe it is about 1.5 GB, of which
the weights are roughly half.

`chomp export` writes the model out on its own, in the safetensors format
megalodon-jax already defines, together with the run's tokenizer and a
provenance manifest.

```bash
chomp export runs/my_run --out exports/my_run-step100000
```

`CHECKPOINT` accepts the same three forms `chomp generate` does — a run
directory (uses its latest retained checkpoint), a checkpoint root, or an exact
step directory. Export never writes into the run directory and never modifies
the checkpoint.

The destination must be empty, absent, or a previous export. `--overwrite`
replaces a previous export by deleting the files its manifest claims — never
the directory itself — so a tokenizer file from the old model cannot linger
beside new weights and be picked up by `AutoTokenizer`. A non-empty directory
that is not an export is refused even with `--overwrite`: chomp cannot know
what else lives there.

## What you get

```
exports/my_run-step100000/
├── model.safetensors        # weights + the full MegalodonConfig in the header
├── chomp_export.json        # provenance and the resolved chomp config
├── tokenizer.json           # the run-pinned tokenizer, copied byte-for-byte
├── tokenizer_config.json
├── special_tokens_map.json
└── identity.json            # chomp's tokenizer identity manifest
```

Tokenizer files land at the export root rather than under `tokenizer/`, so
`AutoTokenizer.from_pretrained(export_dir)` resolves without knowing anything
about chomp's run layout. They are copied rather than re-serialized: the
identity manifest hashes their exact bytes, and a round trip through
`save_pretrained` could invalidate the identity export just proved.

## Loading it

The weights file is self-describing. `megalodon_jax.save_checkpoint` embeds the
complete `MegalodonConfig`, a config fingerprint, and a parameter manifest in
the safetensors header, so **chomp is not needed to load it**:

```python
import jax
from megalodon_jax import load_checkpoint

model = load_checkpoint("exports/my_run-step100000/model.safetensors", key=jax.random.key(0))
logits, _ = model(input_ids)
```

`chomp generate` also accepts an export directory wherever it accepts a run
directory, which is the cheapest way to confirm an export is good:

```bash
chomp generate exports/my_run-step100000 --prompt "Hello world" --max-tokens 64 --temperature 0
```

Greedy generation from the export and from the source checkpoint produces
identical text. Inside chomp, `chomp.export.load_export` returns the same
`(params, static)` pair the Orbax path produces.

## Guarantees, and what they are not

**Export is lossless.** Parameters are written at the dtype they were trained
at. `model.param_dtype` is pinned to `float32`, so an export is float32 and
roughly four bytes per parameter — about 760 MB for the 200M recipe.

**Export is verified by default.** safetensors carries no payload checksum, and
upstream's manifest hash covers tensor names, shapes, and dtypes rather than
bytes — nothing in the pipeline would notice a corrupted tensor until
generation produced nonsense. So `chomp export` re-reads the file it just wrote
and compares every parameter. `--no-verify` skips this; the output says
`(NOT verified)` when you do.

**The tokenizer is proven, not assumed.** Token IDs index restored embedding
rows directly, so export applies the same identity check `chomp generate` does
and refuses rather than warns when the run's tokenizer cannot be shown to match
the checkpoint's recorded `tokenizer_identity`.

**There is no bf16 export.** Upstream's `BF16_DTYPE_POLICY` keeps some
parameters at fp32 while casting the rest, and which parameters those are is
upstream's decision, encoded in its model constructor. Guessing at it here
would produce a file that loads and is subtly wrong. Cast downstream instead.

**There is no Hugging Face Transformers export.** A `transformers`-loadable
checkpoint needs a PyTorch `PreTrainedModel` for this architecture, which does
not exist. `megalodon_jax.convert` has `load_upstream_checkpoint` for the
original PyTorch Megalodon weights, but that is an import path, not an export
one. The tokenizer files in an export directory are genuine HF files;
`model.safetensors` is a megalodon-jax file that happens to use the same
container.

**Only `model.backend: megalodon` can be exported.** The `dummy` backend is a
smoke-test model with no serialization format; export refuses it.

## chomp_export.json

```json
{
  "schema_version": 1,
  "chomp_version": "...",
  "megalodon_jax": {"distribution": "megalodon-jax", "version": "0.2.2"},
  "weights_file": "model.safetensors",
  "param_count": 188777472,
  "source": {"run_dir": "...", "step_dir": "...", "step": 100000},
  "training": {"tokens_seen": 13094263113, "eval_status": {...}},
  "tokenizer_identity": {...},
  "tokenizer_files": ["identity.json", "special_tokens_map.json", "..."],
  "config": {"model": {...}, "data": {...}, "train": {...}, "...": {}}
}
```

`config` is the resolved chomp config **after** tokenizer preparation, so its
`model.vocab_size` and special-token IDs are the ones the stored arrays were
actually shaped by, not the ones the YAML asked for. It records the data and
optimizer settings that produced these weights, which the safetensors header
does not carry.

`schema_version` is checked exactly on load. Chomp does not translate across
schema versions anywhere else, and an export that loaded under the wrong schema
could pair the wrong vocabulary with these weights; re-export instead.

Loading cross-checks the two config sources against each other. If
`chomp_export.json` describes a different architecture than the safetensors
header contains — a hand-edited manifest, a directory assembled from two
exports — `load_export` raises rather than generating from a mismatched pair.

## Export is not a checkpoint

An export cannot be resumed from. It has no optimizer state, no RNG, no step,
and no data position. To continue training use `--resume` against the run
directory, or `--init-from` to warm-start a new run's parameters; see
[Checkpointing](checkpointing.md#warm-start---init-from). Export is a one-way
door for inference and distribution.
