# Weight Export

Related: [Checkpointing](checkpointing.md), [Training](training.md), [Config Reference](config-reference.yaml).

A training checkpoint is not a model. `checkpoints/<step>/train_state` is an
Orbax pytree keyed by chomp's config: it carries optimizer moments and RNG
alongside the parameters, sits next to a serialized data-iterator position, and
only chomp can interpret it. For the 200M recipe it is about 1.5 GB, of which
the weights are roughly half.

`chomp export` writes the model out on its own, in the safetensors format
megalodon-jax already defines, together with the run's tokenizer, a Hugging
Face shaped `config.json`, and a provenance manifest.

```bash
chomp export runs/my_run --out exports/my_run-step100000
```

`CHECKPOINT` accepts the same three forms `chomp generate` does — a run
directory (uses its latest retained checkpoint), a checkpoint root, or an exact
step directory. Export never modifies the checkpoint.

The destination must be empty, absent, or a previous export. `--overwrite`
replaces a previous export by deleting the files its manifest claims — never
the directory itself — so a tokenizer file from the old model cannot linger
beside new weights and be picked up by `AutoTokenizer`. A non-empty directory
that is not an export is refused even with `--overwrite`: chomp cannot know
what else lives there.

Export runs on the host, not the accelerator. It copies bytes and computes
nothing, and a second copy of the parameters on the device is exactly what
would break the end-of-run export below: that one runs inside a process still
holding a memory pool sized for its own train step.

## At the end of a run

`export.on_finish` (default `true`) exports the run's final checkpoint into
`runs/my_run/export/` as the run exits, so what a run is for is on disk in a
loadable form without a second command:

```yaml
export:
  on_finish: true
  dir_name: export
  verify: true
```

It is written after the final checkpoint is durable and the checkpoint manager
is closed, and it reads that checkpoint back off disk rather than serializing
the live train state — so it is exactly what a later `chomp export` would
produce. A resumed run replaces the previous export with one from its own final
step.

Only a run that finished on its own terms is exported. A preempted or crashed
run's newest checkpoint is a resume point, not a result, and exporting one
would leave a model in the run directory that nobody decided was finished.
Runs on the `dummy` backend and runs with `checkpoint.enabled: false` are
skipped with a log line.

**An export failure does not fail the run.** By that point the checkpoint is
durable and `chomp export` reproduces the export in seconds; failing a
50-hour run over a convenience copy would be the wrong trade. The failure is
logged at ERROR with its traceback and recorded in the W&B summary as
`export_written: false`, never swallowed.

## What you get

```
exports/my_run-step100000/
├── model.safetensors        # weights + the full MegalodonConfig in the header
├── config.json              # the same architecture, Hugging Face field names
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

## config.json

The safetensors header is authoritative inside chomp, and nothing in chomp
reads `config.json`. It exists because the header is not where the rest of the
world looks, and because a port to another framework should not have to parse
one — or install megalodon-jax — to learn the architecture.

It is built from the header of the file it sits beside, never from the chomp
config that produced it, so it cannot describe a model other than the one in
`model.safetensors`. Every field of the upstream `MegalodonConfig` is present
under its upstream name, so the file reconstructs that config exactly. On top
of that sit the Hugging Face spellings:

```json
{
  "model_type": "megalodon",
  "architectures": ["MegalodonForCausalLM"],
  "torch_dtype": "float32",
  "tie_word_embeddings": true,
  "hidden_size": 1024,
  "num_hidden_layers": 12,
  "num_attention_heads": 1,
  "intermediate_size": 1728,
  "model_dim": 1024, "num_layers": 12, "num_heads": 1, "ffn_hidden_dim": 1728,
  "z_dim": 256, "value_dim": 2048, "cema_ndim": 16, "chunk_size": 512, "...": "",
  "megalodon_jax": {
    "format": "megalodon-jax", "format_version": "3",
    "rope_layout": "adjacent_pair",
    "normalization_storage": "plus_one",
    "bias_schema": "upstream",
    "initializer_schema": "split-boundary-internal-v1",
    "tying": "tied",
    "dtype_policy": "fp32-params-fp32-or-bf16-compute",
    "config_fingerprint": "...", "parameter_manifest_sha256": "...",
    "weights_file": "model.safetensors", "version": "0.2.2"
  }
}
```

The HF keys are aliases, not translations: only pairs whose meaning is
identical are listed, and export refuses to write a file where an alias
collided with a native field. `vocab_size`, `attention_dropout`, and the
special-token IDs already spell the same in both vocabularies. Fields with no
HF counterpart — `z_dim`, `value_dim`, `cema_ndim`, `chunk_size`,
`norm_num_groups`, `rope_base`, `scale_emb`, `rescale_nffn` — keep their
upstream names, which is what `PretrainedConfig.attribute_map` is for on the
PyTorch side.

`max_position_embeddings` is deliberately absent. Megalodon has no positional
table and no architectural context bound, so any value would be a training
detail dressed up as a constraint. `torch_dtype` is the storage dtype of
ordinary parameters; under upstream's bf16 policy normalization, CEMA, and
affine parameters stay fp32 regardless, so it describes the bulk of the file
rather than every tensor in it.

The `megalodon_jax` block is the weight-layout contract, copied from the
header. It is what a port has to agree with to read the tensors correctly: how
RoPE pairs are laid out, whether normalization scales are stored as `1 + w`,
which projections carry biases, and how the initializer was split. Check
`format_version` before trusting a layout assumption.

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

**Exports are not byte-reproducible.** Two exports of the same checkpoint
contain identical tensors but can differ in the safetensors header: upstream
serializes its metadata map from an unordered container, so the JSON keys come
out in a different order and the files are the same length but not the same
bytes. Compare exports by loading and comparing tensors, never by hashing the
file.

**There is no bf16 export.** Upstream's `BF16_DTYPE_POLICY` keeps some
parameters at fp32 while casting the rest, and which parameters those are is
upstream's decision, encoded in its model constructor. Guessing at it here
would produce a file that loads and is subtly wrong. Cast downstream instead.

**There is no Hugging Face Transformers export.** A `transformers`-loadable
checkpoint needs a PyTorch `PreTrainedModel` for this architecture, which does
not exist. `megalodon_jax.convert` has `load_upstream_checkpoint` for the
original PyTorch Megalodon weights, but that is an import path, not an export
one. The tokenizer files in an export directory are genuine HF files and
`config.json` is a genuine HF-shaped config; `model.safetensors` is a
megalodon-jax file that happens to use the same container, with megalodon-jax
tensor names and layout. An export directory is what a PyTorch port needs to
read, not something `AutoModel` can load today.

**Only `model.backend: megalodon` can be exported.** The `dummy` backend is a
smoke-test model with no serialization format; export refuses it.

## chomp_export.json

```json
{
  "schema_version": 1,
  "chomp_version": "...",
  "megalodon_jax": {"distribution": "megalodon-jax", "version": "0.2.2"},
  "weights_file": "model.safetensors",
  "config_file": "config.json",
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
