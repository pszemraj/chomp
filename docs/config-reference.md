# Configuration Reference

chomp uses frozen dataclasses for configuration. YAML files map directly to the
config tree, and dot-path overrides are supported from the CLI.

> [!IMPORTANT]
> Never use `fp16`. Megalodon's CEMA and normalization layers are unstable with fp16.
> Use `bfloat16` compute with `float32` params/accumulation (the Megalodon defaults).
<a id="quick-reference"></a>
## Quick Reference
The 10 most commonly adjusted fields for typical experiments:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| [`train.steps`](#train.steps) | `int` | `100` | Total training steps |
| [`train.batch_size`](#train.batch_size) | `int` | `2` | Micro-batch size per device |
| [`train.seq_len`](#train.seq_len) | `int` | `128` | Sequence length (tokens) |
| [`train.grad_accum`](#train.grad_accum) | `int` | `1` | Gradient accumulation steps |
| [`optim.lr`](#optim.lr) | `float` | `3e-4` | Peak learning rate |
| [`optim.warmup_steps`](#optim.warmup_steps) | `int` | `10` | Linear warmup steps |
| [`model.num_layers`](#model.num_layers) | `int` | `2` | Number of transformer layers |
| [`model.model_dim`](#model.model_dim) | `int` | `128` | Hidden dimension |
| [`data.hf_dataset`](#data.hf_dataset) | `str` | `"Zyphra/Zyda-2"` | HuggingFace dataset path |
| [`checkpoint.save_every`](#checkpoint.save_every) | `int` | `5000` | Steps between checkpoints |

<a id="cli-override"></a>
## CLI Override Syntax
Override any field using dot-path notation:

```bash
chomp train configs/my_config.yaml \
  --override train.steps=2000 \
  --override train.batch_size=4 \
  --override optim.lr=1e-4
```

Nested fields use multiple dots:

```bash
--override data.tokenizer.kind=hf
--override logging.wandb.enabled=true
```

Overrides are parsed as YAML scalars. This means optional fields with `null`
defaults can still be set to numbers or booleans (e.g.,
`--override optim.muon.consistent_rms=0.2`). Use quotes to force a string.

> [!NOTE]
> Unknown keys or invalid values fail fast during validation with actionable error messages.
<a id="variables"></a>
## Variables and interpolation
Define reusable values under a top-level `variables` section and reference them
elsewhere in the config.

Supported forms:

- Exact value: `chunk_size: $variables.chunk_size` (type preserved)
- Inline string: `tags: ["seq{$variables.seq_len}", "chunk{$variables.chunk_size}"]`
- Brace form: `tags: ["seq_${variables.seq_len}"]`

Example:

```yaml
variables:
  chunk_size: 1024
  seq_len: 2048
model:
  chunk_size: $variables.chunk_size
train:
  seq_len: $variables.seq_len
logging:
  wandb:
    tags: ["seq{$variables.seq_len}", "chunk{$variables.chunk_size}"]
```

Variables can be nested (dot paths are supported). Resolution happens before
CLI overrides are applied, and missing/circular references raise a validation
error.

Store personal experiment configs under `configs/custom/`. The directory is tracked
via `.gitkeep`, but `configs/custom/*.yaml` and `configs/custom/*.yml` are ignored.
---

<a id="model"></a>
## model (ModelConfig)
Model architecture and precision policy. Contains 35 fields.

<a id="model-backend"></a>
### Backend Selection
<a id="model.backend"></a>
#### `model.backend`
```yaml
backend: "megalodon" | "dummy" = "megalodon"
```

Model backend implementation.

| Property | Value |
|----------|-------|
| Required | No |
| Valid values | `"megalodon"`, `"dummy"` |

##### megalodon

Full Megalodon architecture from `megalodon_jax`. Use for real training.

##### dummy

Minimal MLP language model for smoke tests. Fast compilation, no external dependencies.
---

<a id="model-shared"></a>
### Architecture (Shared)
<a id="model.vocab_size"></a>
#### `model.vocab_size`
```yaml
vocab_size: int = 256
```

Vocabulary size for the embedding layer.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive; for byte tokenizer, must be ≥ `byte_offset + 256` |

When using an HF tokenizer, vocab size is automatically aligned to
`data.tokenizer.vocab_size_multiple` (default: 128) and at least the tokenizer vocab size.
<a id="model.dropout"></a>
#### `model.dropout`
```yaml
dropout: float = 0.0
```

Dropout rate used by the model backend (dummy and Megalodon).

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `dummy` and `megalodon` |
| Range | `[0.0, 1.0]` |

Setting dropout > 0 disables deterministic training unless `train.deterministic: true` is explicit.
---

<a id="model-dummy"></a>
### Architecture (DummyLM)
These fields only apply when `model.backend: dummy`.

<a id="model.d_model"></a>
#### `model.d_model`
```yaml
d_model: int = 128
```

Hidden dimension for DummyLM backend.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `dummy` only |
| Constraints | Must be positive |

---

<a id="model-megalodon"></a>
### Architecture (Megalodon)
These fields apply when `model.backend: megalodon`.

<a id="model.model_dim"></a>
#### `model.model_dim`
```yaml
model_dim: int = 128
```

Hidden dimension (d_model equivalent).

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |
| Constraints | Must be positive; must be divisible by `num_heads` |
| Recommended | 768 (100M), 1024 (300M), 2048 (1B+) |

<a id="model.num_layers"></a>
#### `model.num_layers`
```yaml
num_layers: int = 2
```

Number of transformer layers.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |
| Constraints | Must be positive |
| Recommended | 12 (100M), 24 (300M), 32+ (1B+) |

<a id="model.num_heads"></a>
#### `model.num_heads`
```yaml
num_heads: int = 1
```

Number of attention heads.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |
| Constraints | Must be positive; must divide `model_dim` evenly |

If `model_dim % num_heads != 0`, validation fails with:
`model.model_dim (X) must be divisible by model.num_heads (Y)`
<a id="model.z_dim"></a>
#### `model.z_dim`
```yaml
z_dim: int = 64
```

Dimension of the complex exponential moving average (CEMA) state.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |
| Recommended | `model_dim // 2` |

<a id="model.value_dim"></a>
#### `model.value_dim`
```yaml
value_dim: int = 128
```

Dimension of value projections in attention.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |
| Recommended | Equal to `model_dim` |

<a id="model.ffn_hidden_dim"></a>
#### `model.ffn_hidden_dim`
```yaml
ffn_hidden_dim: int = 256
```

Hidden dimension of the feed-forward network.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |
| Recommended | `4 * model_dim` (standard), `8/3 * model_dim` (SwiGLU) |

<a id="model.cema_ndim"></a>
#### `model.cema_ndim`
```yaml
cema_ndim: int = 16
```

Number of CEMA dimensions per head.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

<a id="model.chunk_size"></a>
#### `model.chunk_size`
```yaml
chunk_size: int = 128
```

Internal chunk size for attention computation.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |
| Constraints | Must be positive; must be ≤ `train.seq_len`; must divide `train.seq_len` evenly |

If `train.seq_len % model.chunk_size != 0`, validation fails with:
`train.seq_len (X) must be divisible by model.chunk_size (Y)`
**Related:** [`train.seq_len`](#train.seq_len)

<a id="model.max_cache_len"></a>
#### `model.max_cache_len`
```yaml
max_cache_len: int | null = null
```

Maximum cache length for inference. `null` means unlimited.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

Training never uses cache. This field only affects inference mode.
<a id="model.cache_unbounded"></a>
#### `model.cache_unbounded`
```yaml
cache_unbounded: bool = false
```

Allow unbounded cache growth during inference.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

<a id="model.norm_num_groups"></a>
#### `model.norm_num_groups`
```yaml
norm_num_groups: int = 32
```

Number of groups for GroupNorm layers.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

<a id="model.norm_eps"></a>
#### `model.norm_eps`
```yaml
norm_eps: float = 1e-5
```

Epsilon for normalization layers.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

<a id="model.rope_base"></a>
#### `model.rope_base`
```yaml
rope_base: float | null = null
```

Base for rotary position embeddings. `null` uses the model's default.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

<a id="model.swiglu"></a>
#### `model.swiglu`
```yaml
swiglu: bool = false
```

Use SwiGLU activation in the FFN instead of standard GELU.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

When using SwiGLU, adjust `ffn_hidden_dim` to `8/3 * model_dim` for parameter parity.
<a id="model.rescale_nffn"></a>
#### `model.rescale_nffn`
```yaml
rescale_nffn: bool = false
```

Apply FFN output rescaling for training stability.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

<a id="model.scale_emb"></a>
#### `model.scale_emb`
```yaml
scale_emb: bool = false
```

Scale embeddings by `sqrt(model_dim)`.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

<a id="model.norm_affine"></a>
#### `model.norm_affine`
```yaml
norm_affine: bool = true
```

Use learnable affine parameters in normalization layers.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

<a id="model.attention_dropout"></a>
#### `model.attention_dropout`
```yaml
attention_dropout: float = 0.0
```

Dropout rate applied to attention weights.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |
| Range | `[0.0, 1.0]` |

<a id="model.hidden_dropout"></a>
#### `model.hidden_dropout`
```yaml
hidden_dropout: float = 0.0
```

Dropout rate applied to hidden states.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |
| Range | `[0.0, 1.0]` |

<a id="model.max_positions"></a>
#### `model.max_positions`
```yaml
max_positions: int = 1_000_000
```

Maximum sequence positions supported.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

<a id="model.init_mode"></a>
#### `model.init_mode`
```yaml
init_mode: "gaussian" | "xavier" | "he" | "bert" | "none" = "he"
```

Weight initialization scheme.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |
| Recommended | `"he"` (default) |

<a id="model.use_checkpoint"></a>
#### `model.use_checkpoint`
```yaml
use_checkpoint: bool = false
```

Enable gradient checkpointing (activation recomputation) to reduce memory.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |
| Recommended | `true` for large models or long sequences |

In `megalodon-jax`, checkpointing is disabled when `train.deterministic=true`.
Set `train.deterministic=false` (with dropout at 0.0 if desired) to enable it.
<a id="model.output_size"></a>
#### `model.output_size`
```yaml
output_size: int = -1
```

Override output projection size. `-1` uses `vocab_size`.

| Property | Value |
|----------|-------|
| Required | No |
| Backend | `megalodon` only |

---

<a id="model-special-tokens"></a>
### Special Tokens
<a id="model.pad_token_id"></a>
#### `model.pad_token_id`
```yaml
pad_token_id: int = 0
```

Token ID used for padding.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Should differ from `eos_token_id` (recommended) |

## pad_token_id should differ from eos_token_id

Megalodon zero-masks pad embeddings and the training loop masks pad positions in loss.
If `pad_token_id == eos_token_id`, EOS tokens may be treated as padding depending on the
model implementation, which can degrade EOS supervision.

If your tokenizer sets `pad_token_id == eos_token_id`, chomp will emit a warning after tokenizer
resolution. Prefer a tokenizer with distinct pad/eos, or set
`data.tokenizer.auto_set_special_tokens: false` and override `model.pad_token_id` explicitly
(only safe if you never emit pad tokens).
**Related:** [`model.eos_token_id`](#model.eos_token_id)

<a id="model.bos_token_id"></a>
#### `model.bos_token_id`
```yaml
bos_token_id: int = 1
```

Beginning-of-sequence token ID.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be in `[0, vocab_size)` if `add_bos: true` |

**Related:** [`data.tokenizer.add_bos`](#data.tokenizer.add_bos)

<a id="model.eos_token_id"></a>
#### `model.eos_token_id`
```yaml
eos_token_id: int = 2
```

End-of-sequence token ID.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be in `[0, vocab_size)` |

**Related:** [`data.tokenizer.add_eos`](#data.tokenizer.add_eos), [`data.train_on_eos`](#data.train_on_eos)

---

<a id="model-dtype"></a>
### Dtype Policy
<a id="model.param_dtype"></a>
#### `model.param_dtype`
```yaml
param_dtype: "float32" | "bfloat16" = "float32"
```

Dtype for model parameters (weights).

| Property | Value |
|----------|-------|
| Required | No |
| Recommended | `"float32"` |

<a id="model.compute_dtype"></a>
#### `model.compute_dtype`
```yaml
compute_dtype: "float32" | "bfloat16" = "bfloat16"
```

Dtype for forward/backward computation.

| Property | Value |
|----------|-------|
| Required | No |
| Recommended | `"bfloat16"` |

**Never use `fp16`** — Megalodon's CEMA and normalization layers are numerically unstable with fp16.
<a id="model.accum_dtype"></a>
#### `model.accum_dtype`
```yaml
accum_dtype: "float32" | "bfloat16" = "float32"
```

Dtype for gradient accumulation.

| Property | Value |
|----------|-------|
| Required | No |
| Recommended | `"float32"` |

<a id="model.softmax_dtype"></a>
#### `model.softmax_dtype`
```yaml
softmax_dtype: "float32" | "bfloat16" = "float32"
```

Dtype for softmax computation in attention.

| Property | Value |
|----------|-------|
| Required | No |
| Recommended | `"float32"` for numerical stability |

---

<a id="model-advanced"></a>
### Advanced
<a id="model.gemm_backend"></a>
#### `model.gemm_backend`
```yaml
gemm_backend: "default" = "default"
```

GEMM backend selection. Currently only `"default"` is supported.

| Property | Value |
|----------|-------|
| Required | No |
| Valid values | `"default"` |

<a id="data"></a>
## data (DataConfig)
Dataset, tokenizer, and packing configuration. Contains 23 fields.

<a id="data-backend"></a>
### Backend Selection
<a id="data.backend"></a>
#### `data.backend`
```yaml
backend: "hf" | "local_text" = "hf"
```

Data source backend.

| Property | Value |
|----------|-------|
| Required | No |
| Valid values | `"hf"`, `"local_text"` |

##### hf (HuggingFace Streaming)

Stream data from HuggingFace datasets. Use for real training.

##### local_text

Repeat a fixed text string. Use for offline smoke tests.
---

<a id="data-hf"></a>
### HF Streaming Fields
These fields apply when `data.backend: hf`.

<a id="data.hf_dataset"></a>
#### `data.hf_dataset`
```yaml
hf_dataset: str = "Zyphra/Zyda-2"
```

HuggingFace dataset path.

| Property | Value |
|----------|-------|
| Required | Yes (when `backend: hf`) |
| Constraints | Must be non-empty |

<a id="data.hf_name"></a>
#### `data.hf_name`
```yaml
hf_name: str = "sample-100BT"
```

Dataset configuration/subset name.

| Property | Value |
|----------|-------|
| Required | Yes (when `backend: hf`) |
| Constraints | Must be non-empty |

<a id="data.hf_split"></a>
#### `data.hf_split`
```yaml
hf_split: str = "train"
```

Dataset split for training.

| Property | Value |
|----------|-------|
| Required | Yes (when `backend: hf`) |
| Constraints | Must be non-empty |

<a id="data.hf_eval_split"></a>
#### `data.hf_eval_split`
```yaml
hf_eval_split: str | null = null
```

Preferred split for evaluation. Falls back to train split if missing.
Set to `null` to skip eval-split lookup and always derive eval texts from train.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be `null` or non-empty |

<a id="data.text_key"></a>
#### `data.text_key`
```yaml
text_key: str = "text"
```

Column name containing text data in the dataset.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be non-empty |

<a id="data.max_eval_samples"></a>
#### `data.max_eval_samples`
```yaml
max_eval_samples: int = 1000
```

Maximum examples to use for evaluation.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be ≥ 0 |

Evaluation texts are selected once at run start and reused for the entire run. If the
evaluation split is missing, chomp uses the first `max_eval_samples` examples from the
shuffled training stream. For this train-split fallback path, if `data.seed: 0`
(default) and `train.seed` is non-zero, the shuffle seed defaults to `train.seed`.
---

<a id="data-shuffle"></a>
### Shuffling & Repeat
<a id="data.shuffle"></a>
#### `data.shuffle`
```yaml
shuffle: bool = true
```

Enable shuffling of the data stream.

| Property | Value |
|----------|-------|
| Required | No |

<a id="data.shuffle_buffer_size"></a>
#### `data.shuffle_buffer_size`
```yaml
shuffle_buffer_size: int = 10_000
```

Buffer size for shuffle operation.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive when `shuffle: true` |

<a id="data.seed"></a>
#### `data.seed`
```yaml
seed: int = 0
```

Random seed for data shuffling.

| Property | Value |
|----------|-------|
| Required | No |

<a id="data.repeat"></a>
#### `data.repeat`
```yaml
repeat: bool = true
```

Repeat the dataset indefinitely.

| Property | Value |
|----------|-------|
| Required | No |
| Recommended | `true` for pretraining |

---

<a id="data-network"></a>
### Network Resilience
<a id="data.max_retries"></a>
#### `data.max_retries`
```yaml
max_retries: int = 3
```

Maximum retries for failed network requests.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be ≥ 0 |

<a id="data.retry_delay_sec"></a>
#### `data.retry_delay_sec`
```yaml
retry_delay_sec: float = 1.0
```

Delay between retry attempts in seconds.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be ≥ 0 |

<a id="data.state_update_interval"></a>
#### `data.state_update_interval`
```yaml
state_update_interval: int = 2_000
```

Cache HF state every N examples for retry recovery.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be > 0 |

Smaller values provide better recovery from network failures but add overhead.
---

<a id="data-local"></a>
### Local Text (Debug)
<a id="data.local_text"></a>
#### `data.local_text`
```yaml
local_text: str = "Hello from chomp.\n"
```

Fixed text content for `local_text` backend.

| Property | Value |
|----------|-------|
| Required | Yes (when `backend: local_text`) |
| Constraints | Must be non-empty |

---

<a id="data-packing"></a>
### Packing Configuration
<a id="data.packing_mode"></a>
#### `data.packing_mode`
```yaml
packing_mode: "sequential" | "bin" = "sequential"
```

Token packing strategy.

| Property | Value |
|----------|-------|
| Required | No |
| Valid values | `"sequential"`, `"bin"` |

##### sequential

Stream documents and concatenate tokens sequentially. Simple and deterministic.

##### bin

First-fit-decreasing bin packing. Better packing efficiency but requires buffering documents.
<a id="data.packing_buffer_docs"></a>
#### `data.packing_buffer_docs`
```yaml
packing_buffer_docs: int = 128
```

Number of documents to buffer for bin packing.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive; when `packing_mode: bin`, must be ≥ `batch_size × grad_accum` |

If `packing_buffer_docs < batch_size × grad_accum` with bin packing:
`data.packing_buffer_docs must be >= train.batch_size * train.grad_accum (N), got M`
**Related:** [`train.batch_size`](#train.batch_size), [`train.grad_accum`](#train.grad_accum)

<a id="data.packing_max_docs_per_bin"></a>
#### `data.packing_max_docs_per_bin`
```yaml
packing_max_docs_per_bin: int | null = null
```

Maximum documents per packed sequence. `null` means unlimited.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive when set |

---

<a id="data-loss"></a>
### Packed-Doc Loss Behavior
<a id="data.mask_boundary_loss"></a>
#### `data.mask_boundary_loss`
```yaml
mask_boundary_loss: bool = true
```

Mask loss at document boundary positions (separator tokens).

| Property | Value |
|----------|-------|
| Required | No |

<a id="data.train_on_eos"></a>
#### `data.train_on_eos`
```yaml
train_on_eos: bool = true
```

Include EOS token in loss computation.

| Property | Value |
|----------|-------|
| Required | No |

**Related:** [`model.eos_token_id`](#model.eos_token_id)

---

<a id="data-pipeline"></a>
### Pipeline Settings
<a id="data.grain_prefetch"></a>
#### `data.grain_prefetch`
```yaml
grain_prefetch: int = 0
```

Grain pipeline prefetch buffer size. `0` disables prefetching.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be ≥ 0 |
| Recommended | `2` for production training |

<a id="data.device_put"></a>
#### `data.device_put`
```yaml
device_put: bool = false
```

Transfer batches to device in the iterator (vs. training loop).

| Property | Value |
|----------|-------|
| Required | No |
| Recommended | `false` (let training loop handle device placement) |

---

<a id="data.tokenizer"></a>
### data.tokenizer (TokenizerConfig)
Nested tokenizer configuration. Contains 10 fields.

<a id="data.tokenizer.kind"></a>
#### `data.tokenizer.kind`
```yaml
kind: "byte" | "hf" = "byte"
```

Tokenizer type.

| Property | Value |
|----------|-------|
| Required | No |
| Valid values | `"byte"`, `"hf"` |

##### byte

Raw byte tokenizer (vocab size 256+offset). No external files needed.
Good for infrastructure bring-up.

##### hf

HuggingFace tokenizer via `transformers.AutoTokenizer`.
**Recommended for real pretraining.**
<a id="data.tokenizer.hf_name_or_path"></a>
#### `data.tokenizer.hf_name_or_path`
```yaml
hf_name_or_path: str | null = "pszemraj/bytebpe-tokenizer-32k-mlm"
```

HuggingFace tokenizer name or local path.

| Property | Value |
|----------|-------|
| Required | Yes (when `kind: hf`) |
| Constraints | Must be set when `kind: hf` |

<a id="data.tokenizer.hf_use_fast"></a>
#### `data.tokenizer.hf_use_fast`
```yaml
hf_use_fast: bool = true
```

Use the fast Rust tokenizer implementation.

| Property | Value |
|----------|-------|
| Required | No |
| Recommended | `true` |

<a id="data.tokenizer.hf_trust_remote_code"></a>
#### `data.tokenizer.hf_trust_remote_code`
```yaml
hf_trust_remote_code: bool = false
```

Trust remote code when loading tokenizer.

| Property | Value |
|----------|-------|
| Required | No |

Only enable if you trust the tokenizer source.
<a id="data.tokenizer.vocab_size_multiple"></a>
#### `data.tokenizer.vocab_size_multiple`
```yaml
vocab_size_multiple: int = 128
```

Round model vocab size up to this multiple for GPU-aligned embeddings.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive |
| Recommended | `128` (aligns with tensor core sizes) |

<a id="data.tokenizer.auto_set_special_tokens"></a>
#### `data.tokenizer.auto_set_special_tokens`
```yaml
auto_set_special_tokens: bool = true
```

Automatically update model config with tokenizer's special token IDs.

| Property | Value |
|----------|-------|
| Required | No |
| Recommended | `true` for HF tokenizers |

<a id="data.tokenizer.byte_offset"></a>
#### `data.tokenizer.byte_offset`
```yaml
byte_offset: int = 0
```

Reserve IDs `[0..byte_offset-1]` for special tokens in byte tokenizer.
Raw bytes map to `[byte_offset..byte_offset+255]`.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be ≥ 0; if `add_bos` or `add_eos` is true, must be > 0 |

If using byte tokenizer with `add_bos: true` or `add_eos: true`, you must set `byte_offset > 0`
to reserve space for special token IDs.
**Related:** [`data.tokenizer.add_bos`](#data.tokenizer.add_bos), [`data.tokenizer.add_eos`](#data.tokenizer.add_eos)

<a id="data.tokenizer.add_bos"></a>
#### `data.tokenizer.add_bos`
```yaml
add_bos: bool = false
```

Prepend BOS token to each document.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | If true with byte tokenizer, requires `byte_offset > 0` and `bos_token_id < byte_offset` |

**Related:** [`model.bos_token_id`](#model.bos_token_id)

<a id="data.tokenizer.add_eos"></a>
#### `data.tokenizer.add_eos`
```yaml
add_eos: bool = false
```

Append EOS token to each document.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | If true with byte tokenizer, requires `byte_offset > 0` and `eos_token_id < byte_offset` |

**Related:** [`model.eos_token_id`](#model.eos_token_id)

<a id="data.tokenizer.max_doc_tokens"></a>
#### `data.tokenizer.max_doc_tokens`
```yaml
max_doc_tokens: int | null = null
```

Truncate documents to this many tokens before packing. `null` defaults to
`4 * train.seq_len` at runtime. Set to `0` (or a negative value) to disable truncation.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive when set (0 or negative disables truncation) |

---

<a id="train"></a>
## train (TrainConfig)
Training loop configuration. Contains 18 fields.

<a id="train.seed"></a>
#### `train.seed`
```yaml
seed: int = 0
```

Random seed for training reproducibility.

| Property | Value |
|----------|-------|
| Required | No |

<a id="train.steps"></a>
#### `train.steps`
```yaml
steps: int = 100
```

Total training steps.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive; must be > `optim.warmup_steps` |

<a id="train.batch_size"></a>
#### `train.batch_size`
```yaml
batch_size: int = 2
```

Micro-batch size per gradient accumulation step.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive |

Effective batch size = `batch_size × grad_accum`
**Related:** [`train.grad_accum`](#train.grad_accum)

<a id="train.seq_len"></a>
#### `train.seq_len`
```yaml
seq_len: int = 128
```

Sequence length in tokens.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be ≥ 8; must be divisible by `model.chunk_size` |

If `seq_len % chunk_size != 0`:
`train.seq_len (X) must be divisible by model.chunk_size (Y)`
**Related:** [`model.chunk_size`](#model.chunk_size)

<a id="train.grad_accum"></a>
#### `train.grad_accum`
```yaml
grad_accum: int = 1
```

Gradient accumulation steps.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive |

Uses scan-based accumulation within a single JIT boundary for efficiency.
<a id="train.jit"></a>
#### `train.jit`
```yaml
jit: bool = true
```

Enable JIT compilation of the training step.

| Property | Value |
|----------|-------|
| Required | No |
| Recommended | `true` (always) |

<a id="train.deterministic"></a>
#### `train.deterministic`
```yaml
deterministic: bool | null = null
```

Force deterministic training. `null` auto-derives from dropout settings.

| Property | Value |
|----------|-------|
| Required | No |

When `null`: deterministic if all dropout rates are 0.0.
In `megalodon-jax`, `train.deterministic=true` disables activation checkpointing
even if `model.use_checkpoint=true`.
<a id="train.allow_cpu"></a>
#### `train.allow_cpu`
```yaml
allow_cpu: bool = false
```

Allow training on CPU (for debugging only).

| Property | Value |
|----------|-------|
| Required | No |

If `false` and no GPU is available, training fails immediately with an assertion error.
<a id="train.log_every"></a>
#### `train.log_every`
```yaml
log_every: int = 25
```

Log metrics every N steps.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive |

<a id="train.eval_every"></a>
#### `train.eval_every`
```yaml
eval_every: int = 2500
```

Run evaluation every N steps. Set to `0` to disable periodic evaluation.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be ≥ 0 |

<a id="train.generate_every"></a>
#### `train.generate_every`
```yaml
generate_every: int = 5000
```

Run text generation every N steps. Set to `0` to disable.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be ≥ 0 |

<a id="train.generate_input_len"></a>
#### `train.generate_input_len`
```yaml
generate_input_len: int | null = null
```

Prompt length (tokens) for generation samples. `null` uses half of
`train.seq_len`. If a sample is longer, the prompt is taken from the first or
last `generate_input_len` tokens at random.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive and ≤ `train.seq_len` when set |

<a id="train.generate_max_tokens"></a>
#### `train.generate_max_tokens`
```yaml
generate_max_tokens: int | null = null
```

Maximum number of new tokens to generate. `null` uses `model.chunk_size + 16`.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive when set |

<a id="train.generate_temperature"></a>
#### `train.generate_temperature`
```yaml
generate_temperature: float | null = null
```

Sampling temperature. `null` uses the Megalodon default (`1.0`).

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be ≥ 0 when set |

<a id="train.generate_top_k"></a>
#### `train.generate_top_k`
```yaml
generate_top_k: int | null = null
```

Top-k sampling cutoff. `null` uses the Megalodon default.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive when set |

<a id="train.generate_top_p"></a>
#### `train.generate_top_p`
```yaml
generate_top_p: float | null = null
```

Nucleus sampling threshold. `null` uses the Megalodon default.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be in (0, 1] when set |

<a id="train.profile"></a>
#### `train.profile`
```yaml
profile: bool = false
```

Enable JAX profiler trace collection.

| Property | Value |
|----------|-------|
| Required | No |

<a id="train.profile_dir"></a>
#### `train.profile_dir`
```yaml
profile_dir: str | null = null
```

Directory for profiler traces. `null` uses a default location.

| Property | Value |
|----------|-------|
| Required | No |

---

<a id="optim"></a>
## optim (OptimConfig)
Optimizer configuration for AdamW or Muon with linear warmup and cosine decay.
Contains 9 top-level fields plus nested `optim.muon` and `optim.adam`
sub-configs.

<a id="optim.name"></a>
#### `optim.name`
```yaml
name: "adamw" | "muon" = "adamw"
```

Optimizer algorithm. `"muon"` applies Muon to eligible 2D weight matrices and AdamW
to everything else.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be `"adamw"` or `"muon"` |

<a id="optim.lr"></a>
#### `optim.lr`
```yaml
lr: float = 3e-4
```

Peak learning rate.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive |
| Recommended | `3e-4` (small models), `1e-4` to `6e-5` (large models) |

<a id="optim.weight_decay"></a>
#### `optim.weight_decay`
```yaml
weight_decay: float = 0.01
```

AdamW weight decay coefficient.

| Property | Value |
|----------|-------|
| Required | No |

<a id="optim.grad_clip_norm"></a>
#### `optim.grad_clip_norm`
```yaml
grad_clip_norm: float = 1.0
```

Maximum gradient norm for clipping. Set to `0` to disable.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be ≥ 0 |

<a id="optim.warmup_steps"></a>
#### `optim.warmup_steps`
```yaml
warmup_steps: int = 10
```

Linear warmup steps.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be ≥ 0; **must be < `train.steps`** |

If `warmup_steps >= train.steps`:
`optim.warmup_steps (X) must be < train.steps (Y)`
**Related:** [`train.steps`](#train.steps)

<a id="optim.decay_steps"></a>
#### `optim.decay_steps`
```yaml
decay_steps: int | null = null
```

Cosine decay duration (in steps) after warmup. The effective schedule horizon is
`optim.warmup_steps + optim.decay_steps`. `null` defaults to
`train.steps - optim.warmup_steps` so the schedule ends at `train.steps`.
Set explicitly if you plan to resume with a different `train.steps` but keep the
same LR schedule.

Implementation note: Optax's `warmup_cosine_decay_schedule` expects
`decay_steps` to be the total horizon including warmup. chomp therefore passes
`optim.warmup_steps + optim.decay_steps` to Optax.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive when set |

**Related:** [`train.steps`](#train.steps)

<a id="optim.min_lr_ratio"></a>
#### `optim.min_lr_ratio`
```yaml
min_lr_ratio: float = 0.0
```

Minimum LR as a ratio of peak LR at the end of cosine decay.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be in `[0.0, 1.0]` |
| Recommended | `0.0` (decay to zero) or `0.1` (10% floor) |

<a id="optim.muon"></a>
#### `optim.muon`
```yaml
muon:
  lr_scale: float = 100.0
  weight_decay_mult: float = 1.0
  momentum: float = 0.95
  ns_steps: int = 5
  nesterov: bool = true
  consistent_rms: float | null = null
  allow_all_2d: bool = false
  allow_tied_embed: bool = false
```

Muon-specific settings. These are only used when `optim.name=muon`.

| Property | Value |
|----------|-------|
| Required | No |

<a id="optim.muon.lr_scale"></a>
#### `optim.muon.lr_scale`
```yaml
lr_scale: float = 100.0
```

Multiplier applied to `optim.lr` to set Muon's peak LR. Effective Muon LR is
`optim.lr * optim.muon.lr_scale`. The default (`100.0`) reflects the best
10k-step Muon sweep result so far; see `docs/optimization.md` for details.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive |

<a id="optim.muon.weight_decay_mult"></a>
#### `optim.muon.weight_decay_mult`
```yaml
weight_decay_mult: float = 1.0
```

Multiplier applied to `optim.weight_decay` for Muon parameters. Set to `0` to
disable Muon weight decay or scale it independently from AdamW.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be >= 0 |

<a id="optim.muon.momentum"></a>
#### `optim.muon.momentum`
```yaml
momentum: float = 0.95
```

Muon momentum coefficient (`beta` in the Optax implementation).

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be in `(0, 1)` |

<a id="optim.muon.ns_steps"></a>
#### `optim.muon.ns_steps`
```yaml
ns_steps: int = 5
```

Number of Newton-Schulz iterations for Muon.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive |

<a id="optim.muon.nesterov"></a>
#### `optim.muon.nesterov`
```yaml
nesterov: bool = true
```

Use Nesterov momentum in Muon.

| Property | Value |
|----------|-------|
| Required | No |

<a id="optim.muon.consistent_rms"></a>
#### `optim.muon.consistent_rms`
```yaml
consistent_rms: float | null = null
```

Muon RMS scaling factor. Set to `0.2` to enable consistent RMS scaling (Optax
recommendation) or leave `null` to disable shape scaling and match the earlier
Muon behavior.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be >= 0 or null |

<a id="optim.muon.allow_all_2d"></a>
#### `optim.muon.allow_all_2d`
```yaml
allow_all_2d: bool = false
```

Allow Muon updates on all 2D tensors (overrides the projection-weight whitelist).

| Property | Value |
|----------|-------|
| Required | No |

<a id="optim.muon.allow_tied_embed"></a>
#### `optim.muon.allow_tied_embed`
```yaml
allow_tied_embed: bool = false
```

Allow Muon updates on the tied token embedding matrix (`model.embed.weight`).
If `false` (default), Muon applies to 2D weight matrices named `*.weight` and
excludes embeddings. If `true`, Muon applies to all 2D tensors.

| Property | Value |
|----------|-------|
| Required | No |

<a id="optim.adam"></a>
#### `optim.adam`
```yaml
adam:
  b1: float = 0.9
  b2: float = 0.999
  eps: float = 1e-8
  nesterov: bool = false
```

AdamW-specific settings. For `optim.name=adamw`, these apply to the full model.
For `optim.name=muon`, they apply to the non-Muon parameter group.

| Property | Value |
|----------|-------|
| Required | No |

<a id="optim.adam.b1"></a>
#### `optim.adam.b1`
```yaml
b1: float = 0.9
```

AdamW beta1 coefficient.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be in `(0, 1)` |

<a id="optim.adam.b2"></a>
#### `optim.adam.b2`
```yaml
b2: float = 0.999
```

AdamW beta2 coefficient.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be in `(0, 1)` |

<a id="optim.adam.eps"></a>
#### `optim.adam.eps`
```yaml
eps: float = 1e-8
```

AdamW epsilon for numerical stability.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive |

<a id="optim.adam.nesterov"></a>
#### `optim.adam.nesterov`
```yaml
nesterov: bool = false
```

Enable Nesterov momentum for AdamW. This is kept independent from
`optim.muon.nesterov` to avoid accidentally running NadamW on non-Muon params.

| Property | Value |
|----------|-------|
| Required | No |

---

<a id="checkpoint"></a>
## checkpoint (CheckpointConfig)
Orbax checkpointing configuration. Contains 5 fields.

<a id="checkpoint.enabled"></a>
#### `checkpoint.enabled`
```yaml
enabled: bool = true
```

Enable checkpointing.

| Property | Value |
|----------|-------|
| Required | No |

<a id="checkpoint.root_dir"></a>
#### `checkpoint.root_dir`
```yaml
root_dir: str | null = null
```

Checkpoint directory. `null` uses `<run_dir>/checkpoints`. Relative paths are resolved against
`run_dir`.

| Property | Value |
|----------|-------|
| Required | No |

<a id="checkpoint.save_every"></a>
#### `checkpoint.save_every`
```yaml
save_every: int = 5000
```

Save checkpoint every N steps.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive when checkpointing is enabled |

<a id="checkpoint.max_to_keep"></a>
#### `checkpoint.max_to_keep`
```yaml
max_to_keep: int = 3
```

Maximum number of checkpoints to retain.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be positive when checkpointing is enabled |

<a id="checkpoint.async_save"></a>
#### `checkpoint.async_save`
```yaml
async_save: bool = true
```

Use asynchronous checkpoint saving.

| Property | Value |
|----------|-------|
| Required | No |
| Recommended | `true` for production (reduces training stalls) |

---

<a id="logging"></a>
## logging (LoggingConfig)
Logging and metrics configuration. Contains 7 fields.

<a id="logging.project"></a>
#### `logging.project`
```yaml
project: str = "chomp"
```

Project name for run organization.

| Property | Value |
|----------|-------|
| Required | No |

<a id="logging.run_dir"></a>
#### `logging.run_dir`
```yaml
run_dir: str | null = null
```

Run directory for outputs. `null` auto-generates a timestamped directory.

| Property | Value |
|----------|-------|
| Required | No |

<a id="logging.metrics_file"></a>
#### `logging.metrics_file`
```yaml
metrics_file: str = "metrics.jsonl"
```

Filename for JSONL metrics output (within run_dir).

| Property | Value |
|----------|-------|
| Required | No |

<a id="logging.level"></a>
#### `logging.level`
```yaml
level: "DEBUG" | "INFO" | "WARNING" | "ERROR" = "INFO"
```

Logging level.

| Property | Value |
|----------|-------|
| Required | No |

<a id="logging.console_use_rich"></a>
#### `logging.console_use_rich`
```yaml
console_use_rich: bool = true
```

Use Rich library for formatted console output.

| Property | Value |
|----------|-------|
| Required | No |

<a id="logging.log_file"></a>
#### `logging.log_file`
```yaml
log_file: str | null = "train.log"
```

Log file name (within run_dir). `null` disables file logging.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be non-empty if set |

---

<a id="logging.wandb"></a>
### logging.wandb (WandbConfig)
Weights & Biases integration. Contains 6 fields.

<a id="logging.wandb.enabled"></a>
#### `logging.wandb.enabled`
```yaml
enabled: bool = false
```

Enable W&B logging.

| Property | Value |
|----------|-------|
| Required | No |

<a id="logging.wandb.project"></a>
#### `logging.wandb.project`
```yaml
project: str | null = null
```

W&B project name.

| Property | Value |
|----------|-------|
| Required | No |

<a id="logging.wandb.entity"></a>
#### `logging.wandb.entity`
```yaml
entity: str | null = null
```

W&B entity (username or team).

| Property | Value |
|----------|-------|
| Required | No |

<a id="logging.wandb.run_name"></a>
#### `logging.wandb.run_name`
```yaml
run_name: str | null = null
```

W&B run name. `null` auto-generates.

| Property | Value |
|----------|-------|
| Required | No |

<a id="logging.wandb.mode"></a>
#### `logging.wandb.mode`
```yaml
mode: "online" | "offline" | "disabled" = "online"
```

W&B sync mode.

| Property | Value |
|----------|-------|
| Required | No |
| Constraints | Must be one of `"online"`, `"offline"`, `"disabled"` |

<a id="logging.wandb.tags"></a>
#### `logging.wandb.tags`
```yaml
tags: list[str] = []
```

W&B run tags.

| Property | Value |
|----------|-------|
| Required | No |

YAML should provide a list; tags are stored internally as a tuple.
---

<a id="debug"></a>
## debug (DebugConfig)
Debug configuration. Contains 2 fields.

<a id="debug.nan_check"></a>
#### `debug.nan_check`
```yaml
nan_check: bool = true
```

Check for NaN/Inf in loss every step.

| Property | Value |
|----------|-------|
| Required | No |
| Recommended | `true` (catches instability early) |

<a id="debug.check_device_every"></a>
#### `debug.check_device_every`
```yaml
check_device_every: int = 100
```

Verify GPU backend every N steps.

| Property | Value |
|----------|-------|
| Required | No |

---

<a id="validation"></a>
## Validation Rules & Constraints
chomp validates all configuration at load time with actionable error messages.

### Cross-Field Dependencies

| Constraint | Error Message |
|------------|---------------|
| `train.seq_len % model.chunk_size == 0` | `train.seq_len (X) must be divisible by model.chunk_size (Y)` |
| `model.chunk_size <= train.seq_len` | `model.chunk_size (X) must be <= train.seq_len (Y)` |
| `train.generate_input_len <= train.seq_len` | `train.generate_input_len must be <= train.seq_len (X), got Y` |
| `model.model_dim % model.num_heads == 0` | `model.model_dim (X) must be divisible by model.num_heads (Y)` |
| `optim.warmup_steps < train.steps` | `optim.warmup_steps (X) must be < train.steps (Y)` |
| `data.packing_buffer_docs >= batch_size × grad_accum` (bin mode) | `data.packing_buffer_docs must be >= train.batch_size * train.grad_accum (N), got M` |
| `byte_offset > 0` when `add_bos` or `add_eos` (byte tokenizer) | `byte tokenizer with add_bos/add_eos requires data.tokenizer.byte_offset > 0` |

### Critical Gotchas

## Never Use fp16

Megalodon's CEMA and normalization layers are numerically unstable with fp16.
Always use the recommended precision policy:

```yaml
model:
  param_dtype: float32
  compute_dtype: bfloat16
  accum_dtype: float32
  softmax_dtype: float32
```
## pad_token_id Should Differ from eos_token_id

If your tokenizer uses the same ID for padding and EOS (common with GPT-2 style tokenizers),
chomp emits a warning after tokenizer resolution. Prefer a tokenizer with distinct pad/eos, or
disable `data.tokenizer.auto_set_special_tokens` and override `model.pad_token_id` explicitly:

```yaml
model:
  pad_token_id: 3   # Different from eos_token_id
  eos_token_id: 2
```
---

<a id="index"></a>
## Field Index
All configuration fields by section:

**model** (35 fields): [`backend`](#model.backend), [`vocab_size`](#model.vocab_size), [`d_model`](#model.d_model), [`dropout`](#model.dropout), [`model_dim`](#model.model_dim), [`num_layers`](#model.num_layers), [`num_heads`](#model.num_heads), [`z_dim`](#model.z_dim), [`value_dim`](#model.value_dim), [`ffn_hidden_dim`](#model.ffn_hidden_dim), [`cema_ndim`](#model.cema_ndim), [`chunk_size`](#model.chunk_size), [`max_cache_len`](#model.max_cache_len), [`cache_unbounded`](#model.cache_unbounded), [`norm_num_groups`](#model.norm_num_groups), [`norm_eps`](#model.norm_eps), [`rope_base`](#model.rope_base), [`swiglu`](#model.swiglu), [`rescale_nffn`](#model.rescale_nffn), [`scale_emb`](#model.scale_emb), [`norm_affine`](#model.norm_affine), [`attention_dropout`](#model.attention_dropout), [`hidden_dropout`](#model.hidden_dropout), [`pad_token_id`](#model.pad_token_id), [`bos_token_id`](#model.bos_token_id), [`eos_token_id`](#model.eos_token_id), [`max_positions`](#model.max_positions), [`init_mode`](#model.init_mode), [`use_checkpoint`](#model.use_checkpoint), [`output_size`](#model.output_size), [`param_dtype`](#model.param_dtype), [`compute_dtype`](#model.compute_dtype), [`accum_dtype`](#model.accum_dtype), [`softmax_dtype`](#model.softmax_dtype), [`gemm_backend`](#model.gemm_backend)

**data** (23 fields): [`backend`](#data.backend), [`hf_dataset`](#data.hf_dataset), [`hf_name`](#data.hf_name), [`hf_split`](#data.hf_split), [`hf_eval_split`](#data.hf_eval_split), [`text_key`](#data.text_key), [`shuffle`](#data.shuffle), [`shuffle_buffer_size`](#data.shuffle_buffer_size), [`seed`](#data.seed), [`repeat`](#data.repeat), [`max_retries`](#data.max_retries), [`retry_delay_sec`](#data.retry_delay_sec), [`state_update_interval`](#data.state_update_interval), [`local_text`](#data.local_text), [`packing_mode`](#data.packing_mode), [`packing_buffer_docs`](#data.packing_buffer_docs), [`packing_max_docs_per_bin`](#data.packing_max_docs_per_bin), [`mask_boundary_loss`](#data.mask_boundary_loss), [`train_on_eos`](#data.train_on_eos), [`grain_prefetch`](#data.grain_prefetch), [`max_eval_samples`](#data.max_eval_samples), [`tokenizer`](#data.tokenizer), [`device_put`](#data.device_put)

**data.tokenizer** (10 fields): [`kind`](#data.tokenizer.kind), [`hf_name_or_path`](#data.tokenizer.hf_name_or_path), [`hf_use_fast`](#data.tokenizer.hf_use_fast), [`hf_trust_remote_code`](#data.tokenizer.hf_trust_remote_code), [`vocab_size_multiple`](#data.tokenizer.vocab_size_multiple), [`auto_set_special_tokens`](#data.tokenizer.auto_set_special_tokens), [`byte_offset`](#data.tokenizer.byte_offset), [`add_bos`](#data.tokenizer.add_bos), [`add_eos`](#data.tokenizer.add_eos), [`max_doc_tokens`](#data.tokenizer.max_doc_tokens)

**train** (18 fields): [`seed`](#train.seed), [`steps`](#train.steps), [`batch_size`](#train.batch_size), [`seq_len`](#train.seq_len), [`grad_accum`](#train.grad_accum), [`jit`](#train.jit), [`deterministic`](#train.deterministic), [`allow_cpu`](#train.allow_cpu), [`log_every`](#train.log_every), [`eval_every`](#train.eval_every), [`generate_every`](#train.generate_every), [`generate_input_len`](#train.generate_input_len), [`generate_max_tokens`](#train.generate_max_tokens), [`generate_temperature`](#train.generate_temperature), [`generate_top_k`](#train.generate_top_k), [`generate_top_p`](#train.generate_top_p), [`profile`](#train.profile), [`profile_dir`](#train.profile_dir)

**optim** (9 fields): [`name`](#optim.name), [`lr`](#optim.lr), [`weight_decay`](#optim.weight_decay), [`grad_clip_norm`](#optim.grad_clip_norm), [`warmup_steps`](#optim.warmup_steps), [`decay_steps`](#optim.decay_steps), [`min_lr_ratio`](#optim.min_lr_ratio), [`muon`](#optim.muon), [`adam`](#optim.adam)

**optim.muon** (8 fields): [`lr_scale`](#optim.muon.lr_scale), [`weight_decay_mult`](#optim.muon.weight_decay_mult), [`momentum`](#optim.muon.momentum), [`ns_steps`](#optim.muon.ns_steps), [`nesterov`](#optim.muon.nesterov), [`consistent_rms`](#optim.muon.consistent_rms), [`allow_all_2d`](#optim.muon.allow_all_2d), [`allow_tied_embed`](#optim.muon.allow_tied_embed)

**optim.adam** (4 fields): [`b1`](#optim.adam.b1), [`b2`](#optim.adam.b2), [`eps`](#optim.adam.eps), [`nesterov`](#optim.adam.nesterov)

**checkpoint** (5 fields): [`enabled`](#checkpoint.enabled), [`root_dir`](#checkpoint.root_dir), [`save_every`](#checkpoint.save_every), [`max_to_keep`](#checkpoint.max_to_keep), [`async_save`](#checkpoint.async_save)

**logging** (7 fields): [`project`](#logging.project), [`run_dir`](#logging.run_dir), [`metrics_file`](#logging.metrics_file), [`level`](#logging.level), [`console_use_rich`](#logging.console_use_rich), [`log_file`](#logging.log_file), [`wandb`](#logging.wandb)

**logging.wandb** (6 fields): [`enabled`](#logging.wandb.enabled), [`project`](#logging.wandb.project), [`entity`](#logging.wandb.entity), [`run_name`](#logging.wandb.run_name), [`mode`](#logging.wandb.mode), [`tags`](#logging.wandb.tags)

**debug** (2 fields): [`nan_check`](#debug.nan_check), [`check_device_every`](#debug.check_device_every)

**Total: 109 fields**
