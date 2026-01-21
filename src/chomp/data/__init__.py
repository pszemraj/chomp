"""Data loading for chomp.

Minimal by design.

Phases 0–2 shipped with synthetic batches to bring the trainer up quickly.
That was useful for bootstrapping, but it's a trap: it lets people believe the
trainer works without exercising streaming/tokenization/packing.

From this point forward, training should use *real* data through this module.

v0 (this draft):
- HF streaming or local_text (offline)
- Tokenize + pack to fixed [A,B,T]

Later:
- Grain pipeline + iterator checkpointing lives here, but we keep the public
  contract the same: `build_train_iterator(cfg)` yields `Batch` objects.
"""

from __future__ import annotations

from .pipeline import build_train_iterator, data_fingerprint, build_tokenizer

__all__ = ["build_train_iterator", "data_fingerprint", "build_tokenizer"]
