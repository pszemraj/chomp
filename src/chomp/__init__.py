"""chomp: minimal JAX/Equinox pretraining harness.

This package is intentionally small.

If you find yourself wanting to add a new subpackage, pause and ask:
can this be a function inside an existing module?

Implemented so far:
- config + logging + device validation
- model integration (dummy or Megalodon-JAX)
- single-GPU train_step with scan-based grad accumulation
- Orbax checkpointing + resume (train_state + data iterator state)
- minimal HF streaming + tokenize + pack iterator (no Grain yet)
"""

from __future__ import annotations

try:
    from chomp._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
