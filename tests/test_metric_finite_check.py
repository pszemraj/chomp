"""Finite metric checks should catch NaNs/Infs early."""

from __future__ import annotations

import pytest

from chomp.train import _check_finite_metrics


def test_finite_check_rejects_nan_loss():
    with pytest.raises(RuntimeError, match="loss"):
        _check_finite_metrics({"loss": float("nan"), "grad_norm": 1.0}, step=3)


def test_finite_check_rejects_inf_grad_norm():
    with pytest.raises(RuntimeError, match="grad_norm"):
        _check_finite_metrics({"loss": 1.0, "grad_norm": float("inf")}, step=3)
