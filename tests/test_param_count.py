import jax

from chomp.config import Config, ModelConfig, TrainConfig
from chomp.model import build_model
from chomp.utils.tree import param_count


def test_dummy_param_count():
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=128, d_model=64, dropout=0.0),
        train=TrainConfig(allow_cpu=True),
    )
    key = jax.random.PRNGKey(0)
    params, static = build_model(cfg, key=key)
    n = param_count(params)
    expected = 2 * cfg.model.vocab_size * cfg.model.d_model
    assert n == expected
