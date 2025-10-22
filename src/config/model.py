"""
    src/config/model.py
    -------------------
    Configuration file for the model parameters and definitions
"""
from dataclasses import dataclass
from typing import Final, Literal, Sequence, Tuple


@dataclass(slots=True)
class ModelConfig:
    n_latent        : int   = 10
    min_variance    : float = 1e-4
    dropout_rate    : float = 0.20
    init_kernel     : float = 10.0
    init_bias       : float = 10.0

    def __post_init__(self):
        if self.n_latent <= 0:
            raise ValueError("n_latent must be positive")
        if self.min_variance < 0:
            raise ValueError("min_variance cannot be negative")
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0.0 and 1.0")


# Less strict, ... uses Sequence rather than Tuple[int, ...]
#   however perhaps should be changed later on...
def _validate_layers(name: str, layers: Sequence[int]):
    if any(layer <= 0 for layer in layers):
        raise ValueError(f"All {name} layers must be positive")


@dataclass(slots=True)
class VaeConfig(ModelConfig):
    encoder_layers  : Tuple[int, ...] = (200, 80)
    decoder_layers  : Tuple[int, ...] = (80, 200)

    beta    : float = 0.50
    beta_annealing_step : int = 100_000
    kl_warmup_steps : int   = 20

    def __post_init__(self):
        ModelConfig.__post_init__(self)

        _validate_layers("encoder", self.encoder_layers)
        _validate_layers("decoder", self.decoder_layers)

        if not (0.0 <= self.beta <= 1.0):
            raise ValueError("beta must be between 0.0 and 1.0")



MODEL_CONFIGS = {
    "vae": VaeConfig,
}
def get_config(model_type: str) -> ModelConfig:
    """
    """
    model_type = model_type.lower()
    try: return MODEL_CONFIGS[model_type]()
    except KeyError:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Supported types: {', '.join(VALID_MODELS)}")
