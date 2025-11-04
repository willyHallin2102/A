"""
"""
from __future__ import annotations

import tensorflow as tf
tfk = tf.keras
from abc import ABC, abstractmethod


class GenAi(tfk.Model, ABC):
    """
    Abstract base class for all generative AI models. Provides a unified interface
    (sample, save, load, etc).
    """

    @abstractmethod
    def sample(self, conditions: tf.Tensor, n_samples: int=1) -> tf.Tensor:
        raise ModuleNotFoundError("sample() must be implemented by subclass")