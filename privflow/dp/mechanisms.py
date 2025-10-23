"""Noise mechanisms used throughout the framework."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .._deps import np


class NoiseMechanism(Protocol):
    """Protocol describing a noise mechanism."""

    def noise(self, sensitivity: float, size: int | tuple[int, ...]) -> np.ndarray:
        """Sample random noise with the provided sensitivity."""


@dataclass
class LaplaceMechanism:
    """Laplace mechanism for pure-epsilon differential privacy."""

    epsilon: float

    def noise(self, sensitivity: float, size: int | tuple[int, ...]) -> np.ndarray:
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive for the Laplace mechanism.")
        scale = sensitivity / self.epsilon
        return np.random.laplace(loc=0.0, scale=scale, size=size)


@dataclass
class GaussianMechanism:
    """Gaussian mechanism for approximate differential privacy."""

    epsilon: float
    delta: float

    def noise(self, sensitivity: float, size: int | tuple[int, ...]) -> np.ndarray:
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive for the Gaussian mechanism.")
        if not (0 < self.delta < 1):
            raise ValueError("Delta must be within (0, 1) for the Gaussian mechanism.")
        sigma = math.sqrt(2 * math.log(1.25 / self.delta)) * sensitivity / self.epsilon
        return np.random.normal(loc=0.0, scale=sigma, size=size)
