"""Privacy accounting utilities for PrivFlow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PrivacyBudget:
    """Represents an (epsilon, delta) privacy budget."""

    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive.")
        if not (0 <= self.delta < 1):
            raise ValueError("Delta must be in [0, 1).")


class DPAccountant:
    """Simple accountant that tracks cumulative privacy loss."""

    def __init__(self, total_budget: PrivacyBudget):
        self._total = total_budget
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0

    @property
    def remaining(self) -> PrivacyBudget:
        """Return the remaining (epsilon, delta)."""

        return PrivacyBudget(
            epsilon=max(self._total.epsilon - self._spent_epsilon, 0.0),
            delta=max(self._total.delta - self._spent_delta, 0.0),
        )

    @property
    def spent(self) -> PrivacyBudget:
        """Return the spent (epsilon, delta)."""

        return PrivacyBudget(epsilon=self._spent_epsilon, delta=self._spent_delta)

    def spend(self, epsilon: float, delta: float = 0.0) -> None:
        """Record usage of part of the privacy budget."""

        if epsilon < 0 or delta < 0:
            raise ValueError("Privacy usage must be non-negative.")
        new_epsilon = self._spent_epsilon + epsilon
        new_delta = self._spent_delta + delta
        if new_epsilon > self._total.epsilon + 1e-12 or new_delta > self._total.delta + 1e-12:
            raise ValueError(
                "Privacy budget exceeded: requested (epsilon, delta)=({:.3f}, {:.3e}) "
                "with remaining ({:.3f}, {:.3e}).".format(
                    epsilon,
                    delta,
                    self._total.epsilon - self._spent_epsilon,
                    self._total.delta - self._spent_delta,
                )
            )
        self._spent_epsilon = new_epsilon
        self._spent_delta = new_delta

    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        """Check if the requested privacy usage fits in the remaining budget."""

        try:
            self._validate_spend(epsilon, delta)
        except ValueError:
            return False
        return True

    def _validate_spend(self, epsilon: float, delta: float) -> None:
        if epsilon < 0 or delta < 0:
            raise ValueError("Privacy usage must be non-negative.")
        if (self._spent_epsilon + epsilon > self._total.epsilon + 1e-12) or (
            self._spent_delta + delta > self._total.delta + 1e-12
        ):
            raise ValueError("Privacy budget exceeded.")

    def fraction_used(self) -> Optional[float]:
        """Return the maximum fraction of epsilon or delta that has been consumed."""

        if self._total.epsilon == 0 or self._total.delta == 0:
            return None
        epsilon_fraction = self._spent_epsilon / self._total.epsilon if self._total.epsilon else 0.0
        delta_fraction = self._spent_delta / self._total.delta if self._total.delta else 0.0
        return max(epsilon_fraction, delta_fraction)
