"""Tabular feature synthesizer with differential privacy guarantees."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from .._deps import np

from ..dp.accountant import DPAccountant, PrivacyBudget
from ..dp.mechanisms import GaussianMechanism, LaplaceMechanism


@dataclass
class ColumnMetadata:
    """Metadata describing how to privately model a column."""

    name: str
    kind: str  # "continuous" or "categorical"
    categories: Sequence[int | float] | None = None
    clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"continuous", "categorical"}:
            raise ValueError("kind must be either 'continuous' or 'categorical'.")
        if self.kind == "categorical" and not self.categories:
            raise ValueError("Categorical columns require the 'categories' field.")


class DPFeatureSynthesizer:
    """Fits independent marginal distributions with differential privacy."""

    def __init__(
        self,
        columns: Sequence[ColumnMetadata],
        total_budget: PrivacyBudget,
        *,
        temperature: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.columns = list(columns)
        self.total_budget = total_budget
        self.temperature = temperature
        self.rng = np.random.default_rng(random_state)
        self._fitted = False
        self._column_stats: List[dict[str, np.ndarray | float | tuple[float, float]]] = []

        has_continuous = any(col.kind == "continuous" for col in self.columns)
        if has_continuous and self.total_budget.delta <= 0:
            raise ValueError("Continuous columns require a positive delta for the Gaussian mechanism.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")

    def fit(self, data: np.ndarray, accountant: DPAccountant) -> None:
        """Fit DP marginals for each column.

        Parameters
        ----------
        data:
            A 2D array with shape (n_samples, n_features).
        accountant:
            Accountant used to ensure the privacy budget is not exceeded.
        """

        if data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional (n_samples, n_features).")
        if data.shape[1] != len(self.columns):
            raise ValueError("Number of columns in data does not match metadata.")

        accountant.spend(self.total_budget.epsilon, self.total_budget.delta)
        self._column_stats.clear()

        n_samples = data.shape[0]
        epsilon_per_column = self.total_budget.epsilon / len(self.columns)
        continuous_columns = [col for col in self.columns if col.kind == "continuous"]
        delta_per_cont = (
            self.total_budget.delta / len(continuous_columns) if continuous_columns else 0.0
        )

        for idx, column in enumerate(self.columns):
            column_data = data[:, idx]
            if column.kind == "continuous":
                clip_range = column.clip
                if clip_range is None:
                    lower = float(np.min(column_data))
                    upper = float(np.max(column_data))
                else:
                    lower, upper = clip_range
                clipped = np.clip(column_data, lower, upper)
                sensitivity_mean = (upper - lower) / max(n_samples, 1)
                sensitivity_var = (upper - lower) ** 2 / max(n_samples, 1)

                gaussian = GaussianMechanism(
                    epsilon=epsilon_per_column,
                    delta=delta_per_cont,
                )
                noisy_mean = float(clipped.mean() + gaussian.noise(sensitivity_mean, 1)[0])
                noisy_var = float(clipped.var(ddof=0) + gaussian.noise(sensitivity_var, 1)[0])
                noisy_var = max(noisy_var, 1e-6)
                stat = {
                    "type": "continuous",
                    "mean": noisy_mean,
                    "var": noisy_var,
                    "clip": (lower, upper),
                }
            else:
                categories = np.array(column.categories, dtype=float)
                laplace = LaplaceMechanism(epsilon=epsilon_per_column)
                counts = np.zeros(len(categories), dtype=float)
                for cat_idx, category in enumerate(categories):
                    counts[cat_idx] = np.sum(column_data == category)
                noisy_counts = counts + laplace.noise(1.0, counts.shape)
                noisy_counts = np.clip(noisy_counts, a_min=1e-9, a_max=None)
                probs = noisy_counts / noisy_counts.sum()
                stat = {
                    "type": "categorical",
                    "categories": categories,
                    "probs": probs,
                }
            self._column_stats.append(stat)

        self._fitted = True

    def sample(self, n_samples: int) -> np.ndarray:
        """Draw synthetic records using the sanitized marginals."""

        if not self._fitted:
            raise RuntimeError("Synthesizer must be fitted before sampling.")
        samples = np.zeros((n_samples, len(self.columns)))
        for idx, stat in enumerate(self._column_stats):
            if stat["type"] == "continuous":
                mean = float(stat["mean"])
                var = float(stat["var"])
                std = math.sqrt(var / max(self.temperature, 1e-6))
                lower, upper = stat["clip"]
                draws = self.rng.normal(loc=mean, scale=std, size=n_samples)
                samples[:, idx] = np.clip(draws, lower, upper)
            else:
                categories = stat["categories"]
                probs = stat["probs"]
                indices = self.rng.choice(len(categories), size=n_samples, p=probs)
                samples[:, idx] = categories[indices]
        return samples
