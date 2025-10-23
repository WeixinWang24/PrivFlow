"""Logistic regression models used for knowledge distillation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .._deps import np

from ..dp.accountant import DPAccountant, PrivacyBudget


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class DPLogisticRegression:
    """Binary logistic regression trained with DP-SGD."""

    learning_rate: float = 0.1
    epochs: int = 50
    batch_size: int = 128
    clipping_norm: float = 1.0
    noise_multiplier: float = 1.0
    fit_intercept: bool = True
    privacy_usage: Optional[PrivacyBudget] = None
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")

    def fit(self, X: np.ndarray, y: np.ndarray, accountant: DPAccountant | None = None) -> None:
        if self.privacy_usage and accountant:
            accountant.spend(self.privacy_usage.epsilon, self.privacy_usage.delta)

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        y = np.asarray(y, dtype=float).reshape(n_samples)
        if np.any((y < 0) | (y > 1)):
            raise ValueError("Labels must be in the range [0, 1].")
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.epochs):
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                batch_X = X[batch_idx]
                batch_y = y[batch_idx]
                logits = batch_X @ self.coef_ + self.intercept_
                probs = _sigmoid(logits)
                errors = probs - batch_y
                grad_w = errors[:, None] * batch_X
                grad_b = errors

                grad = np.concatenate([grad_w, grad_b[:, None]], axis=1)
                norms = np.linalg.norm(grad, axis=1)
                scaling = np.minimum(1.0, self.clipping_norm / (norms + 1e-12))
                grad_w = grad_w * scaling[:, None]
                grad_b = grad_b * scaling

                mean_grad_w = grad_w.mean(axis=0)
                mean_grad_b = grad_b.mean()

                noise_scale = self.noise_multiplier * self.clipping_norm / max(len(batch_idx), 1)
                noise_w = rng.normal(loc=0.0, scale=noise_scale, size=mean_grad_w.shape)
                noise_b = rng.normal(loc=0.0, scale=noise_scale)

                self.coef_ -= self.learning_rate * (mean_grad_w + noise_w)
                if self.fit_intercept:
                    self.intercept_ -= self.learning_rate * (mean_grad_b + noise_b)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.coef_
        if self.fit_intercept:
            logits += self.intercept_
        return np.clip(_sigmoid(logits), 1e-6, 1 - 1e-6)


@dataclass
class DistilledLogisticRegression:
    """Student model trained on the teacher's soft predictions."""

    learning_rate: float = 0.05
    epochs: int = 200
    batch_size: int = 128
    temperature: float = 1.0
    l2: float = 0.0
    fit_intercept: bool = True
    random_state: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        y = np.asarray(y, dtype=float).reshape(n_samples)
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.epochs):
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                batch_X = X[batch_idx]
                batch_y = y[batch_idx]

                logits = (batch_X @ self.coef_ + self.intercept_) / max(self.temperature, 1e-6)
                probs = _sigmoid(logits)
                errors = probs - batch_y

                grad_w = (batch_X.T @ errors) / (len(batch_idx) * max(self.temperature, 1e-6))
                grad_b = errors.mean() / max(self.temperature, 1e-6)

                if self.l2:
                    grad_w += self.l2 * self.coef_

                self.coef_ -= self.learning_rate * grad_w
                if self.fit_intercept:
                    self.intercept_ -= self.learning_rate * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.coef_
        if self.fit_intercept:
            logits += self.intercept_
        return np.clip(_sigmoid(logits), 1e-6, 1 - 1e-6)
