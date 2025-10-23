"""High level orchestration for the synthetic data pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._deps import np

from .dp.accountant import DPAccountant, PrivacyBudget
from .features import ColumnMetadata, DPFeatureSynthesizer
from .models import DPLogisticRegression, DistilledLogisticRegression


@dataclass
class DistillationConfig:
    """Configuration options for knowledge distillation."""

    temperature: float = 2.0
    student_epochs: int = 300
    student_batch_size: int = 128
    student_lr: float = 0.05
    student_l2: float = 0.0
    synthetic_batch_size: int = 2048


class PrivFlowPipeline:
    """Build a synthetic dataset with DP guarantees and knowledge distillation."""

    def __init__(
        self,
        *,
        feature_columns: list[ColumnMetadata],
        feature_budget: PrivacyBudget,
        teacher: DPLogisticRegression,
        student: DistilledLogisticRegression,
        total_budget: PrivacyBudget,
        distillation: Optional[DistillationConfig] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.accountant = DPAccountant(total_budget)
        self.feature_synthesizer = DPFeatureSynthesizer(
            feature_columns,
            total_budget=feature_budget,
            random_state=random_state,
        )
        self.teacher = teacher
        self.student = student
        self.distillation = distillation or DistillationConfig()
        self.random_state = random_state
        self._teacher_fitted = False
        self._student_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the feature synthesizer and both models."""

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional.")
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional label vector.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        # Step 1: learn privatized feature marginals
        self.feature_synthesizer.fit(X, accountant=self.accountant)

        # Step 2: train the teacher model with DP-SGD
        self.teacher.fit(X, y, accountant=self.accountant)
        self._teacher_fitted = True

        # Step 3: distill knowledge into the student using synthetic feature batches
        self._distill_student()
        self._student_fitted = True

    def _distill_student(self) -> None:
        synth_batch_size = self.distillation.synthetic_batch_size
        temperature = max(self.distillation.temperature, 1e-6)

        # sample features and get soft labels from the teacher
        synthetic_features = self.feature_synthesizer.sample(synth_batch_size)
        teacher_probs = self.teacher.predict_proba(synthetic_features)

        # calibrate the targets with temperature scaling
        teacher_probs = np.clip(teacher_probs, 1e-6, 1 - 1e-6)
        logits = np.log(teacher_probs / (1.0 - teacher_probs))
        softened_targets = 1.0 / (1.0 + np.exp(-logits / temperature))

        self.student.learning_rate = self.distillation.student_lr
        self.student.epochs = self.distillation.student_epochs
        self.student.batch_size = self.distillation.student_batch_size
        self.student.temperature = temperature
        self.student.l2 = self.distillation.student_l2
        self.student.random_state = self.random_state
        self.student.fit(synthetic_features, softened_targets)

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic dataset of the requested size."""

        if not self._student_fitted:
            raise RuntimeError("Pipeline must be fitted before sampling.")
        features = self.feature_synthesizer.sample(n_samples)
        probs = self.student.predict_proba(features)
        rng = np.random.default_rng(self.random_state)
        labels = rng.binomial(1, probs)
        return features, labels

    def report(self) -> dict[str, float]:
        """Return the privacy budget consumption report."""

        spent = self.accountant.spent
        remaining = self.accountant.remaining
        return {
            "epsilon_spent": spent.epsilon,
            "delta_spent": spent.delta,
            "epsilon_remaining": remaining.epsilon,
            "delta_remaining": remaining.delta,
        }
