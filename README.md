# PrivFlow

PrivFlow provides a lightweight framework for generating synthetic tabular datasets that balance privacy and utility through differential privacy (DP) and knowledge distillation. The package couples DP-aware feature modelling with a teacher-student learning scheme so that the released synthetic samples preserve statistical signal while bounding the risk of leaking individual records.

## Key components

- **Privacy accountant** – tracks usage of the global $(\varepsilon, \delta)$ budget across feature sanitisation and teacher training.
- **DP feature synthesiser** – builds independent marginals per column using either the Gaussian or Laplace mechanism to sanitise summary statistics before sampling synthetic feature vectors.
- **DP teacher model** – a logistic regression trained with DP-SGD that learns predictive structure from the original dataset under noise injection and gradient clipping.
- **Student model** – a distilled logistic regression that mimics the teacher on privatised synthetic features, producing calibrated probabilities used to label the final synthetic dataset.

## Quick start

```python
from privflow.dp.accountant import PrivacyBudget
from privflow.features import ColumnMetadata
from privflow.models import DPLogisticRegression, DistilledLogisticRegression
from privflow.pipeline import PrivFlowPipeline, DistillationConfig
from privflow.data import load_dummy_classification

# Built-in binary classification dataset with three continuous features
X, y = load_dummy_classification()

columns = [
    ColumnMetadata(name="x0", kind="continuous", clip=(-3, 3)),
    ColumnMetadata(name="x1", kind="continuous", clip=(-3, 3)),
    ColumnMetadata(name="x2", kind="continuous", clip=(-3, 3)),
]

feature_budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
teacher_budget = PrivacyBudget(epsilon=3.0, delta=1e-5)
total_budget = PrivacyBudget(epsilon=4.0, delta=2e-5)

teacher = DPLogisticRegression(
    learning_rate=0.15,
    epochs=50,
    batch_size=64,
    clipping_norm=1.0,
    noise_multiplier=1.1,
    privacy_usage=teacher_budget,
)
student = DistilledLogisticRegression()
distillation = DistillationConfig(temperature=2.5, synthetic_batch_size=4096)

pipeline = PrivFlowPipeline(
    feature_columns=columns,
    feature_budget=feature_budget,
    teacher=teacher,
    student=student,
    total_budget=total_budget,
    distillation=distillation,
    random_state=42,
)

pipeline.fit(X, y)
synthetic_X, synthetic_y = pipeline.generate(500)
print(pipeline.report())
```

## Extending the framework

- Swap in alternative synthesiser strategies by implementing new feature modules that respect the `DPAccountant` interface.
- Replace the teacher with any model that exposes a `fit` and `predict_proba` method while consuming privacy budget through the accountant.
- Add richer student architectures (e.g., neural networks) that learn from the teacher's soft predictions.

## Bundled sample data

PrivFlow ships with a `load_dummy_classification` helper that returns a small binary classification dataset containing three continuous features and a binary label. The dataset lives in `privflow/data/dummy_classification.csv` so that end-to-end experiments can run without sourcing external data.

The current implementation focuses on binary classification. The modular design allows extending to multi-class and regression tasks by adapting the teacher/student models and feature synthesiser.
