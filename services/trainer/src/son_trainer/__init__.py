"""Training pipeline: stage orchestration, hyperparameters, resume, preferences.

Control flow is fully exercised by tests against a scripted backend. Actual
optimization needs torch / PEFT / TRL and a GPU, and lives behind
:class:`~son_trainer.backends.TrainingBackend`. See AGENTS.md: this machine has
no GPU, so no number produced by this package has been measured.
"""

from son_trainer.backends import (
    StageRequest,
    StageResult,
    TrainingBackend,
    TrainSample,
)
from son_trainer.checkpoints import (
    Checkpoint,
    CheckpointStore,
    JobState,
    LocalCheckpointStore,
    resume_point,
)
from son_trainer.hyperparams import (
    DatasetProfile,
    estimate_duration_minutes,
    recommend_batch_size,
    recommend_hyperparams,
    recommend_learning_rate,
    recommend_max_steps,
)
from son_trainer.pipeline import TrainRequest, TrainRun, resolve_stages, run_training
from son_trainer.preferences import (
    PairBuildReport,
    PreferencePair,
    build_preference_pairs,
    parse_decision,
    to_train_samples,
)

__all__ = [
    "Checkpoint",
    "CheckpointStore",
    "DatasetProfile",
    "JobState",
    "LocalCheckpointStore",
    "PairBuildReport",
    "PreferencePair",
    "StageRequest",
    "StageResult",
    "TrainRequest",
    "TrainRun",
    "TrainSample",
    "TrainingBackend",
    "build_preference_pairs",
    "estimate_duration_minutes",
    "parse_decision",
    "recommend_batch_size",
    "recommend_hyperparams",
    "recommend_learning_rate",
    "recommend_max_steps",
    "resolve_stages",
    "resume_point",
    "run_training",
    "to_train_samples",
]
