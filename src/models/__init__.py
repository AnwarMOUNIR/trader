"""
Multi-Horizon Deep Neural Network (DNN) Module.

Provides:
- Feature Engineering & Normalization: FeatureEngineer, CausalRollingScaler, FeaturePipeline, MultiHorizonDataset, DatasetTensors
- Triple-Barrier Supervisory Labeling: TripleBarrierLabeler, BarrierConfig, LabelResult
- Composite Multi-Horizon Loss Functions: FocalLoss, MultiHorizonHuberLoss, DifferentialSharpeLoss, CompositeMultiHorizonLoss, LossWeights
- Neural Network Architecture: MultiHorizonTCT, ModelConfig, PredictionOutput, CausalTCNBranch, GatedResidualNetwork, TemporalBlock, CausalConv1d
- Training & Inference Engine: DNNTrainer, DNNPredictor, PurgedTimeSeriesSplit, TrainerConfig
"""

from src.models.feature_pipeline import (
    MACRO_FEATURE_COLS,
    MICRO_FEATURE_COLS,
    SENTIMENT_FEATURE_COLS,
    CausalRollingScaler,
    DatasetTensors,
    FeatureEngineer,
    FeaturePipeline,
    MultiHorizonDataset,
)
from src.models.labeler import BarrierConfig, LabelResult, TripleBarrierLabeler
from src.models.loss_functions import (
    CompositeMultiHorizonLoss,
    DifferentialSharpeLoss,
    FocalLoss,
    LossWeights,
    MultiHorizonHuberLoss,
)
from src.models.multi_horizon_dnn import (
    CausalConv1d,
    CausalTCNBranch,
    GatedResidualNetwork,
    ModelConfig,
    MultiHorizonTCT,
    PredictionOutput,
    TemporalBlock,
)
from src.models.trainer import (
    DNNPredictor,
    DNNTrainer,
    PurgedTimeSeriesSplit,
    TrainerConfig,
)

__all__ = [
    # Feature Pipeline
    "MACRO_FEATURE_COLS",
    "MICRO_FEATURE_COLS",
    "SENTIMENT_FEATURE_COLS",
    "CausalRollingScaler",
    "FeatureEngineer",
    "FeaturePipeline",
    "DatasetTensors",
    "MultiHorizonDataset",
    # Labeler
    "BarrierConfig",
    "LabelResult",
    "TripleBarrierLabeler",
    # Loss Functions
    "FocalLoss",
    "MultiHorizonHuberLoss",
    "DifferentialSharpeLoss",
    "CompositeMultiHorizonLoss",
    "LossWeights",
    # Architecture
    "ModelConfig",
    "PredictionOutput",
    "CausalConv1d",
    "TemporalBlock",
    "CausalTCNBranch",
    "GatedResidualNetwork",
    "MultiHorizonTCT",
    # Trainer & Inference
    "PurgedTimeSeriesSplit",
    "TrainerConfig",
    "DNNTrainer",
    "DNNPredictor",
]
